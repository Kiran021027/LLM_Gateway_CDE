import os
import yaml
import litellm
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

# --- Pydantic Models for OpenAI Compatibility ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    embed_with_model: Optional[str] = None

    class Config:
        extra = "allow"


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None

    class Config:
        extra = "allow"


# --- Configuration Management ---
def validate_api_keys(config: dict):
    """
    Validates the API keys loaded from the configuration.
    Checks for the correct prefixes for OpenAI and Anthropic keys.
    """
    model_list = config.get("model_list", [])
    for model in model_list:
        api_key_source = model.get("litellm_params", {}).get("api_key")
        if not api_key_source:
            continue

        api_key = None
        if api_key_source.startswith("os.environ/"):
            env_var_name = api_key_source.split('/')[-1]
            api_key = os.getenv(env_var_name)
            if not api_key:
                raise HTTPException(
                    status_code=500,
                    detail=f"API key environment variable '{env_var_name}' is not set."
                )
        else:
            # The key is hardcoded in the config file
            api_key = api_key_source

        if not api_key:
            # This case should ideally not be hit if config is well-formed, but as a safeguard:
            raise HTTPException(
                status_code=500,
                detail=f"API key for model '{model.get('model_name')}' is missing or could not be resolved."
            )

        # For non-Azure OpenAI keys, validate the 'sk-' prefix
        is_azure_model = model.get("litellm_params", {}).get("model", "").startswith("azure/")
        if "gpt" in model.get("model_name", "") and not is_azure_model and not api_key.startswith("sk-"):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid OpenAI API key format for model '{model.get('model_name')}'. Key must start with 'sk-'. The provided key starts with '{api_key[:4]}...'"
            )

        # Validate Anthropic keys
        if "claude" in model.get("model_name", "") and not api_key.startswith("sk-ant-"):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid Anthropic API key format for model '{model.get('model_name')}'. Key must start with 'sk-ant-'. The provided key starts with '{api_key[:8]}...'"
            )


def load_config():
    """Loads the config.yaml file and validates API keys."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            validate_api_keys(config_data)  # Validate keys after loading
            return config_data
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="config.yaml not found.")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing config.yaml: {e}")

config = load_config()
import pprint
pprint.pprint(config)

litellm.model_list = config.get("model_list", [])

# --- FastAPI Application ---
app = FastAPI(
    title="Universal Generative AI Gateway",
    description="A centralized, standardized, and governable control plane for all LLM interactions."
)

# --- API Endpoints ---
@app.get("/v1/models")
async def get_models():
    """
    Endpoint to list available models.
    Dynamically generates the list from the gateway's config.
    """
    model_list = config.get("model_list", [])
    model_aliases = config.get("router_settings", {}).get("model_group_alias", {})

    available_models = set()
    for model in model_list:
        available_models.add(model.get("model_name"))

    for alias in model_aliases.keys():
        available_models.add(alias)

    models_data = [
        {
            "id": model_id,
            "object": "model",
            "created": 1, # Placeholder timestamp
            "owned_by": "organization-owner" # Placeholder
        } for model_id in sorted(list(available_models))
    ]

    return {"object": "list", "data": models_data}


async def _prepare_litellm_call(request_data: dict, http_request: Request):
    """
    Prepares the data payload for a litellm call by handling authentication
    and merging model-specific configuration.
    """
    # 1. Check for API key in the Authorization header (Bearer Token)
    auth_header = http_request.headers.get("Authorization")
    if auth_header:
        try:
            scheme, token = auth_header.split()
            if scheme.lower() == "bearer" and token:
                request_data["api_key"] = token
        except ValueError:
            pass  # Ignore malformed headers

    # 2. If no API key from header, merge configuration from config.yaml
    if "api_key" not in request_data:
        model_name = request_data.get("model")
        model_aliases = config.get("router_settings", {}).get("model_group_alias", {})

        # Resolve model alias to the actual model group name
        if model_name in model_aliases:
            model_name = model_aliases[model_name]

        # Find the matching model configuration
        model_info = next((m for m in config.get("model_list", []) if m.get("model_name") == model_name), None)

        if model_info:
            litellm_params = model_info.get("litellm_params", {}).copy()

            # For Azure, 'end_point' from config becomes 'api_base' for litellm
            if "end_point" in litellm_params:
                litellm_params["api_base"] = litellm_params.pop("end_point")

            # Merge the resolved parameters into the request data
            request_data = {**request_data, **litellm_params}

    return request_data


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Primary endpoint for chat completions.
    Supports both streaming and non-streaming responses.
    Also supports chaining an embedding request.
    """
    request_data = request.model_dump(exclude_none=True)
    embedding_model = request_data.pop("embed_with_model", None)

    try:
        # Prepare the primary litellm call
        call_data = await _prepare_litellm_call(request_data, http_request)
        response = await litellm.acompletion(**call_data)

        # If embedding is requested, perform the second call (non-streaming only)
        if embedding_model and response.choices and not request.stream:
            text_to_embed = response.choices[0].message.content
            embedding_request_data = {"model": embedding_model, "input": text_to_embed}

            # Prepare and execute the embedding call
            embedding_call_data = await _prepare_litellm_call(embedding_request_data, http_request)
            embedding_response = await litellm.aembedding(**embedding_call_data)

            # Attach the embedding to the response.
            # We convert the response to a dictionary to add the new key.
            response_dict = response.model_dump()
            if response_dict.get("choices"):
                 response_dict["choices"][0]["embedding"] = embedding_response.data[0]['embedding']

            # The endpoint will now return a dict instead of a ModelResponse object,
            # which FastAPI will serialize to JSON.
            return response_dict

    except Exception as e:
        litellm.print_verbose(f"Gateway Error: {e}")
        if isinstance(e, litellm.exceptions.RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        if isinstance(e, litellm.exceptions.AuthenticationError):
             raise HTTPException(status_code=401, detail=str(e))
        if isinstance(e, litellm.exceptions.BadRequestError):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        async def stream_generator():
            async for chunk in response:
                yield f"data: {chunk.json()}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        return response


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, http_request: Request):
    """
    Endpoint for creating embeddings.
    """
    request_data = request.model_dump(exclude_none=True)

    try:
        call_data = await _prepare_litellm_call(request_data, http_request)
        response = await litellm.aembedding(**call_data)

    except Exception as e:
        litellm.print_verbose(f"Gateway Error: {e}")
        if isinstance(e, litellm.exceptions.RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        if isinstance(e, litellm.exceptions.AuthenticationError):
             raise HTTPException(status_code=401, detail=str(e))
        if isinstance(e, litellm.exceptions.BadRequestError):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return response


if __name__ == "__main__":
    import uvicorn
    # This is for local development and testing.
    # In production, you would use a Gunicorn or similar ASGI server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
