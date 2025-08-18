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
def load_config():
    """Loads the config.yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
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
    A helper function to prepare the data payload for a litellm call.
    It handles authentication, configuration lookups, and the Azure workaround.
    Returns the final data for the call and a flag indicating if it's an Azure request.
    """
    is_azure = False

    # 1. Check for API key in the Authorization header
    auth_header = http_request.headers.get("Authorization")
    if auth_header:
        try:
            scheme, token = auth_header.split()
            if scheme.lower() == "bearer" and token:
                request_data["api_key"] = token
        except ValueError:
            pass

    # 2. If no key from header, look for it in the config file
    if "api_key" not in request_data:
        model_name = request_data.get("model")
        model_aliases = config.get("router_settings", {}).get("model_group_alias", {})
        if model_name in model_aliases:
            model_name = model_aliases[model_name]

        model_info = next((m for m in config.get("model_list", []) if m.get("model_name") == model_name), None)

        if model_info:
            litellm_params_from_config = model_info.get("litellm_params", {})
            if litellm_params_from_config.get("model", "").startswith("azure/"):
                is_azure = True
            request_data = {**request_data, **litellm_params_from_config}

    return request_data, is_azure


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Primary endpoint for chat completions.
    Supports both streaming and non-streaming responses.
    Also supports chaining an embedding request.
    """
    request_data = request.model_dump(exclude_none=True)

    # Extract the embedding request details before they are removed
    embedding_model = request_data.pop("embed_with_model", None)

    call_data, is_azure = await _prepare_litellm_call(request_data, http_request)

    response = None
    original_globals = {}

    try:
        if is_azure:
            original_globals = {"api_key": litellm.api_key, "api_base": litellm.api_base, "api_version": litellm.api_version}
            litellm.api_key = call_data.pop("api_key", None)
            litellm.api_base = call_data.pop("end_point", None)
            litellm.api_version = call_data.pop("api_version", None)

        response = await litellm.acompletion(**call_data)

        # After getting a successful response, check if we need to embed it
        if embedding_model and response.choices and not request.stream:
            text_to_embed = response.choices[0].message.content

            embedding_request_data = {"model": embedding_model, "input": text_to_embed}

            # Prepare the embedding call, it might be a different provider
            embedding_call_data, embedding_is_azure = await _prepare_litellm_call(embedding_request_data, http_request)

            embedding_response = None
            embedding_original_globals = {}
            try:
                if embedding_is_azure:
                    embedding_original_globals = {"api_key": litellm.api_key, "api_base": litellm.api_base, "api_version": litellm.api_version}
                    litellm.api_key = embedding_call_data.pop("api_key", None)
                    litellm.api_base = embedding_call_data.pop("end_point", None)
                    litellm.api_version = embedding_call_data.pop("api_version", None)

                embedding_response = await litellm.aembedding(**embedding_call_data)
                # Attach embedding to the original response
                response.choices[0]['embedding'] = embedding_response.data[0]['embedding']

            finally:
                if embedding_is_azure and embedding_original_globals:
                    litellm.api_key = embedding_original_globals["api_key"]
                    litellm.api_base = embedding_original_globals["api_base"]
                    litellm.api_version = embedding_original_globals["api_version"]


    except Exception as e:
        litellm.print_verbose(f"Gateway Error: {e}")
        if isinstance(e, litellm.exceptions.RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        if isinstance(e, litellm.exceptions.AuthenticationError):
             raise HTTPException(status_code=401, detail=str(e))
        if isinstance(e, litellm.exceptions.BadRequestError):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if is_azure and original_globals:
            litellm.api_key = original_globals["api_key"]
            litellm.api_base = original_globals["api_base"]
            litellm.api_version = original_globals["api_version"]

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
    call_data, is_azure = await _prepare_litellm_call(request_data, http_request)

    response = None
    original_globals = {}

    try:
        if is_azure:
            original_globals = {"api_key": litellm.api_key, "api_base": litellm.api_base, "api_version": litellm.api_version}
            litellm.api_key = call_data.pop("api_key", None)
            litellm.api_base = call_data.pop("end_point", None)
            litellm.api_version = call_data.pop("api_version", None)

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
    finally:
        if is_azure and original_globals:
            litellm.api_key = original_globals["api_key"]
            litellm.api_base = original_globals["api_base"]
            litellm.api_version = original_globals["api_version"]

    return response


if __name__ == "__main__":
    import uvicorn
    # This is for local development and testing.
    # In production, you would use a Gunicorn or similar ASGI server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
