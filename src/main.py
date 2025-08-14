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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Primary endpoint for chat completions.
    Supports both streaming and non-streaming responses.
    """
    # Convert Pydantic models to dictionaries for litellm
    data = request.dict(exclude_none=True)
    messages = [msg.dict() for msg in request.messages]
    data["messages"] = messages

    # === Start of New Authentication and Routing Logic ===

    # 1. Check for API key in the Authorization header
    auth_header = http_request.headers.get("Authorization")
    if auth_header:
        try:
            scheme, token = auth_header.split()
            if scheme.lower() == "bearer" and token:
                data["api_key"] = token
        except ValueError:
            # Malformed header, ignore and fall through to config-based key lookup
            pass

    # 2. If no key from header, look for it in the config file
    if "api_key" not in data:
        model_name = data.get("model")

        # Resolve model alias if it exists
        model_aliases = config.get("router_settings", {}).get("model_group_alias", {})
        if model_name in model_aliases:
            model_name = model_aliases[model_name]

        # Find the model's configuration in the model_list
        model_info = next((m for m in config.get("model_list", []) if m.get("model_name") == model_name), None)

        if model_info:
            # Get the entire litellm_params dictionary from the config
            litellm_params_from_config = model_info.get("litellm_params", {})

            # Combine the config params and the request data.
            # The config params (e.g., api_key, end_point, model) take precedence over any request values.
            data = {**data, **litellm_params_from_config}

    # === End of New Logic ===

    try:
        # Asynchronous call to litellm.completion
        response = await litellm.acompletion(**data)

        if request.stream:
            # For streaming responses, return a StreamingResponse
            async def stream_generator():
                async for chunk in response:
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # For non-streaming, return the complete response
            return response

    except Exception as e:
        # Handle exceptions from litellm and return an appropriate error
        # In a production system, you would have more granular error handling
        litellm.print_verbose(f"Gateway Error: {e}")
        # Map litellm exceptions to HTTP status codes
        if isinstance(e, litellm.exceptions.RateLimitError):
            raise HTTPException(status_code=429, detail=str(e))
        if isinstance(e, litellm.exceptions.AuthenticationError):
             raise HTTPException(status_code=401, detail=str(e))
        if isinstance(e, litellm.exceptions.BadRequestError):
             raise HTTPException(status_code=400, detail=str(e))
        # Generic fallback
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # This is for local development and testing.
    # In production, you would use a Gunicorn or similar ASGI server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
