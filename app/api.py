# app/api.py
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import TriageAgent
from app.config import USE_TRANSFORMER as _config_use_transformer

# ---------- Pydantic models ----------

class ChatMessageModel(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatTriageRequest(BaseModel):
    messages: list[ChatMessageModel]

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "There is heavy flooding downtown, water is rising and cars are stuck.",
                    }
                ]
            }
        }


class ChatTriageResponse(BaseModel):
    reply: str
    category: str
    severity: str
    location: Optional[str] = None
    guidance: str
    summary: str

    class Config:
        schema_extra = {
            "example": {
                "reply": "Understood. This looks like a high severity flood situation. Based on your description, this appears to be a high severity flood situation. Stay safe, avoid unnecessary risk, follow instructions from local authorities, and contact emergency services in urgent situations.",
                "category": "flood",
                "severity": "high",
                "location": "downtown",
                "guidance": "Based on your description, this appears to be a high severity flood situation. Stay safe, avoid unnecessary risk, follow instructions from local authorities, and contact emergency services in urgent situations.",
                "summary": "There is heavy flooding downtown, water is rising and cars are stuck.",
            }
        }


# ---------- FastAPI app setup ----------

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the simple web UI."""
    index_path = BASE_DIR / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")


# Decide which classifier to use based on env
_env_use_transformer = os.getenv("USE_TRANSFORMER")
_use_transformer = (
    _env_use_transformer.lower() == "true"
    if _env_use_transformer is not None
    else _config_use_transformer
)

agent = TriageAgent(use_transformer=_use_transformer)


# ---------- Health check ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- Single-turn triage (legacy endpoint) ----------

def _run_chat(req: ChatTriageRequest) -> ChatTriageResponse:
    history = [m.model_dump() for m in req.messages]
    out = agent.run_chat(history)

    return ChatTriageResponse(**out)


@app.post("/triage", response_model=ChatTriageResponse)
def triage(req: ChatTriageRequest) -> ChatTriageResponse:
    """
    Canonical chat triage endpoint for both tests and UI.
    Accepts a list of chat-style messages to align with frontend/tests.
    """
    return _run_chat(req)


# ---------- Multi-turn chat triage (agent) ----------

@app.post("/chat_triage", response_model=ChatTriageResponse)
def chat_triage(req: ChatTriageRequest) -> ChatTriageResponse:
    """
    Agent endpoint.

    Frontend sends the full conversation as a list of messages.
    Backend:
      - builds a summary of the conversation
      - classifies incident type
      - estimates severity and location hint
      - retrieves RAG guidance
      - returns a natural-language reply plus structured fields
    """
    return _run_chat(req)
