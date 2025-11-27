import os
from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import TriagePipeline


class TriageRequest(BaseModel):
    text: str


# Toggle via env var; default is baseline (TF-IDF)
USE_TRANSFORMER = os.getenv("USE_TRANSFORMER", "false").lower() == "true"

app = FastAPI()
pipeline = TriagePipeline(use_transformer=USE_TRANSFORMER)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TriageRequest):
    result = pipeline.run(req.text)
    return result
