import os
from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import TriagePipeline


class TriageRequest(BaseModel):
    text: str


app = FastAPI()

# Read from environment: USE_TRANSFORMER=true / false
_use_transformer = os.getenv("USE_TRANSFORMER", "false").lower() == "true"
pipeline = TriagePipeline(use_transformer=_use_transformer)



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TriageRequest):
    result = pipeline.run(req.text)
    return result
