from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import TriagePipeline


class TriageRequest(BaseModel):
    text: str


app = FastAPI()
pipeline = TriagePipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TriageRequest):
    result = pipeline.run(req.text)
    return result
