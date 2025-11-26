from dataclasses import dataclass

from .models.classifier import IncidentClassifier


@dataclass
class TriageResult:
    category: str
    raw_text: str


class TriagePipeline:
    def __init__(self):
        self.classifier = IncidentClassifier()
        # later: add extractor, RAG engine, etc.

    def run(self, text: str) -> dict:
        category = self.classifier.predict_one(text)
        # placeholder until we add extraction and RAG
        return {
            "category": category,
            "text": text,
        }
