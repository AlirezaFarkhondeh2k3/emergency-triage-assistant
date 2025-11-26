from dataclasses import dataclass

from .models.classifier import IncidentClassifier, TransformerIncidentClassifier



@dataclass
class TriageResult:
    category: str
    raw_text: str


class TriagePipeline:
    def __init__(self, use_transformer: bool = False):
        if use_transformer:
            self.classifier = TransformerIncidentClassifier()
        else:
            self.classifier = IncidentClassifier()

    def run(self, text: str) -> dict:
        category = self.classifier.predict_one(text)
        # placeholder until we add extraction and RAG
        return {
            "category": category,
            "text": text,
        }
