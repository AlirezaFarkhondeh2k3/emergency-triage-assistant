from dataclasses import dataclass

from .models.classifier import IncidentClassifier
from .models.rag import IncidentRAG
# from .models.transformer_classifier import TransformerIncidentClassifier  # already there if you added it


@dataclass
class TriageResult:
    category: str
    raw_text: str
    guidance: str | None = None


class TriagePipeline:
    def __init__(self, use_transformer: bool = False, use_rag: bool = True):
        # Classifier selection (baseline vs transformer)
        if use_transformer:
            from .models.transformer_classifier import TransformerIncidentClassifier

            self.classifier = TransformerIncidentClassifier()
        else:
            self.classifier = IncidentClassifier()

        # RAG module (can be disabled if ever needed)
        self.rag = IncidentRAG() if use_rag else None

    def run(self, text: str) -> dict:
        category = self.classifier.predict_one(text)

        guidance = None
        if self.rag is not None:
            # pass both the user text and the predicted category
            rag_result = self.rag.retrieve(text=text, category=category)
            guidance = rag_result.guidance

        return {
            "category": category,
            "text": text,
            "guidance": guidance,
        }
