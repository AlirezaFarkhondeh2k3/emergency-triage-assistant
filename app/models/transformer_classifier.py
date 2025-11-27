from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer


class TransformerIncidentClassifier:
    """Predicts incident category using sentence-transformer embeddings
    and a sklearn classifier trained in scripts/train_transformer_classifier.py.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        artifact_dir = Path(__file__).resolve().parent / "transformer_artifacts"
        clf_path = artifact_dir / "transformer_classifier.joblib"

        if not clf_path.exists():
            raise RuntimeError(
                f"Transformer classifier artifact not found at {clf_path}. "
                "Run scripts/train_transformer_classifier.py first."
            )

        # Load encoder and classifier
        self.encoder = SentenceTransformer(model_name)
        self.clf = joblib.load(clf_path)

    def predict_one(self, text: str) -> str:
        """Return a single category label for the given text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        emb = self.encoder.encode([text])  # shape (1, 384)
        pred = self.clf.predict(emb)[0]
        return str(pred)
