from pathlib import Path
from typing import List

import joblib


class IncidentClassifier:
    def __init__(self, model_dir: Path | None = None):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / "artifacts"

        tfidf_path = model_dir / "tfidf.joblib"
        clf_path = model_dir / "classifier.joblib"

        if not tfidf_path.exists() or not clf_path.exists():
            raise RuntimeError(
                f"Model artifacts not found in {model_dir}. "
                "Run scripts/train_classifier.py first."
            )

        self.vectorizer = joblib.load(tfidf_path)
        self.clf = joblib.load(clf_path)

    def predict_one(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        pred = self.clf.predict(X)[0]
        return str(pred)

    def predict_batch(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        return [str(p) for p in preds]
