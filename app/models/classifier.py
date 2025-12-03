from dataclasses import dataclass
from typing import Dict
import re

import joblib
import numpy as np

from app.config import (
    ARTIFACTS_DIR,
    TRANSFORMER_ARTIFACTS_DIR,
    USE_TRANSFORMER,
    CRISIS_CATEGORIES,
)


@dataclass
class ClassificationResult:
    category: str
    severity: str
    location: str | None = None
    raw: Dict | None = None


class CrisisClassifier:
    """
    Wraps baseline (TF-IDF + linear) and optional transformer classifier.

    We classify on raw user text (handled in TriagePipeline) and apply
    keyword-based overrides for obvious cases like flood/fire/etc.
    """

    def __init__(self):
        # baseline: tfidf + linear classifier
        baseline_model_path = ARTIFACTS_DIR / "classifier.joblib"
        tfidf_path = ARTIFACTS_DIR / "tfidf.joblib"
        self.tfidf = joblib.load(tfidf_path)
        self.baseline_clf = joblib.load(baseline_model_path)

        # transformer classifier (already trained by you)
        transformer_path = TRANSFORMER_ARTIFACTS_DIR / "transformer_classifier.joblib"
        self.transformer_clf = None
        if transformer_path.exists():
            self.transformer_clf = joblib.load(transformer_path)

        # decide if we should use transformer at runtime
        self.use_transformer = bool(USE_TRANSFORMER and self.transformer_clf is not None)

    # --------- model predictors --------- #

    def _predict_category_baseline(self, text: str) -> str:
        X = self.tfidf.transform([text])
        y_pred = self.baseline_clf.predict(X)[0]
        return str(y_pred)

    def _predict_category_transformer(self, text: str) -> str:
        """
        This assumes transformer_classifier.joblib accepts raw text features.
        If it expects embeddings, adapt this method accordingly.
        """
        X = np.array([text], dtype=object).reshape(1, -1)
        y_pred = self.transformer_clf.predict(X)[0]
        return str(y_pred)

    # --------- rule-based overrides --------- #

    def _apply_category_overrides(self, text: str, model_category: str) -> tuple[str, str]:
        """
        If the model prediction is bad or outside CRISIS_CATEGORIES, try to recover
        with simple keyword rules. Returns (final_category, source).

        source is one of: "model", "keyword_override", "fallback".
        """
        lowered = text.lower()

        # Flood-specific detection (common home/basement phrases)
        water_basement = re.search(r"\bwater\b.*\bbasement\b", lowered) or re.search(
            r"\bbasement\b.*\bwater\b", lowered
        )
        if (
            any(k in lowered for k in ["flood", "flooded", "flooding"])
            or water_basement
            or ("water" in lowered and any(h in lowered for h in ["house", "home", "apartment"]))
        ):
            return "flood", "keyword_override"

        # Smoke-only incidents -> fire
        smoke_terms = [
            "heavy smoke",
            "thick smoke",
            "smoke in my apartment",
            "smoke everywhere",
            "smoke in the building",
            "smoke coming from",
        ]
        if any(term in lowered for term in smoke_terms):
            return "fire", "keyword_override"

        # Only trust the model if it predicts a specific crisis category != "other"
        if model_category in CRISIS_CATEGORIES and model_category != "other":
            return model_category, "model"

        # Keyword-based overrides for obvious situations
        if any(k in lowered for k in ["flood", "flooding", "water is rising", "water rising", "river overflow"]):
            return "flood", "keyword_override"

        if any(k in lowered for k in ["earthquake", "tremor", "strong shaking", "ground shaking"]):
            return "earthquake", "keyword_override"

        if any(k in lowered for k in ["fire", "burning", "wildfire", "smoke everywhere"]):
            return "fire", "keyword_override"

        # If model said "other" and we found no better match, keep "other"
        return "other", "fallback"

    def _infer_severity(self, text: str) -> str:
        """
        Simple rule-based severity. Replace with a dedicated model if needed.
        """
        lowered = text.lower()

        # Strong indicators of life-threatening danger
        high_keywords = [
            "trapped",
            "cannot breathe",
            "can't breathe",
            "unconscious",
            "house destroyed",
            "building collapsed",
            "roof collapsed",
            "severe bleeding",
            "heart attack",
            "no pulse",
        ]
        if any(k in lowered for k in high_keywords):
            return "high"

        # Clear emergency but not obviously life-threatening
        medium_keywords = [
            "water rising",
            "water is rising",
            "rising water",
            "road blocked",
            "roads are blocked",
            "cars are stuck",
            "car is stuck",
            "injured",
            "injuries",
            "need help",
            "need assistance",
            "cannot leave the building",
            "can't leave the building",
        ]
        if any(k in lowered for k in medium_keywords):
            return "medium"

        return "low"

    # --------- public API --------- #

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the incident text into (category, severity), with keyword overrides.
        """

        # Prefer transformer if configured and loaded, otherwise baseline
        if self.use_transformer:
            try:
                category_raw = self._predict_category_transformer(text)
                model_name = "transformer"
            except Exception:
                category_raw = self._predict_category_baseline(text)
                model_name = "baseline_fallback"
        else:
            category_raw = self._predict_category_baseline(text)
            model_name = "baseline"

        category_norm, category_source = self._apply_category_overrides(text, category_raw)
        severity = self._infer_severity(text)

        return ClassificationResult(
            category=category_norm,
            severity=severity,
            location=None,
            raw={
                "model": model_name,
                "raw_label": category_raw,
                "category_source": category_source,
            },
        )
