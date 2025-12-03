from app.models.classifier import CrisisClassifier
from app.config import CRISIS_CATEGORIES


def test_crisis_classifier_predicts_category_and_severity():
    clf = CrisisClassifier()

    text = (
        "There is heavy flooding in the city, roads are blocked and "
        "people need help near the river."
    )

    result = clf.classify(text)

    # Category should be one of the configured crisis labels
    assert result.category in CRISIS_CATEGORIES

    # Severity should be normalized to one of these
    assert result.severity in ["low", "medium", "high"]

    # Location can be None or a non-empty string
    assert result.location is None or isinstance(result.location, str)
    if isinstance(result.location, str):
        assert result.location.strip() != ""

    # Optionally: raw metadata from the model/LLM is allowed to be anything,
    # but we can at least assert the attribute exists
    assert hasattr(result, "raw")
