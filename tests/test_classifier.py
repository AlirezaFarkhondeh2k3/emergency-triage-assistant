from app.models.classifier import IncidentClassifier


def test_incident_classifier_predicts_category():
    clf = IncidentClassifier()
    pred = clf.predict_one("There is heavy flooding and roads are blocked.")
    assert isinstance(pred, str)
    assert len(pred) > 0
