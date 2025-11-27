from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_triage_basic():
    payload = {"text": "Severe flooding downtown, streets are blocked, people rescued."}
    resp = client.post("/triage", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["text"] == payload["text"]
    assert isinstance(data["category"], str)
    assert data["category"]
