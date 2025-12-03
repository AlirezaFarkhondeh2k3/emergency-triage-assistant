from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_triage_basic_chat_flow():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "There is heavy flooding downtown, water is rising and cars are stuck.",
            }
        ]
    }

    resp = client.post("/triage", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    # exact schema
    assert set(data.keys()) == {
        "reply",
        "category",
        "severity",
        "location",
        "guidance",
        "summary",
    }

    # basic sanity checks
    assert isinstance(data["reply"], str) and data["reply"].strip()
    assert isinstance(data["summary"], str) and data["summary"].strip()
    assert isinstance(data["guidance"], str) and data["guidance"].strip()

    assert data["category"] in ["flood", "fire", "earthquake", "storm", "landslide", "other"]
    assert data["severity"] in ["low", "medium", "high"]
    assert isinstance(data["location"], str)
