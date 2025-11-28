import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture()
def pred_one_request_json():
    with open("test/test_predict_req.json", "r", encoding="utf-8") as f:
        return json.load(f)

def test_healthcheck_success(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["features_known"] 

def test_predict_one_success(client: TestClient, pred_one_request_json):
    resp = client.post("/predict_one", json=pred_one_request_json)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert isinstance(data["probability"], float)

def test_predict_one_invalid(client: TestClient, pred_one_request_json):
    del pred_one_request_json["Gender"]
    resp = client.post("/predict_one", json=pred_one_request_json)
    assert resp.status_code == 422
    data = resp.json()
    assert data["detail"][0]["type"] == "missing"
    assert data["detail"][0]["msg"] == "Field required"