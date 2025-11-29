import json
import pytest
import pandas as pd

from typing import Any
from unittest.mock import MagicMock
from httpx import Response
from fastapi.testclient import TestClient

from pytest_mock import MockerFixture
from app.main import app
from model.model_features import INPUT_FEATURES

SINGLE_PROB = 0.9
BATCH_PROB = [0.9, 0.1]


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def pred_one_request_json():
    with open("test/data/test_predict_req.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def pred_batch_request_json():
    with open("test/data/test_batch_predict_req.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def mock_get_proba(mocker: MockerFixture) -> MagicMock:
    mock: MagicMock = mocker.patch("app.main.get_proba")
    return mock


@pytest.fixture()
def mock_get_proba_one(mock_get_proba: MagicMock) -> MagicMock:
    mock_get_proba.return_value = [SINGLE_PROB]
    return mock_get_proba


@pytest.fixture()
def mock_get_proba_batch_two(mock_get_proba: MagicMock) -> MagicMock:
    mock_get_proba.return_value = BATCH_PROB
    return mock_get_proba


def test_healthcheck_success(client: TestClient) -> None:
    resp: Response = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["features_known"]


def _get_df_from_json(from_json: list[dict[str, Any]]) -> pd.DataFrame:
    called_df = pd.DataFrame(from_json)
    called_df.columns = called_df.columns.str.replace("_", " ")
    called_df = called_df[INPUT_FEATURES]
    return called_df


def test_predict_one_success(
    client: TestClient,
    pred_one_request_json,
    mock_get_proba_one: MagicMock,
) -> None:
    resp = client.post("/predict_one", json=pred_one_request_json)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert isinstance(data["probability"], float)
    assert data["probability"] == SINGLE_PROB

    called_df = _get_df_from_json([pred_one_request_json])
    called_with_df = mock_get_proba_one.call_args.args[0]
    mock_get_proba_one.assert_called_once()
    assert called_df.equals(called_with_df)


def test_predict_one_invalid(client: TestClient, pred_one_request_json) -> None:
    del pred_one_request_json["Gender"]
    resp = client.post("/predict_one", json=pred_one_request_json)
    assert resp.status_code == 422
    data = resp.json()
    assert data["detail"][0]["type"] == "missing"
    assert data["detail"][0]["msg"] == "Field required"


def test_batch_predict_success(
    client: TestClient,
    pred_batch_request_json,
    mock_get_proba_batch_two: MagicMock,
) -> None:
    resp = client.post("/predict", json=pred_batch_request_json)
    assert resp.status_code == 200
    data = resp.json()
    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)
    assert all(
        [isinstance(x, float) for x in data["probabilities"]]
    ), "List is not made up of floats"
    assert data["probabilities"] == BATCH_PROB

    called_df = _get_df_from_json(pred_batch_request_json)
    called_with_df = mock_get_proba_batch_two.call_args.args[0]
    mock_get_proba_batch_two.assert_called_once()
    assert called_df.equals(called_with_df)


def test_batch_predict_invalid(client: TestClient, pred_batch_request_json) -> None:
    del pred_batch_request_json[0]["Gender"]
    resp = client.post("/predict", json=pred_batch_request_json)
    assert resp.status_code == 422
    data = resp.json()
    assert data["detail"][0]["type"] == "missing"
    assert data["detail"][0]["msg"] == "Field required"


def test_batch_predict_integration(client: TestClient, pred_batch_request_json) -> None:
    resp = client.post("/predict", json=pred_batch_request_json)
    assert resp.status_code == 200
    data = resp.json()
    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)
    assert all(
        [isinstance(x, float) for x in data["probabilities"]]
    ), "List is not made up of floats"


def test_predict_one_integration(client: TestClient, pred_one_request_json):
    resp = client.post("/predict_one", json=pred_one_request_json)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert isinstance(data["probability"], float)