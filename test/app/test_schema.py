import pytest
import json

from pydantic import ValidationError

import app.schema as asch


@pytest.fixture()
def pred_one_request_json():
    with open("test/data/test_predict_req.json", "r", encoding="utf-8") as f:
        return json.load(f) 


def test_internet_service_validation_failure(pred_one_request_json: dict) -> None:
    mod_json = pred_one_request_json
    mod_json["Internet_Service"] = "No"
    with pytest.raises(ValidationError) as excinfo:
        asch.CustomerFeatures(**mod_json)

    assert "Invalid configuration for internet service" in str(excinfo.value)

    # Also check that there aren't any "No internet service" fields if
    # InternetService is available.
    mod_json["Internet_Service"] = "DSL"
    internet_service_dep_fields = [
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
        "Streaming_Movies",
    ]
    for field in internet_service_dep_fields:
        mod_json_copy = mod_json.copy()
        mod_json_copy[field] = "No internet service"
        with pytest.raises(ValidationError) as excinfo:
            asch.CustomerFeatures(**mod_json_copy)
            pytest.fail(f"Expected exception for field: {field}")

        assert "Invalid configuration for internet service" in str(
            excinfo.value
        ), f"Failed check for field {field}"


def test_phone_service_validation_failure(pred_one_request_json: dict) -> None:
    mod_json = pred_one_request_json
    mod_json["Phone_Service"] = "No"
    with pytest.raises(ValidationError) as excinfo:
        asch.CustomerFeatures(**mod_json)

    assert "Invalid configuration for phone service" in str(excinfo.value)

    # Also check that there aren't any "No phone service" fields if
    # PhoneService is available.
    mod_json["Phone_Service"] = "Yes"
    phone_service_dep_fields = ["Multiple_Lines"]

    for field in phone_service_dep_fields:
        mod_json_copy = mod_json.copy()
        mod_json_copy[field] = "No phone service"
        with pytest.raises(ValidationError) as excinfo:
            asch.CustomerFeatures(**mod_json_copy)
            pytest.fail(f"Expected exception for field: {field}")

        assert "Invalid configuration for phone service" in str(
            excinfo.value
        ), f"Failed check for field {field}"
