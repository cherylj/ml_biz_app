import json
from locust import HttpUser, between, task


with open("test/data/test_predict_req.json", "r", encoding="utf-8") as f:
    PREDICT_ONE_JSON = json.load(f)

with open("test/data/test_batch_predict_req.json", "r", encoding="utf-8") as f:
    PREDICT_BATCH_JSON = json.load(f)

with open("test/data/test_batch_predict_25_req.json", "r", encoding="utf-8") as f:
    PREDICT_LARGE_BATCH_JSON = json.load(f)


class TelcoChurnUser_Small(HttpUser):
    """
    Simulated user that:
      - Checks /health occasionally
      - Calls /predict_one with a single customer payload
      - Calls /predict with a batch of 2 predictions
    """

    wait_time = between(0.5, 3)
    host = "http://localhost:8000"

    @task(3)
    def predict_one(self):
        # /predict_one expects a single CustomerFeatures JSON object
        self.client.post("/predict_one", json=PREDICT_ONE_JSON)

    @task(3)
    def predict_batch(self):
        # call /predict with two CustomerFeatures JSON objects
        self.client.post("/predict", json=PREDICT_BATCH_JSON)

    @task(1)
    def health_check(self):
        # Hit the health endpoint sometimes too
        self.client.get("/health")


class TelcoChurnUser_Large(HttpUser):
    """
    Simulated user that:
      - Calls /predict with a batch of 25 predictions frequently
    """

    wait_time = between(0.5, 3)
    host = "http://localhost:8000"

    @task(3)
    def predict_batch(self):
        # call /predict with 25 CustomerFeatures JSON objects
        self.client.post("/predict", json=PREDICT_LARGE_BATCH_JSON)
