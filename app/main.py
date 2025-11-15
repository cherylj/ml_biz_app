# app.py
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

from model.model_features import INPUT_FEATURES
from model.predict import get_proba
from app.schema import CustomerFeatures, PredictBatchResponse, PredictOneResponse

APP_NAME = "churn-api"

app = FastAPI(title=APP_NAME, version="1.0.0")

def _to_dataframe(req: list[CustomerFeatures]) -> pd.DataFrame:
    rows = [row.model_dump(by_alias=True, exclude_none=True) for row in req]
    df = pd.DataFrame(rows)
    df.columns = df.columns.str.replace("_", " ")

    # Otherwise, try to use model feature names (if present).
    missing = [f for f in INPUT_FEATURES if f not in df.columns]
    if missing:
        print(missing)
        raise HTTPException(
            status_code=400,
            detail=(
                "Request is missing features expected by the model: "
                f"{missing}. Either include them or supply 'feature_order'."
            ),
        )
    df = df[INPUT_FEATURES]
    return df


# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "features_known": bool(INPUT_FEATURES)}


@app.post("/predict", response_model=PredictBatchResponse)
def predict_batch(req: list[CustomerFeatures]):
    """
    Returns the churn probability (class 1) for each record.
    """
    try:
        probs = get_proba(_to_dataframe(req))
        if len(probs) != len(req):
            raise HTTPException(
                status_code=500,
                detail="Internal server error: mismatch prediction length",
            )
        return PredictBatchResponse(probabilities=probs)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("foo", exc_info=e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_one", response_model=PredictOneResponse)
def predict(req: CustomerFeatures):
    return PredictOneResponse(probability=predict_batch([req]).probabilities[0])
