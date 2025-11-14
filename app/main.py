# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from model.model_features import INPUT_FEATURES
from model.predict import get_proba
from app.schema import CustomerFeatures, PredictResponse

APP_NAME = "churn-api"

app = FastAPI(title=APP_NAME, version="1.0.0")
    #df =req.model_dump(by_alias=True))
# ---- Helpers ----
def _to_dataframe(req: CustomerFeatures) -> pd.DataFrame:
    row = req.model_dump(by_alias=True, exclude_none=True)
    df = pd.DataFrame([row])
    df.columns = df.columns.str.replace("_", " ")

    # Otherwise, try to use model feature names (if present).
    missing = [f for f in INPUT_FEATURES if f not in df.columns]
    if missing:
        print(missing)
        raise HTTPException(
            status_code=400,
            detail=("Request is missing features expected by the model: "
                    f"{missing}. Either include them or supply 'feature_order'.")
        )
    df = df[INPUT_FEATURES]
    return df

# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "features_known": bool(INPUT_FEATURES)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: CustomerFeatures):
    """
    Returns the churn probability (class 1) for each record.
    """
    try:
        ###### INSERT CALL TO get_probability here
        probs = get_proba(_to_dataframe(req))
        return PredictResponse(probabilities=probs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


