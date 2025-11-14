import joblib
from model.model_features import YES_NO_FEATURES
import model.column_dropper as column_dropper

PIPELINE_PATH = "model/models/model_pipeline.joblib"
pipeline = joblib.load(PIPELINE_PATH)

def _prep_df(df):
    # As of right now, the only data prep we need to do is to change 
    # our Yes / No columns to 0 / 1
    for col in YES_NO_FEATURES:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

def get_proba(df):
    df = _prep_df(df)
    return pipeline.predict_proba(df)