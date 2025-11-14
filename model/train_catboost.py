import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier

from model_features import MULTI_CAT_FEATURES, NUMERIC_FEATURES, FEATURES_TO_DROP
from column_dropper import ColumnDropper

PIPELINE_PATH = "model/models/model_pipeline.joblib"
pipeline = joblib.load(PIPELINE_PATH)
quit()

# Create a helper function to measure the lift of a model
# By default we will assume 10% for the percentage we are
# targeting.
def lift_at_k(y_true, y_scores, k=0.1):
    n = len(y_true)
    cutoff = int(n * k)

    # Sort by predicted churn probability (descending)
    df = pd.DataFrame({"y": y_true, "score": y_scores})
    df = df.sort_values("score", ascending=False)

    # Top K%
    topk = df.head(cutoff)

    # Churn rate in top K%
    churn_topk = topk["y"].mean()

    # Overall churn rate
    churn_base = df["y"].mean()

    # Lift
    return churn_topk / churn_base


def lift_at_k_scorer(estimator, X, y, k=0.1):
    # get proba from pipeline
    p = estimator.predict_proba(X)[:, 1]
    return lift_at_k(y, p, k)

RANDOM_STATE = 42

# --------------------- Data Pre Processing ---------------------
# Read in the cleaned data
dir_data= "../data"
cleaned_df = pd.read_csv(dir_data + "/cleaned_telco_data_no_OHE.csv", index_col=0)

# --------------------- Data Splitting ---------------------
# We'll use a train / test split with 70/30 and stratification of the
# Churn value.
X = cleaned_df.drop(['Churn'], axis=1)
y = cleaned_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=0.3,
      random_state=RANDOM_STATE,
      stratify=y
)

# Specify common parameters for the CatBoost model
params = {'iterations': 500,
    'eval_metric': 'PRAUC',
    'verbose': 100,
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': RANDOM_STATE,
    'loss_function': 'Logloss'
}

# Create our transformer to scale the numeric columns and perform OHE
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), MULTI_CAT_FEATURES)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)
pre.set_output(transform="pandas")

steps = [
    ("prep", pre),
    ("drop", ColumnDropper(FEATURES_TO_DROP)),
    (("model", CatBoostClassifier(**params)))
]

pipe = Pipeline(steps=steps)

param_grid = {
    "model__depth": [4, 6, 8],
    "model__learning_rate": [0.03, 0.05],
    "model__l2_leaf_reg": [3, 5, 7, 10],
    "model__min_data_in_leaf": [1, 25, 50],
    "model__auto_class_weights": ["Balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=lift_at_k_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
    error_score="raise"
)

gs.fit(X_train, y_train)
best_cb = gs.best_estimator_

y_scores = best_cb.predict_proba(X_test)[:, 1]
lift = lift_at_k(y_test, y_scores, k=0.1)

print('   Best params:', gs.best_params_)
print('   Best lift score:', gs.best_score_)
print('   Out of sample lift score:', lift)

feature_names = best_cb.named_steps["prep"].get_feature_names_out()
print(feature_names)

scaler = best_cb.named_steps["prep"].named_transformers_["num"]

print("Means:", scaler.mean_)  
print("Stds:", scaler.scale_) 

joblib.dump(best_cb, "models/model_pipeline.joblib")