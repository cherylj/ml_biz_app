import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

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
cleaned_df = pd.read_csv(dir_data + "/cleaned_telco_data.csv", index_col=0)

# Remove the perfectly correlated features
corr_cols = ['Multiple Lines_No phone service', 'Online Security_No internet service',
             'Online Backup_No internet service', 'Device Protection_No internet service',
             'Tech Support_No internet service', 'Streaming TV_No internet service',
             'Streaming Movies_No internet service']

cleaned_df = cleaned_df.drop(corr_cols, axis=1)

# We also need to drop the two features of lowest importance (also found in our
# Jupyter notebook)
cleaned_df = cleaned_df.drop(['Device Protection_No', 'Contract_One year'], axis=1)

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

numeric_cols = ['Tenure Months', 'Monthly Charges', 'CLTV']


# Specify common parameters for the CatBoost model
params = {'iterations': 500,
    'eval_metric': 'PRAUC',
    'verbose': 100,
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': RANDOM_STATE,
    'loss_function': 'Logloss'
}

# Only scale the numeric columns.  We know that all the numeric columns are
# very high in feature importance, so we don't need to make sure we didn't
# remove them here.
pre = ColumnTransformer(
transformers=[
    ("num", StandardScaler(), numeric_cols)
],
remainder="passthrough"
)

steps = [
    ("prep", pre),
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

best_cb.named_steps["model"].save_model('models/catboost_model.cbm')