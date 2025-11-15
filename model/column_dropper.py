from sklearn.base import BaseEstimator, TransformerMixin

# Create a custom pipeline step that will drop correlated columns
# after the one-hot encoding step in the pipeline
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # nothing to fit
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)