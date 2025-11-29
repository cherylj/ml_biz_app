from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np

from pytest_mock import MockerFixture

from model.model_features import YES_NO_FEATURES
from model.predict import get_proba


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    all_rows = []
    for row in [0, 1]:
        counter = 0
        df_dict = {"test_field_1": "foo", "test_field_2": "bar"}

        for f in YES_NO_FEATURES:
            df_dict[f] = "No" if (counter % 2 == row) else "Yes"
            counter = counter + 1
        all_rows.append(df_dict)

    return pd.DataFrame(all_rows)


def test_get_proba_success(test_df: pd.DataFrame, mocker: MockerFixture) -> None:
    return_probs = np.array([[0.15, 0.85], [0.4, 0.6]])

    pipeline_mock: MagicMock = mocker.patch("model.predict.pipeline.predict_proba")
    pipeline_mock.return_value = return_probs

    # We need to make a copy since the get_proba function will modify the df
    # that you pass in
    pre_call_df = test_df.copy()

    probabilities = get_proba(test_df)
    assert len(probabilities) == 2
    assert probabilities[0] == 0.85
    assert probabilities[1] == 0.6

    pipeline_mock.assert_called_once()
    called_with_df = pipeline_mock.call_args.args[0]

    assert type(called_with_df) == pd.DataFrame

    # Check that the Yes / No conversion worked as expected
    for row in [0, 1]:
        test_row = pre_call_df.iloc[row]
        called_row = called_with_df.iloc[row]
        for f in YES_NO_FEATURES:
            assert called_row[f] == (
                0 if test_row[f] == "No" else 1
            ), f"Failed to check int conversion for feature {f}"
