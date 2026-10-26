import pandas as pd
import numpy as np
import pytest

from src.pipeline.preprocess import preprocess


@pytest.fixture
def valid_raw_df():
    return pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 5.8],
        "sepal width (cm)": [3.5, 3.0, 2.7],
        "petal length (cm)": [1.4, 1.4, 4.1],
        "petal width (cm)": [0.2, 0.2, 1.0],
        "species": ["setosa", "setosa", "versicolor"],
        "collected_at": pd.date_range("2026-01-01", periods=3)
    })


def test_preprocess_removes_duplicates(valid_raw_df, tmp_path):
    df = pd.concat(
        [valid_raw_df, valid_raw_df.iloc[[0]]],
        ignore_index=True
    )

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 3


def test_preprocess_handles_missing_numeric_values(valid_raw_df, tmp_path):
    df = valid_raw_df.copy()
    df.loc[0, "sepal length (cm)"] = np.nan

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isna().sum() == 0


def test_preprocess_handles_missing_target(valid_raw_df, tmp_path):
    df = valid_raw_df.copy()
    df.loc[0, "species"] = np.nan

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["species"].isna().sum() == 0
    assert len(result) == 2


def test_preprocess_handles_empty_dataframe(tmp_path):
    df = pd.DataFrame(columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
        "collected_at"
    ])

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 0
