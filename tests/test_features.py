import pandas as pd
import numpy as np

from src.pipeline.features import engineer_features


def create_input_file(tmp_path, df):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.to_csv(input_path, index=False)

    return input_path, output_path


def test_engineer_features_creates_expected_columns(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0]
    })

    input_path, output_path = create_input_file(tmp_path, df)

    result = engineer_features(str(input_path), str(output_path))

    assert "sepal_area" in result.columns
    assert "petal_area" in result.columns
    assert "sepal_to_petal_length_ratio" in result.columns
    assert "petal_length_bin" in result.columns


def test_engineer_features_calculates_areas(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0]
    })

    input_path, output_path = create_input_file(tmp_path, df)

    result = engineer_features(str(input_path), str(output_path))

    assert result["sepal_area"].iloc[0] == 20
    assert result["petal_area"].iloc[0] == 2


def test_engineer_features_calculates_ratio(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0]
    })

    input_path, output_path = create_input_file(tmp_path, df)

    result = engineer_features(str(input_path), str(output_path))

    assert result["sepal_to_petal_length_ratio"].iloc[0] == 2.5


def test_engineer_features_creates_length_bins(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 5.0, 5.0],
        "sepal width (cm)": [3.0, 3.0, 3.0],
        "petal length (cm)": [1.0, 3.0, 6.0],
        "petal width (cm)": [1.0, 1.0, 1.0]
    })

    input_path, output_path = create_input_file(tmp_path, df)

    result = engineer_features(str(input_path), str(output_path))

    assert list(result["petal_length_bin"].astype(str)) == [
        "short", "medium", "long"
    ]


def test_engineer_features_handles_zero_petal_length(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [1.0]
    })

    input_path, output_path = create_input_file(tmp_path, df)

    result = engineer_features(str(input_path), str(output_path))

    assert pd.isna(
        result["sepal_to_petal_length_ratio"].iloc[0]
    )
