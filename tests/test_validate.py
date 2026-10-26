import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError


def valid_df():
    return pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
        "sepal_area": [17.85],
        "petal_area": [0.28],
        "sepal_to_petal_length_ratio": [3.642857],
        "petal_length_bin": ["short"]
    })


def test_validate_accepts_valid_data(tmp_path):
    df = valid_df()
    input_path = tmp_path / "valid.csv"
    df.to_csv(input_path, index=False)

    result = validate(str(input_path))

    assert len(result) == 1


def test_validate_fails_on_missing_feature(tmp_path):
    df = valid_df().drop(columns=["sepal_area"])
    input_path = tmp_path / "invalid.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_fails_on_invalid_species(tmp_path):
    df = valid_df()
    df.loc[0, "species"] = "unicorn"

    input_path = tmp_path / "invalid.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_fails_on_out_of_range_value(tmp_path):
    df = valid_df()
    df.loc[0, "sepal length (cm)"] = 50.0

    input_path = tmp_path / "invalid.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(
        DataValidationError,
        match="Validation failed"
    ):
        validate(str(input_path))
