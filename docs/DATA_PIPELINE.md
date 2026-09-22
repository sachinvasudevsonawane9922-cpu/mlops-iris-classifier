# Data Pipeline

## Pipeline Stages

| Stage | Purpose | Input | Output |
|---|---|---|---|
| Collect | Obtain raw Iris data | sklearn Iris dataset | iris_raw.csv |
| Preprocess | Clean data | iris_raw.csv | iris_preprocessed.csv |
| Feature Engineering | Create useful features | iris_preprocessed.csv | iris_features.csv |
| Validate | Check schema, nulls and ranges | iris_features.csv | Validation result |

## Pipeline Flow

Collect → Preprocess → Feature Engineering → Validate

## DVC Pipeline

The pipeline is automated using DVC.

Stages:

1. Collect
2. Preprocess
3. Features
4. Validate

The dependencies and outputs are declared in dvc.yaml.

## Validation Rules

The validation stage checks:

- Required columns are present
- No unexpected null values
- Valid species values
- Expected numeric ranges
