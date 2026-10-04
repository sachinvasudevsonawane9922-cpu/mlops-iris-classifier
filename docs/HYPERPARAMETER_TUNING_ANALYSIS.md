# Hyperparameter Tuning Analysis

## 1. Objective

The objective of this experiment was to develop a baseline model and compare systematic hyperparameter tuning methods using Grid Search and Random Search.

The experiment used 5-fold cross-validation with `f1_macro` scoring and tracked all experiments using MLflow.

---

## 2. Baseline Model

The baseline model was:

- Model: DecisionTreeClassifier
- Hyperparameters: Default
- Cross-validation: 5-fold
- Scoring: F1 Macro
- MLflow Run: `baseline_decision_tree`

### Baseline Results

| Metric | Result |
|---|---:|
| CV F1 Macro | 0.9663 |
| Test Accuracy | 0.9000 |
| Total Fits | 5 |

---

## 3. Grid Search

Grid Search was performed using Random Forest with the following parameter space:

- `n_estimators`: 50, 100, 200
- `max_depth`: 3, 5, 10, None
- `min_samples_split`: 2, 5, 10
- `max_features`: sqrt, log2

This resulted in:

- 72 parameter combinations
- 5-fold cross-validation
- 360 total fits

### Best Parameters

```text
n_estimators = 50
max_depth = 3
min_samples_split = 2
max_features = sqrt
