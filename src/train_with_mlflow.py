import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. MLflow configuration
# =========================================================

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

EXPERIMENT_NAME = "iris-classification-baseline"

mlflow.set_experiment(EXPERIMENT_NAME)


# =========================================================
# 2. Load dataset
# =========================================================

DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(df.head())


# =========================================================
# 3. Select features and target
# =========================================================

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

target_col = "species"

X = df[feature_cols].copy()
y = df[target_col].copy()


# Fill missing values
X = X.fillna(X.median())


# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# =========================================================
# 4. Train-test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)


# =========================================================
# 5. Define models
# =========================================================

models = [
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=200,
            C=1.0
        ),
        {
            "model_type": "LogisticRegression",
            "max_iter": 200,
            "C": 1.0,
        },
    ),

    (
        "random_forest_shallow",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
        ),
        {
            "model_type": "RandomForest",
            "n_estimators": 50,
            "max_depth": 3,
            "random_state": 42,
        },
    ),

    (
        "random_forest_deep",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        ),
        {
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": "None",
            "random_state": 42,
        },
    ),
]


# =========================================================
# 6. Train and track models
# =========================================================

results = []

for run_name, model, params in models:

    with mlflow.start_run(run_name=run_name):

        print("\n" + "=" * 60)
        print(f"Training: {run_name}")
        print("=" * 60)

        # Log parameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Precision Macro: {precision:.4f}")
        print(f"Recall Macro:    {recall:.4f}")
        print(f"F1 Macro:        {f1:.4f}")

        # =================================================
        # Confusion Matrix
        # =================================================

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_,
        )

        disp.plot(ax=ax)

        ax.set_title(f"Confusion Matrix - {run_name}")

        plt.tight_layout()

        cm_filename = f"confusion_matrix_{run_name}.png"

        plt.savefig(cm_filename)

        plt.close()

        # Log artifact
        mlflow.log_artifact(cm_filename)

        # Delete local artifact
        if os.path.exists(cm_filename):
            os.remove(cm_filename)

        # =================================================
        # Log model
        # =================================================

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        # Store result
        results.append(
            {
                "run_name": run_name,
                "run_id": mlflow.active_run().info.run_id,
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )


# =========================================================
# 7. Model comparison
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "run_name",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ].to_string(index=False)
)


# =========================================================
# 8. Select best model
# =========================================================

best_result = results_df.loc[
    results_df["f1_macro"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model:    {best_result['run_name']}")
print(f"Run ID:   {best_result['run_id']}")
print(f"F1 Score: {best_result['f1_macro']:.4f}")

print("\nMLflow experiment:")
print(EXPERIMENT_NAME)