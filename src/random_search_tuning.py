import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def load_dataset(path: str):
    df = pd.read_csv(path)

    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def run_random_search(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }

    n_iter = 30

    print(f"Random Search iterations: {n_iter}")
    print(f"Random Search total fits: {n_iter * 5}")

    with mlflow.start_run(run_name="random_search_random_forest"):

        model = RandomForestClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        best_cv_score = random_search.best_score_
        test_score = random_search.best_estimator_.score(
            X_test,
            y_test
        )

        results = pd.DataFrame(random_search.cv_results_)

        results.to_csv(
            "random_search_all_candidates.csv",
            index=False
        )

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )
        mlflow.log_param(
            "search_type",
            "RandomizedSearchCV"
        )
        mlflow.log_param(
            "cv_folds",
            5
        )
        mlflow.log_param(
            "n_iter",
            n_iter
        )
        mlflow.log_param(
            "total_fits",
            n_iter * 5
        )

        mlflow.log_params(
            {
                f"best_{key}": str(value)
                for key, value in random_search.best_params_.items()
            }
        )

        mlflow.log_metric(
            "best_cv_f1_macro",
            best_cv_score
        )

        mlflow.log_metric(
            "test_accuracy",
            test_score
        )

        mlflow.log_artifact(
            "random_search_all_candidates.csv"
        )

        print(f"Best CV f1_macro: {best_cv_score:.4f}")
        print(
            f"Best parameters: "
            f"{random_search.best_params_}"
        )
        print(f"Test accuracy: {test_score:.4f}")
        print(
            f"Random Search total fits: "
            f"{n_iter * 5}"
        )


if __name__ == "__main__":
    run_random_search(
        "data/processed/iris_features.csv"
    )
