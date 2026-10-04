import mlflow
import pandas as pd


def compare_runs():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(
        "iris-hyperparameter-tuning"
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment 'iris-hyperparameter-tuning' not found."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"]
    )

    records = []

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")

        if run_name == "baseline_decision_tree":
            cv_f1 = run.data.metrics.get("cv_f1_macro_mean")
            total_fits = 5

        elif run_name == "grid_search_random_forest":
            cv_f1 = run.data.metrics.get("best_cv_f1_macro")
            total_fits = 360

        elif run_name == "random_search_random_forest":
            cv_f1 = run.data.metrics.get("best_cv_f1_macro")
            total_fits = 150

        else:
            continue

        records.append(
            {
                "Run Name": run_name,
                "CV F1 Macro": cv_f1,
                "Test Accuracy": run.data.metrics.get(
                    "test_accuracy"
                ),
                "Total Fits": total_fits,
            }
        )

    comparison = pd.DataFrame(records)

    print("\nComparative Performance Analysis")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    compare_runs()
