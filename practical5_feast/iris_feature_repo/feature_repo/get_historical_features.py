from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

entity_df = pd.DataFrame(
    {
        "sample_id": [0, 1, 2],
        "event_timestamp": pd.to_datetime(
            [
                "2026-08-15 15:20:02",
                "2026-08-15 15:21:02",
                "2026-08-15 15:22:02",
            ],
            utc=True,
        ),
    }
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal length (cm)",
        "iris_measurements:sepal width (cm)",
        "iris_measurements:petal length (cm)",
        "iris_measurements:petal width (cm)",
        "iris_engineered_features:sepal_area",
        "iris_engineered_features:petal_area",
        "iris_engineered_features:sepal_to_petal_length_ratio",
        "iris_engineered_features:petal_length_bin",
    ],
).to_df()

print("Historical feature retrieval result:")
print(training_df)