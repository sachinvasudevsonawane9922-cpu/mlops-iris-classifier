from feast import FeatureStore
import pandas as pd
from sklearn.cluster import KMeans

store = FeatureStore(repo_path=".")

entity_df = pd.DataFrame(
    {
        "sample_id": range(149),
        "event_timestamp": pd.Timestamp(
            "2026-08-15 15:20:02",
            tz="UTC",
        )
        + pd.to_timedelta(range(149), unit="min"),
    }
)

feature_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal length (cm)",
        "iris_measurements:sepal width (cm)",
        "iris_measurements:petal length (cm)",
        "iris_measurements:petal width (cm)",
        "iris_engineered_features:sepal_area",
        "iris_engineered_features:petal_area",
    ],
).to_df()

X = feature_df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "sepal_area",
        "petal_area",
    ]
]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

feature_df["cluster"] = kmeans.fit_predict(X)

print("Clustering completed successfully.")
print(feature_df[["sample_id", "cluster"]].head(10))