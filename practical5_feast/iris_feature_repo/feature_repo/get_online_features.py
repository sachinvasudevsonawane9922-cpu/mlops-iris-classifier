from feast import FeatureStore

store = FeatureStore(repo_path=".")

sample_ids = [0, 1, 2]

features = store.get_online_features(
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
    entity_rows=[
        {"sample_id": sample_id}
        for sample_id in sample_ids
    ],
).to_dict()

print("Online feature retrieval result:")
for key, value in features.items():
    print(f"{key}: {value}")