# Feature Store Analysis

## 1. Feature Store Setup

Feast was used to create a feature repository for the Iris dataset.

The feature repository contains:
- Entity: `sample_id`
- Feature View: `iris_measurements`
- Feature View: `iris_engineered_features`
- Feature Service: `iris_feature_service`

## 2. Offline and Online Feature Retrieval

Historical features were successfully retrieved using Feast's historical feature retrieval.

Online features were also successfully retrieved for sample IDs 0, 1, and 2 from the SQLite online store.

## 3. Point-in-Time Correctness

Historical feature retrieval uses event timestamps to retrieve features corresponding to the required point in time. This helps avoid using future feature values during training.

## 4. Feature Reuse

The same registered features were reused for a K-Means clustering task.

Clustering was completed successfully using:
- Sepal length
- Sepal width
- Petal length
- Petal width
- Sepal area
- Petal area

## 5. Benefits Observed

Using a feature store provides:
- Centralized feature definitions
- Reusable features across different ML tasks
- Consistent online and historical feature retrieval
- Reduced duplication of feature engineering
- Better organization of machine-learning features

## 6. Conclusion

The practical demonstrated how Feast can be used to define, register, materialize, retrieve, and reuse machine-learning features for both online and historical use cases.