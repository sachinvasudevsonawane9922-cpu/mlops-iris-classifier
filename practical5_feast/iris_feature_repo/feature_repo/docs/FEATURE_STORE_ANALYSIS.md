# Feature Store Analysis

## Observed Benefits

### 1. Centralized Feature Management
Feast provides a centralized repository for defining and registering ML features. In this practical, the original Iris measurements and engineered features were registered as Feature Views.

### 2. Online Feature Retrieval
The SQLite online store provides fast retrieval of the latest feature values for inference. Online feature retrieval was successfully demonstrated using the registered Iris features.

### 3. Historical Feature Retrieval
Feast supports historical feature retrieval using event timestamps. Historical features were successfully retrieved for multiple samples, demonstrating point-in-time feature retrieval for training.

### 4. Feature Reusability
The registered features can be reused by different ML workflows without recreating the feature engineering logic. The clustering demonstration successfully reused the stored features.

### 5. Reduced Training-Serving Skew
The same registered feature definitions are used for both online and historical retrieval, helping maintain consistency between training and inference.

### 6. Reduced Duplicate Feature Engineering
Once features are registered in Feast, other models can reuse them instead of implementing the same feature transformations again.

## Conclusion

This practical demonstrated how Feast can be used as a Feature Store in an MLOps workflow. The feature repository was created successfully, features were registered and materialized into the online store, and both online and historical feature retrieval were demonstrated. Feature reusability was also demonstrated using a clustering workflow.
