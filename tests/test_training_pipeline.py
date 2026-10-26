import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_model_trains_without_error():
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)

    assert hasattr(model, "predict")


def test_model_prediction_shape():
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)


def test_model_prediction_probabilities():
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
