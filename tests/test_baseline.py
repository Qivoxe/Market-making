import numpy as np
import pytest

from src.market_maker.ml.baseline import BaselineClassifier


def make_dataset():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(300, 5))

    y = np.where(
        X[:, 0] > 0.5,
        1,
        np.where(X[:, 0] < -0.5, -1, 0),
    )

    return X, y


def test_baseline_classifier_trains():
    X, y = make_dataset()

    classifier = BaselineClassifier()

    result = classifier.fit(X, y)

    assert result.model is classifier.model
    assert result.scaler is classifier.scaler
    assert result.predictions.shape[0] == 60
    assert result.probabilities.shape[0] == 60
    assert result.probabilities.shape[1] == 3


def test_baseline_classifier_metrics_are_valid():
    X, y = make_dataset()

    classifier = BaselineClassifier()

    result = classifier.fit(X, y)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.balanced_accuracy <= 1.0


def test_confusion_matrix_shape():
    X, y = make_dataset()

    classifier = BaselineClassifier()

    result = classifier.fit(X, y)

    assert result.confusion.shape == (3, 3)


def test_predict_after_training():
    X, y = make_dataset()

    classifier = BaselineClassifier()

    classifier.fit(X, y)

    predictions = classifier.predict(X[:10])

    assert predictions.shape == (10,)
    assert set(predictions).issubset({-1, 0, 1})


def test_predict_proba_after_training():
    X, y = make_dataset()

    classifier = BaselineClassifier()

    classifier.fit(X, y)

    probabilities = classifier.predict_proba(X[:10])

    assert probabilities.shape == (10, 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(10),
    )


def test_mismatched_lengths():
    classifier = BaselineClassifier()

    X = np.ones((10, 5))
    y = np.ones(9)

    with pytest.raises(ValueError):
        classifier.fit(X, y)


def test_invalid_test_size():
    with pytest.raises(ValueError):
        BaselineClassifier(test_size=0)

    with pytest.raises(ValueError):
        BaselineClassifier(test_size=1)