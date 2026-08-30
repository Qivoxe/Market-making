import numpy as np
import pytest

from src.market_maker.ml.predictor import predict_signal


class FakeModel:
    classes_ = np.array([-1, 0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

    def predict_proba(self, X):
        assert X.shape == (1, 5)
        return np.asarray(
            [self.probabilities],
            dtype=float,
        )


def test_predict_up_signal():
    model = FakeModel([0.05, 0.15, 0.80])

    signal = predict_signal(
        model=model,
        mid_price=100.0,
        spread=1.0,
        bid_volume=100.0,
        ask_volume=100.0,
        imbalance=0.2,
    )

    assert signal.action == "BUY"
    assert signal.confidence == pytest.approx(0.80)


def test_predict_down_signal():
    model = FakeModel([0.80, 0.15, 0.05])

    signal = predict_signal(
        model=model,
        mid_price=100.0,
        spread=1.0,
        bid_volume=100.0,
        ask_volume=100.0,
        imbalance=-0.2,
    )

    assert signal.action == "SELL"
    assert signal.confidence == pytest.approx(0.80)


def test_predict_hold_signal():
    model = FakeModel([0.10, 0.80, 0.10])

    signal = predict_signal(
        model=model,
        mid_price=100.0,
        spread=1.0,
        bid_volume=100.0,
        ask_volume=100.0,
        imbalance=0.0,
    )

    assert signal.action == "HOLD"
    assert signal.confidence == pytest.approx(0.80)


def test_invalid_mid_price():
    model = FakeModel([0.1, 0.8, 0.1])

    with pytest.raises(ValueError):
        predict_signal(
            model=model,
            mid_price=0.0,
            spread=1.0,
            bid_volume=100.0,
            ask_volume=100.0,
            imbalance=0.0,
        )


def test_invalid_imbalance():
    model = FakeModel([0.1, 0.8, 0.1])

    with pytest.raises(ValueError):
        predict_signal(
            model=model,
            mid_price=100.0,
            spread=1.0,
            bid_volume=100.0,
            ask_volume=100.0,
            imbalance=1.5,
        )