from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.market_maker.ml.model import (
    FEATURE_NAMES,
    load_model,
)
from src.market_maker.ml.signal import (
    TradingSignal,
    create_trading_signal,
)


DEFAULT_MODEL_PATH = Path("models/random_forest.joblib")


@dataclass(frozen=True)
class Prediction:
    """
    Complete ML prediction.

    prediction:
        -1 = DOWN
         0 = FLAT
         1 = UP

    directional_score:
        positive -> upward pressure
        negative -> downward pressure
    """

    prediction: int
    down_probability: float
    flat_probability: float
    up_probability: float
    confidence: float
    directional_score: float


def _validate_market_features(
    *,
    mid_price: float,
    spread: float,
    bid_volume: float,
    ask_volume: float,
    imbalance: float,
) -> np.ndarray:
    values = np.asarray(
        [
            mid_price,
            spread,
            bid_volume,
            ask_volume,
            imbalance,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "market features must contain only finite values."
        )

    if mid_price <= 0:
        raise ValueError(
            "mid_price must be greater than zero."
        )

    if spread < 0:
        raise ValueError(
            "spread must be non-negative."
        )

    if bid_volume < 0:
        raise ValueError(
            "bid_volume must be non-negative."
        )

    if ask_volume < 0:
        raise ValueError(
            "ask_volume must be non-negative."
        )

    if not -1.0 <= imbalance <= 1.0:
        raise ValueError(
            "imbalance must be between -1 and 1."
        )

    return values.reshape(1, -1)


def _validate_features(
    features: Sequence[float],
) -> np.ndarray:
    values = np.asarray(
        features,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "features must be a one-dimensional sequence."
        )

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, "
            f"got {len(values)}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "features must contain only finite values."
        )

    return values.reshape(1, -1)


def _probability_map(
    model,
    X: np.ndarray,
) -> dict[int, float]:
    probabilities = np.asarray(
        model.predict_proba(X)[0],
        dtype=float,
    )

    return {
        int(cls): float(probability)
        for cls, probability in zip(
            model.classes_,
            probabilities,
        )
    }


class MLPredictor:
    """
    Wrapper around the trained Random Forest model.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        self.model = load_model(
            self.model_path
        )

        expected_features = len(FEATURE_NAMES)

        actual_features = getattr(
            self.model,
            "n_features_in_",
            expected_features,
        )

        if actual_features != expected_features:
            raise ValueError(
                f"Model expects {actual_features} features, "
                f"but application expects "
                f"{expected_features}."
            )

    def _features(
        self,
        *,
        mid_price: float,
        spread: float,
        bid_volume: float,
        ask_volume: float,
        imbalance: float,
    ) -> np.ndarray:
        return _validate_market_features(
            mid_price=mid_price,
            spread=spread,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
        )

    def predict(
        self,
        *,
        mid_price: float,
        spread: float,
        bid_volume: float,
        ask_volume: float,
        imbalance: float,
    ) -> Prediction:
        X = self._features(
            mid_price=mid_price,
            spread=spread,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
        )

        probability_map = _probability_map(
            self.model,
            X,
        )

        prediction = max(
            probability_map,
            key=probability_map.get,
        )

        down_probability = probability_map.get(
            -1,
            0.0,
        )

        flat_probability = probability_map.get(
            0,
            0.0,
        )

        up_probability = probability_map.get(
            1,
            0.0,
        )

        confidence = probability_map.get(
            prediction,
            0.0,
        )

        directional_score = (
            up_probability
            - down_probability
        )

        return Prediction(
            prediction=int(prediction),
            down_probability=down_probability,
            flat_probability=flat_probability,
            up_probability=up_probability,
            confidence=float(confidence),
            directional_score=float(
                directional_score
            ),
        )

    def predict_signal(
        self,
        *,
        mid_price: float,
        spread: float,
        bid_volume: float,
        ask_volume: float,
        imbalance: float,
        confidence_threshold: float = 0.60,
    ) -> TradingSignal:
        prediction = self.predict(
            mid_price=mid_price,
            spread=spread,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
        )

        return create_trading_signal(
            prediction=prediction.prediction,
            confidence=prediction.confidence,
            threshold=confidence_threshold,
        )


def predict_signal(
    features: Sequence[float] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    confidence_threshold: float = 0.60,
    *,
    model=None,
    threshold: float | None = None,
    mid_price: float | None = None,
    spread: float | None = None,
    bid_volume: float | None = None,
    ask_volume: float | None = None,
    imbalance: float | None = None,
) -> TradingSignal:
    """
    Functional prediction API.

    Supports:

        predict_signal(
            features=[...]
        )

    and:

        predict_signal(
            model=model,
            mid_price=100,
            spread=1,
            bid_volume=100,
            ask_volume=100,
            imbalance=0.2,
        )

    Also supports the backwards-compatible
    ``threshold=...`` keyword used by the backtest
    comparison code.
    """

    # Backwards compatibility:
    #
    # Some parts of the project call:
    #
    #     predict_signal(..., threshold=0.6)
    #
    # while the original API uses:
    #
    #     confidence_threshold=0.6
    #
    # If threshold is supplied, it takes precedence.
    if threshold is not None:
        confidence_threshold = threshold

    if features is None:
        if (
            mid_price is None
            or spread is None
            or bid_volume is None
            or ask_volume is None
            or imbalance is None
        ):
            raise ValueError(
                "Either features or all market features "
                "must be provided."
            )

        X = _validate_market_features(
            mid_price=mid_price,
            spread=spread,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
        )

    else:
        X = _validate_features(
            features
        )

    loaded_model = (
        model
        if model is not None
        else load_model(model_path)
    )

    probability_map = _probability_map(
        loaded_model,
        X,
    )

    prediction = max(
        probability_map,
        key=probability_map.get,
    )

    confidence = probability_map[
        prediction
    ]

    return create_trading_signal(
        prediction=int(prediction),
        confidence=float(confidence),
        threshold=confidence_threshold,
    )


def predict(
    features: Sequence[float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> int:
    """
    Return raw class.

    -1 = DOWN
     0 = FLAT
     1 = UP
    """

    X = _validate_features(
        features
    )

    model = load_model(
        model_path
    )

    probability_map = _probability_map(
        model,
        X,
    )

    return int(
        max(
            probability_map,
            key=probability_map.get,
        )
    )


def predict_probabilities(
    features: Sequence[float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[int, float]:
    X = _validate_features(
        features
    )

    model = load_model(
        model_path
    )

    return _probability_map(
        model,
        X,
    )