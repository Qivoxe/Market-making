from __future__ import annotations

from typing import Any

import numpy as np

from src.market_maker.ml.signal import (
    TradingSignal,
    generate_signal,
)


def predict_signal(
    model: Any,
    mid_price: float,
    spread: float,
    bid_volume: float,
    ask_volume: float,
    imbalance: float,
    threshold: float = 0.60,
) -> TradingSignal:
    if mid_price <= 0:
        raise ValueError(
            "Mid price must be greater than zero."
        )

    if spread < 0:
        raise ValueError(
            "Spread must not be negative."
        )

    if bid_volume < 0:
        raise ValueError(
            "Bid volume must not be negative."
        )

    if ask_volume < 0:
        raise ValueError(
            "Ask volume must not be negative."
        )

    if not -1.0 <= imbalance <= 1.0:
        raise ValueError(
            "Imbalance must be between -1 and 1."
        )

    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            "Threshold must be in the range (0, 1]."
        )

    features = np.asarray(
        [[
            mid_price,
            spread,
            bid_volume,
            ask_volume,
            imbalance,
        ]],
        dtype=float,
    )

    probabilities = model.predict_proba(features)[0]

    classes = model.classes_

    probability_map = {
        int(label): float(probability)
        for label, probability in zip(
            classes,
            probabilities,
        )
    }

    down_probability = probability_map.get(-1, 0.0)
    flat_probability = probability_map.get(0, 0.0)
    up_probability = probability_map.get(1, 0.0)

    return generate_signal(
        down_probability=down_probability,
        flat_probability=flat_probability,
        up_probability=up_probability,
        threshold=threshold,
    )