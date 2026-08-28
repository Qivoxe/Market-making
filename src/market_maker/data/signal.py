from __future__ import annotations

import numpy as np

from src.market_maker.data.pipeline import build_market_dataset


FEATURE_NAMES = [
    "mid_price",
    "spread",
    "bid_volume",
    "ask_volume",
    "imbalance",
]


def calculate_feature_correlations(
    count_per_regime: int = 1000,
    horizon: int = 1,
    seed: int | None = 42,
) -> dict[str, float]:
    dataset = build_market_dataset(
        count_per_regime=count_per_regime,
        horizon=horizon,
        seed=seed,
    )

    correlations: dict[str, float] = {}

    for index, name in enumerate(FEATURE_NAMES):
        feature = dataset.X[:, index]

        if np.std(feature) == 0 or np.std(dataset.y) == 0:
            correlations[name] = 0.0
        else:
            correlations[name] = float(
                np.corrcoef(feature, dataset.y)[0, 1]
            )

    return correlations