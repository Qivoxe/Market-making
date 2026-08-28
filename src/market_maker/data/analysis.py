from __future__ import annotations

import numpy as np

from src.market_maker.data.pipeline import build_market_dataset


def analyze_dataset(
    count_per_regime: int = 1000,
    horizon: int = 1,
    seed: int | None = 42,
) -> dict[str, float]:
    dataset = build_market_dataset(
        count_per_regime=count_per_regime,
        horizon=horizon,
        seed=seed,
    )

    return {
        "samples": float(len(dataset.X)),
        "features": float(dataset.X.shape[1]),
        "mid_price_mean": float(np.mean(dataset.X[:, 0])),
        "mid_price_std": float(np.std(dataset.X[:, 0])),
        "spread_mean": float(np.mean(dataset.X[:, 1])),
        "spread_std": float(np.std(dataset.X[:, 1])),
        "bid_volume_mean": float(np.mean(dataset.X[:, 2])),
        "ask_volume_mean": float(np.mean(dataset.X[:, 3])),
        "imbalance_mean": float(np.mean(dataset.X[:, 4])),
        "imbalance_std": float(np.std(dataset.X[:, 4])),
        "target_mean": float(np.mean(dataset.y)),
        "target_std": float(np.std(dataset.y)),
    }