from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.market_maker.features.order_flow import OrderFlowFeatures
from src.market_maker.data.target import create_directional_target


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray


def build_dataset(
    snapshots: Sequence[OrderFlowFeatures],
    horizon: int = 1,
    threshold: float = 0.0001,
) -> Dataset:
    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if threshold <= 0:
        raise ValueError("Threshold must be greater than zero.")

    if len(snapshots) <= horizon:
        raise ValueError(
            "Not enough snapshots for the selected horizon."
        )

    X = []
    returns = []

    for i in range(len(snapshots) - horizon):
        current = snapshots[i]
        future = snapshots[i + horizon]

        X.append(
            [
                current.mid_price,
                current.spread,
                current.bid_volume,
                current.ask_volume,
                current.imbalance,
            ]
        )

        future_return = (
            future.mid_price - current.mid_price
        ) / current.mid_price

        returns.append(future_return)

    X_array = np.asarray(X, dtype=float)
    returns_array = np.asarray(returns, dtype=float)

    y_array = create_directional_target(
        returns_array,
        threshold=threshold,
    )

    return Dataset(
        X=X_array,
        y=y_array,
    )