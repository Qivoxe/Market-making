from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.market_maker.features.order_flow import OrderFlowFeatures
from src.market_maker.features.market_features import (
    calculate_mid_returns,
    calculate_imbalance_change,
)
from src.market_maker.data.target import create_directional_target


@dataclass(frozen=True)
class EngineeredDataset:
    X: np.ndarray
    y: np.ndarray


def build_engineered_dataset(
    snapshots: Sequence[OrderFlowFeatures],
    horizon: int = 1,
    threshold: float = 0.0001,
) -> EngineeredDataset:
    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if threshold <= 0:
        raise ValueError("Threshold must be greater than zero.")

    if len(snapshots) <= horizon + 1:
        raise ValueError(
            "Not enough snapshots for the selected horizon."
        )

    mid_prices = np.asarray(
        [snapshot.mid_price for snapshot in snapshots],
        dtype=float,
    )

    imbalances = np.asarray(
        [snapshot.imbalance for snapshot in snapshots],
        dtype=float,
    )

    mid_returns = calculate_mid_returns(mid_prices)
    imbalance_changes = calculate_imbalance_change(
        imbalances
    )

    X = []
    returns = []

    for i in range(1, len(snapshots) - horizon):
        current = snapshots[i]
        future = snapshots[i + horizon]

        X.append(
            [
                current.mid_price,
                current.spread,
                current.bid_volume,
                current.ask_volume,
                current.imbalance,
                mid_returns[i - 1],
                imbalance_changes[i - 1],
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

    return EngineeredDataset(
        X=X_array,
        y=y_array,
    )