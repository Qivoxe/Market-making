from __future__ import annotations

import numpy as np


def calculate_mid_returns(
    mid_prices: np.ndarray,
) -> np.ndarray:
    prices = np.asarray(mid_prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("mid_prices must be one-dimensional.")

    if len(prices) < 2:
        raise ValueError(
            "At least two prices are required."
        )

    if not np.all(np.isfinite(prices)):
        raise ValueError(
            "mid_prices must contain only finite values."
        )

    if np.any(prices <= 0):
        raise ValueError(
            "mid_prices must be greater than zero."
        )

    return np.diff(prices) / prices[:-1]


def calculate_imbalance_change(
    imbalance: np.ndarray,
) -> np.ndarray:
    values = np.asarray(imbalance, dtype=float)

    if values.ndim != 1:
        raise ValueError("imbalance must be one-dimensional.")

    if len(values) < 2:
        raise ValueError(
            "At least two imbalance values are required."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "imbalance must contain only finite values."
        )

    return np.diff(values)


def build_market_features(
    mid_prices: np.ndarray,
    imbalances: np.ndarray,
) -> np.ndarray:
    prices = np.asarray(mid_prices, dtype=float)
    imbalance = np.asarray(imbalances, dtype=float)

    if len(prices) != len(imbalance):
        raise ValueError(
            "mid_prices and imbalances must have the same length."
        )

    if len(prices) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    returns = calculate_mid_returns(prices)
    imbalance_change = calculate_imbalance_change(imbalance)

    return np.column_stack(
        (
            prices[1:],
            imbalance[1:],
            returns,
            imbalance_change,
        )
    )
