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

