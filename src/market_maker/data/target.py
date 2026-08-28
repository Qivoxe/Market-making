from __future__ import annotations

import numpy as np


def create_directional_target(
    returns: np.ndarray,
    threshold: float = 0.0001,
) -> np.ndarray:
    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    target = np.zeros(len(returns), dtype=np.int64)

    target[returns > threshold] = 1
    target[returns < -threshold] = -1

    return target