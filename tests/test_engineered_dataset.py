import numpy as np
import pytest

from src.market_maker.data.engineered_dataset import (
    build_engineered_dataset,
)
from src.market_maker.features.order_flow import OrderFlowFeatures


def snapshot(
    mid_price: float,
    spread: float = 1.0,
    bid_volume: float = 100.0,
    ask_volume: float = 100.0,
    imbalance: float = 0.0,
) -> OrderFlowFeatures:
    return OrderFlowFeatures(
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        imbalance=imbalance,
        mid_price=mid_price,
        spread=spread,
    )


def test_build_engineered_dataset():
    snapshots = [
        snapshot(100.0, imbalance=0.2),
        snapshot(101.0, imbalance=0.4),
        snapshot(102.0, imbalance=0.1),
        snapshot(101.0, imbalance=0.3),
    ]

    dataset = build_engineered_dataset(
        snapshots,
        horizon=1,
    )

    assert dataset.X.shape == (2, 7)
    assert dataset.y.shape == (2,)

    np.testing.assert_allclose(
        dataset.X[0],
        [
            101.0,
            1.0,
            100.0,
            100.0,
            0.4,
            0.01,
            0.2,
        ],
    )


def test_engineered_dataset_horizon_two():
    snapshots = [
        snapshot(100.0, imbalance=0.1),
        snapshot(101.0, imbalance=0.2),
        snapshot(104.0, imbalance=0.5),
        snapshot(103.0, imbalance=0.3),
    ]

    dataset = build_engineered_dataset(
        snapshots,
        horizon=2,
    )

    assert dataset.X.shape == (1, 7)
    assert dataset.y.shape == (1,)


def test_invalid_horizon():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
        snapshot(102.0),
    ]

    with pytest.raises(ValueError):
        build_engineered_dataset(
            snapshots,
            horizon=0,
        )


def test_invalid_threshold():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
        snapshot(102.0),
    ]

    with pytest.raises(ValueError):
        build_engineered_dataset(
            snapshots,
            threshold=0.0,
        )


def test_insufficient_snapshots():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
    ]

    with pytest.raises(ValueError):
        build_engineered_dataset(
            snapshots,
            horizon=1,
        )