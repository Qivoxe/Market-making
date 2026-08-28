import numpy as np
import pytest

from src.market_maker.data.dataset import build_dataset
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


def test_build_dataset():
    snapshots = [
        snapshot(100.0, imbalance=0.2),
        snapshot(101.0, imbalance=0.4),
        snapshot(102.0, imbalance=-0.1),
        snapshot(101.0, imbalance=0.1),
    ]

    dataset = build_dataset(
        snapshots,
        horizon=1,
    )

    assert dataset.X.shape == (3, 5)
    assert dataset.y.shape == (3,)

    np.testing.assert_allclose(
        dataset.X[0],
        [100.0, 1.0, 100.0, 100.0, 0.2],
    )

    np.testing.assert_array_equal(
        dataset.y,
        [1, 1, -1],
    )


def test_horizon_two():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
        snapshot(104.0),
        snapshot(103.0),
    ]

    dataset = build_dataset(
        snapshots,
        horizon=2,
    )

    assert dataset.X.shape == (2, 5)

    np.testing.assert_array_equal(
        dataset.y,
        [1, 1],
    )


def test_invalid_horizon():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
    ]

    with pytest.raises(ValueError):
        build_dataset(
            snapshots,
            horizon=0,
        )


def test_insufficient_snapshots():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
    ]

    with pytest.raises(ValueError):
        build_dataset(
            snapshots,
            horizon=2,
        )


def test_invalid_threshold():
    snapshots = [
        snapshot(100.0),
        snapshot(101.0),
    ]

    with pytest.raises(ValueError):
        build_dataset(
            snapshots,
            horizon=1,
            threshold=0,
        )