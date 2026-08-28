import numpy as np

from src.market_maker.data.pipeline import build_market_dataset

from src.market_maker.data.pipeline import build_market_dataset


def test_build_market_dataset():
    dataset = build_market_dataset(
        count_per_regime=100,
        horizon=1,
        seed=42,
    )

    assert dataset.X.shape == (499, 5)
    assert dataset.y.shape == (499,)


def test_build_market_dataset_with_horizon():
    dataset = build_market_dataset(
        count_per_regime=100,
        horizon=5,
        seed=42,
    )

    assert dataset.X.shape == (495, 5)
    assert dataset.y.shape == (495,)


def test_dataset_contains_finite_values():
    dataset = build_market_dataset(
        count_per_regime=100,
        horizon=1,
        seed=42,
    )

    assert dataset.X.shape[1] == 5
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.y).all()

    assert (dataset.X[:, 0] > 0).all()
    assert (dataset.X[:, 1] >= 0).all()
    assert (dataset.X[:, 2] >= 0).all()
    assert (dataset.X[:, 3] >= 0).all()
    assert ((dataset.X[:, 4] >= -1) & (dataset.X[:, 4] <= 1)).all()