import numpy as np
import pytest

from src.market_maker.features.market_features import (
    calculate_mid_returns,
)


def test_calculate_mid_returns():
    prices = np.array([
        100.0,
        101.0,
        102.0,
    ])

    returns = calculate_mid_returns(prices)

    np.testing.assert_allclose(
        returns,
        [
            0.01,
            0.009900990099009901,
        ],
    )


def test_returns_shape():
    prices = np.array([
        100.0,
        101.0,
        102.0,
        103.0,
    ])

    returns = calculate_mid_returns(prices)

    assert returns.shape == (3,)


def test_requires_at_least_two_prices():
    with pytest.raises(ValueError):
        calculate_mid_returns(
            np.array([100.0])
        )


def test_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        calculate_mid_returns(
            np.array([100.0, 0.0])
        )


def test_rejects_non_finite_prices():
    with pytest.raises(ValueError):
        calculate_mid_returns(
            np.array([100.0, np.nan])
        )


def test_rejects_multidimensional_input():
    with pytest.raises(ValueError):
        calculate_mid_returns(
            np.array([
                [100.0, 101.0],
                [102.0, 103.0],
            ])
        )