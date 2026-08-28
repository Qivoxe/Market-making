import numpy as np
import pytest

from src.market_maker.features.market_features import (
    calculate_mid_returns,
    calculate_imbalance_change,
    build_market_features,
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


def test_calculate_imbalance_change():
    imbalance = np.array([
        0.1,
        0.3,
        0.2,
        -0.1,
    ])

    change = calculate_imbalance_change(imbalance)

    np.testing.assert_allclose(
        change,
        [
            0.2,
            -0.1,
            -0.3,
        ],
    )


def test_imbalance_change_shape():
    imbalance = np.array([
        -0.2,
        0.0,
        0.4,
    ])

    change = calculate_imbalance_change(imbalance)

    assert change.shape == (2,)


def test_imbalance_change_requires_two_values():
    with pytest.raises(ValueError):
        calculate_imbalance_change(
            np.array([0.2])
        )


def test_imbalance_change_rejects_non_finite_values():
    with pytest.raises(ValueError):
        calculate_imbalance_change(
            np.array([0.1, np.nan])
        )


def test_imbalance_change_rejects_multidimensional_input():
    with pytest.raises(ValueError):
        calculate_imbalance_change(
            np.array([
                [0.1, 0.2],
                [0.3, 0.4],
            ])
        )


def test_build_market_features():
    prices = np.array([
        100.0,
        101.0,
        102.0,
    ])

    imbalance = np.array([
        0.2,
        0.4,
        0.1,
    ])

    features = build_market_features(
        prices,
        imbalance,
    )

    assert features.shape == (2, 4)

    np.testing.assert_allclose(
        features[0],
        [
            101.0,
            0.4,
            0.01,
            0.2,
        ],
    )


def test_build_market_features_requires_matching_lengths():
    with pytest.raises(ValueError):
        build_market_features(
            np.array([100.0, 101.0]),
            np.array([0.2]),
        )
