import numpy as np
import pytest

from src.market_maker.data.target import (
    create_directional_target,
)


def test_create_directional_target():
    returns = np.array([
        -0.001,
        -0.00001,
        0.0,
        0.00001,
        0.001,
    ])

    target = create_directional_target(
        returns,
        threshold=0.0001,
    )

    expected = np.array([
        -1,
        0,
        0,
        0,
        1,
    ])

    np.testing.assert_array_equal(
        target,
        expected,
    )


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        create_directional_target(
            np.array([0.1, -0.1]),
            threshold=0,
        )


def test_target_contains_only_three_classes():
    returns = np.array([
        -0.01,
        -0.001,
        0.0,
        0.001,
        0.01,
    ])

    target = create_directional_target(
        returns,
        threshold=0.0001,
    )

    assert set(target).issubset({
        -1,
        0,
        1,
    })