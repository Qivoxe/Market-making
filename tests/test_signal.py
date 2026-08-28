from src.market_maker.data.signal import (
    FEATURE_NAMES,
    calculate_feature_correlations,
)


def test_feature_correlations():
    correlations = calculate_feature_correlations(
        count_per_regime=100,
        horizon=1,
        seed=42,
    )

    assert set(correlations) == set(FEATURE_NAMES)

    for value in correlations.values():
        assert -1 <= value <= 1


def test_feature_correlations_are_finite():
    correlations = calculate_feature_correlations(
        count_per_regime=100,
        horizon=1,
        seed=42,
    )

    for value in correlations.values():
        assert value == value