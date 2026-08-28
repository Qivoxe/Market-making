from src.market_maker.data.generator import generate_regime_snapshots
from src.market_maker.simulation.regime import MarketRegime
import pytest

from src.market_maker.data.generator import (
    generate_market_dataset,
    generate_regime_snapshots,
)

def test_generate_regime_snapshots():
    snapshots = generate_regime_snapshots(
        regime=MarketRegime.NORMAL,
        count=100,
        seed=42,
    )

    assert len(snapshots) == 100
    assert snapshots[0].step == 1
    assert snapshots[-1].step == 100


def test_generator_is_reproducible():
    first = generate_regime_snapshots(
        MarketRegime.NORMAL,
        50,
        seed=42,
    )

    second = generate_regime_snapshots(
        MarketRegime.NORMAL,
        50,
        seed=42,
    )

    assert first == second


def test_all_regimes_generate_data():
    for regime in MarketRegime:
        snapshots = generate_regime_snapshots(
            regime,
            50,
            seed=42,
        )

        assert len(snapshots) == 50

        for snapshot in snapshots:
            assert snapshot.mid_price > 0
            assert snapshot.spread >= 0
            assert -1 <= snapshot.imbalance <= 1

def test_generate_market_dataset():
    snapshots = generate_market_dataset(
        count_per_regime=100,
        seed=42,
    )

    assert len(snapshots) == 500

    for snapshot in snapshots:
        assert snapshot.mid_price > 0
        assert snapshot.spread >= 0
        assert -1 <= snapshot.imbalance <= 1


def test_generate_market_dataset_reproducible():
    first = generate_market_dataset(
        count_per_regime=50,
        seed=42,
    )

    second = generate_market_dataset(
        count_per_regime=50,
        seed=42,
    )

    assert first == second


def test_invalid_count_per_regime():
    with pytest.raises(ValueError):
        generate_market_dataset(
            count_per_regime=0,
        )            