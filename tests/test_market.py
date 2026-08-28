import pytest
import numpy as np

from src.market_maker.simulation.regime import MarketRegime
from src.market_maker.simulation.market import MarketSimulator


def test_market_simulator_generates_snapshot():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
    )

    snapshot = simulator.step_market()

    assert snapshot is not None
    assert snapshot.step == 1
    assert snapshot.mid_price > 0
    assert snapshot.spread >= 0
    assert snapshot.bid_volume >= 0
    assert snapshot.ask_volume >= 0
    assert -1 <= snapshot.imbalance <= 1


def test_generate_snapshots():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
    )

    snapshots = simulator.generate_snapshots(100)

    assert len(snapshots) == 100

    assert snapshots[0].step == 1
    assert snapshots[-1].step == 100


def test_reproducibility():
    first = MarketSimulator(
        initial_price=100.0,
        seed=42,
    )

    second = MarketSimulator(
        initial_price=100.0,
        seed=42,
    )

    first_snapshots = first.generate_snapshots(20)
    second_snapshots = second.generate_snapshots(20)

    for a, b in zip(first_snapshots, second_snapshots):
        assert a == b


def test_different_seed_produces_different_market():
    first = MarketSimulator(
        initial_price=100.0,
        seed=42,
    )

    second = MarketSimulator(
        initial_price=100.0,
        seed=123,
    )

    first_snapshots = first.generate_snapshots(20)
    second_snapshots = second.generate_snapshots(20)

    assert first_snapshots != second_snapshots


def test_invalid_initial_price():
    with pytest.raises(ValueError):
        MarketSimulator(initial_price=0)


def test_invalid_snapshot_count():
    simulator = MarketSimulator(seed=42)

    with pytest.raises(ValueError):
        simulator.generate_snapshots(0)


def test_high_volatility_has_larger_price_moves():
    normal = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.NORMAL,
    )

    volatile = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.HIGH_VOLATILITY,
    )

    normal_moves = np.array(
        [
            normal._price_move()
            for _ in range(1000)
        ]
    )

    volatile_moves = np.array(
        [
            volatile._price_move()
            for _ in range(1000)
        ]
    )

    assert np.std(volatile_moves) > np.std(normal_moves)


def test_high_volatility_generates_snapshots():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.HIGH_VOLATILITY,
    )

    snapshots = simulator.generate_snapshots(100)

    assert len(snapshots) == 100

    for snapshot in snapshots:
        assert snapshot.mid_price > 0
        assert snapshot.spread >= 0
        assert -1 <= snapshot.imbalance <= 1  \


def test_trending_up_has_positive_drift():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.TRENDING_UP,
    )

    moves = np.array(
        [
            simulator._price_move()
            for _ in range(5000)
        ]
    )

    assert np.mean(moves) > 0


def test_trending_down_has_negative_drift():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.TRENDING_DOWN,
    )

    moves = np.array(
        [
            simulator._price_move()
            for _ in range(5000)
        ]
    )

    assert np.mean(moves) < 0          

def test_mean_reverting_move_pushes_toward_fair_price():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.MEAN_REVERTING,
    )

    simulator.reference_price = 110.0

    moves = np.array(
        [
            simulator._price_move()
            for _ in range(5000)
        ]
    )

    assert np.mean(moves) < 0


def test_mean_reverting_move_pushes_up_below_fair_price():
    simulator = MarketSimulator(
        initial_price=100.0,
        seed=42,
        regime=MarketRegime.MEAN_REVERTING,
    )

    simulator.reference_price = 90.0

    moves = np.array(
        [
            simulator._price_move()
            for _ in range(5000)
        ]
    )

    assert np.mean(moves) > 0