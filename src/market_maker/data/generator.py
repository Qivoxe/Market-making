from __future__ import annotations

from src.market_maker.simulation.market import (
    MarketSimulator,
    MarketSnapshot,
)
from src.market_maker.simulation.regime import MarketRegime


def generate_regime_snapshots(
    regime: MarketRegime,
    count: int,
    initial_price: float = 100.0,
    seed: int | None = None,
) -> list[MarketSnapshot]:
    simulator = MarketSimulator(
        initial_price=initial_price,
        seed=seed,
        regime=regime,
    )

    return simulator.generate_snapshots(count)

def generate_market_dataset(
    count_per_regime: int,
    initial_price: float = 100.0,
    seed: int | None = None,
) -> list[MarketSnapshot]:
    if count_per_regime <= 0:
        raise ValueError(
            "Count per regime must be greater than zero."
        )

    snapshots: list[MarketSnapshot] = []

    for index, regime in enumerate(MarketRegime):
        regime_seed = None if seed is None else seed + index

        regime_snapshots = generate_regime_snapshots(
            regime=regime,
            count=count_per_regime,
            initial_price=initial_price,
            seed=regime_seed,
        )

        snapshots.extend(regime_snapshots)

    return snapshots