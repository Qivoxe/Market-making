from __future__ import annotations

from src.market_maker.data.dataset import (
    Dataset,
    build_dataset,
)
from src.market_maker.data.generator import (
    generate_market_dataset,
)


def build_market_dataset(
    count_per_regime: int,
    horizon: int = 1,
    initial_price: float = 100.0,
    threshold: float = 0.0001,
    seed: int | None = None,
) -> Dataset:

    snapshots = generate_market_dataset(
        count_per_regime=count_per_regime,
        initial_price=initial_price,
        seed=seed,
    )

    return build_dataset(
        snapshots=snapshots,
        horizon=horizon,
        threshold=threshold,
    )