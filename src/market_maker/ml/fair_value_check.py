from __future__ import annotations

from src.market_maker.data.generator import (
    generate_market_dataset,
)
from src.market_maker.ml.predictor import (
    MLPredictor,
)


def main() -> None:
    snapshots = generate_market_dataset(
        count_per_regime=500,
        initial_price=100.0,
        seed=42,
    )

    predictor = MLPredictor(
        "models/random_forest.joblib"
    )

    print("ML Fair Value Signal Check")
    print("=" * 70)

    print(
        "step | mid      | score     | "
        "DOWN     FLAT     UP       | action"
    )

    print("-" * 70)

    for step, snapshot in enumerate(
        snapshots[:50]
    ):
        prediction = predictor.predict(
            mid_price=snapshot.mid_price,
            spread=snapshot.spread,
            bid_volume=snapshot.bid_volume,
            ask_volume=snapshot.ask_volume,
            imbalance=snapshot.imbalance,
        )

        print(
            f"{step:4d} | "
            f"{snapshot.mid_price:8.4f} | "
            f"{prediction.directional_score:+.4f} | "
            f"{prediction.down_probability:.4f}   "
            f"{prediction.flat_probability:.4f}   "
            f"{prediction.up_probability:.4f}   | "
            f"{prediction.action}"
        )


if __name__ == "__main__":
    main()