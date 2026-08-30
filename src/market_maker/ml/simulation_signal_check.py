from __future__ import annotations

import numpy as np

from src.market_maker.data.generator import generate_market_dataset
from src.market_maker.ml.model import load_model


def main() -> None:
    snapshots = generate_market_dataset(
        count_per_regime=100,
        initial_price=100.0,
        seed=42,
    )

    model = load_model(
        "models/random_forest.joblib"
    )

    features = np.asarray(
        [
            [
                snapshot.mid_price,
                snapshot.spread,
                snapshot.bid_volume,
                snapshot.ask_volume,
                snapshot.imbalance,
            ]
            for snapshot in snapshots
        ],
        dtype=float,
    )

    probabilities = model.predict_proba(features)
    classes = model.classes_

    print("Simulation Signal Check")
    print("=" * 70)
    print(f"Samples: {len(snapshots)}")
    print(f"Features: {features.shape[1]}")
    print(f"Classes: {classes}")

    print("\nFeature Ranges")
    print("-" * 70)

    names = [
        "mid_price",
        "spread",
        "bid_volume",
        "ask_volume",
        "imbalance",
    ]

    for index, name in enumerate(names):
        print(
            f"{name:15s}"
            f" min={features[:, index].min():.6f}"
            f" max={features[:, index].max():.6f}"
            f" mean={features[:, index].mean():.6f}"
        )

    print("\nProbability Distribution")
    print("-" * 70)

    for index, label in enumerate(classes):
        values = probabilities[:, index]

        print(
            f"{label:>5}:"
            f" min={values.min():.4f}"
            f" max={values.max():.4f}"
            f" mean={values.mean():.4f}"
        )

    confidence = probabilities.max(axis=1)
    predicted_classes = classes[
        probabilities.argmax(axis=1)
    ]

    print("\nConfidence")
    print("-" * 70)

    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
        count = int(
            np.sum(confidence >= threshold)
        )

        print(
            f">= {threshold:.2f}: "
            f"{count} samples"
            f" ({count / len(confidence) * 100:.2f}%)"
        )

    print("\nPredicted Classes")
    print("-" * 70)

    for label in classes:
        count = int(
            np.sum(predicted_classes == label)
        )

        print(
            f"{label:>5}: "
            f"{count} "
            f"({count / len(predicted_classes) * 100:.2f}%)"
        )

    print("\nTop 10 Signals")
    print("-" * 70)

    top_indices = np.argsort(
        confidence
    )[-10:][::-1]

    for index in top_indices:
        probability_map = {
            int(label): float(probability)
            for label, probability in zip(
                classes,
                probabilities[index],
            )
        }

        print(
            f"step={snapshots[index].step:4d} "
            f"mid={snapshots[index].mid_price:10.4f} "
            f"spread={snapshots[index].spread:8.4f} "
            f"imbalance={snapshots[index].imbalance:7.4f} "
            f"DOWN={probability_map.get(-1, 0.0):.4f} "
            f"FLAT={probability_map.get(0, 0.0):.4f} "
            f"UP={probability_map.get(1, 0.0):.4f}"
        )


if __name__ == "__main__":
    main()