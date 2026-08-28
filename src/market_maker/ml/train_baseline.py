from __future__ import annotations

import numpy as np

from src.market_maker.data.pipeline import build_market_dataset
from src.market_maker.ml.baseline import BaselineClassifier


def main() -> None:
    dataset = build_market_dataset(
        count_per_regime=5000,
        horizon=1,
        seed=42,
    )

    classifier = BaselineClassifier(
        test_size=0.2,
        random_state=42,
    )

    result = classifier.fit(
        dataset.X,
        dataset.y,
    )

    print("Dataset")
    print(f"Samples: {len(dataset.X)}")
    print(f"Features: {dataset.X.shape[1]}")

    print("\nTarget Distribution")

    for label in [-1, 0, 1]:
        count = np.sum(dataset.y == label)
        percentage = count / len(dataset.y) * 100

        name = {
            -1: "DOWN",
            0: "FLAT",
            1: "UP",
        }[label]

        print(
            f"{name}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\nModel Performance")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(
        f"Balanced Accuracy: "
        f"{result.balanced_accuracy:.4f}"
    )

    print("\nConfusion Matrix")
    print(result.confusion)

    print("\nClassification Report")

    split_index = int(
        len(dataset.X) * 0.8
    )

    X_test = dataset.X[split_index:]
    y_test = dataset.y[split_index:]

    print(
        classifier.classification_report(
            X_test,
            y_test,
        )
    )


if __name__ == "__main__":
    main()