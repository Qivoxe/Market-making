from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.market_maker.data.dataset import build_dataset
from src.market_maker.data.generator import generate_market_dataset


def main() -> None:
    snapshots = generate_market_dataset(
        count_per_regime=5000,
        initial_price=100.0,
        seed=42,
    )

    thresholds = [
        0.00005,
        0.00010,
        0.00020,
        0.00050,
        0.00100,
    ]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("Target Threshold Experiment")
    print("=" * 70)

    for threshold in thresholds:
        dataset = build_dataset(
            snapshots,
            horizon=1,
            threshold=threshold,
        )

        X = dataset.X
        y = dataset.y

        unique, counts = np.unique(
            y,
            return_counts=True,
        )

        distribution = dict(
            zip(unique.astype(int), counts)
        )

        scores = []

        for train_index, test_index in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(X_test)

            score = balanced_accuracy_score(
                y_test,
                predictions,
            )

            scores.append(score)

        scores_array = np.asarray(
            scores,
            dtype=float,
        )

        down = distribution.get(-1, 0)
        flat = distribution.get(0, 0)
        up = distribution.get(1, 0)

        results.append(
            (
                threshold,
                scores_array.mean(),
                scores_array.std(),
                down / len(y),
                flat / len(y),
                up / len(y),
            )
        )

        print(
            f"Threshold: {threshold:.5f} | "
            f"Balanced Accuracy: {scores_array.mean():.4f} | "
            f"Std: {scores_array.std():.4f} | "
            f"DOWN: {down / len(y):.2%} | "
            f"FLAT: {flat / len(y):.2%} | "
            f"UP: {up / len(y):.2%}"
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    best = results[0]

    print("\nBest Threshold")
    print("=" * 70)
    print(f"Threshold: {best[0]:.5f}")
    print(f"Balanced Accuracy: {best[1]:.4f}")
    print(f"Std: {best[2]:.4f}")
    print(f"DOWN: {best[3]:.2%}")
    print(f"FLAT: {best[4]:.2%}")
    print(f"UP: {best[5]:.2%}")


if __name__ == "__main__":
    main()