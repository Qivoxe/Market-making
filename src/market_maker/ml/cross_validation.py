from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from src.market_maker.data.dataset import build_dataset
from src.market_maker.data.generator import (
    generate_market_dataset,
)


def main() -> None:
    snapshots = generate_market_dataset(
        count_per_regime=5000,
        initial_price=100.0,
        seed=42,
    )

    dataset = build_dataset(
        snapshots,
        horizon=1,
        threshold=0.0001,
    )

    X = dataset.X
    y = dataset.y

    n_samples = len(X)
    n_folds = 5

    fold_size = n_samples // (n_folds + 1)

    scores = []

    print(
        "Time-Series Cross Validation"
    )
    print("=" * 70)

    for fold in range(1, n_folds + 1):
        train_end = (
            fold * fold_size
        )

        test_start = train_end
        test_end = min(
            test_start + fold_size,
            n_samples,
        )

        if test_start >= test_end:
            continue

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_test = X[
            test_start:test_end
        ]

        y_test = y[
            test_start:test_end
        ]

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

        predictions = model.predict(
            X_test
        )

        score = balanced_accuracy_score(
            y_test,
            predictions,
        )

        scores.append(score)

        print(
            f"Fold {fold}: "
            f"train={len(X_train):5d} "
            f"test={len(X_test):5d} "
            f"balanced_accuracy="
            f"{score:.4f}"
        )

    scores_array = np.asarray(
        scores,
        dtype=float,
    )

    print()
    print("Time-Series Cross Validation")
    print("=" * 70)

    print(
        f"Mean Balanced Accuracy: "
        f"{scores_array.mean():.4f}"
    )

    print(
        f"Std:                    "
        f"{scores_array.std():.4f}"
    )

    print(
        f"Min:                    "
        f"{scores_array.min():.4f}"
    )

    print(
        f"Max:                    "
        f"{scores_array.max():.4f}"
    )


if __name__ == "__main__":
    main()