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

    dataset = build_dataset(
        snapshots,
        horizon=1,
        threshold=0.0001,
    )

    X = dataset.X
    y = dataset.y

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = []

    for fold, (train_index, test_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
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

        print(
            f"Fold {fold}: "
            f"{score:.4f}"
        )

    scores_array = np.asarray(
        scores,
        dtype=float,
    )

    print("\nCross Validation")
    print(f"Mean: {scores_array.mean():.4f}")
    print(f"Std: {scores_array.std():.4f}")
    print(f"Min: {scores_array.min():.4f}")
    print(f"Max: {scores_array.max():.4f}")


if __name__ == "__main__":
    main()