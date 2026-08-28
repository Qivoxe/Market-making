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

    configurations = [
        {
            "n_estimators": 200,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "max_features": None,
        },
    ]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("Random Forest Tuning")
    print("=" * 75)

    for number, config in enumerate(
        configurations,
        start=1,
    ):
        scores = []

        for train_index, test_index in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            model = RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                **config,
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

        scores_array = np.asarray(
            scores,
            dtype=float,
        )

        mean_score = scores_array.mean()
        std_score = scores_array.std()

        results.append(
            (
                mean_score,
                std_score,
                config,
            )
        )

        print(
            f"{number:02d} | "
            f"mean={mean_score:.4f} | "
            f"std={std_score:.4f} | "
            f"{config}"
        )

    results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_std, best_config = results[0]

    print("\nBest Configuration")
    print("=" * 75)
    print(f"Balanced Accuracy: {best_score:.4f}")
    print(f"Std: {best_std:.4f}")
    print(f"Configuration: {best_config}")


if __name__ == "__main__":
    main()