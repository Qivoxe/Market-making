from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from src.market_maker.data.engineered_dataset import (
    build_engineered_dataset,
)
from src.market_maker.data.generator import generate_market_dataset


FEATURE_NAMES = [
    "mid_price",
    "spread",
    "bid_volume",
    "ask_volume",
    "imbalance",
    "mid_return",
    "imbalance_change",
]


FEATURE_GROUPS = {
    "raw": [
        "mid_price",
        "spread",
        "bid_volume",
        "ask_volume",
        "imbalance",
    ],
    "order_flow": [
        "bid_volume",
        "ask_volume",
        "imbalance",
    ],
    "price": [
        "mid_price",
        "spread",
        "mid_return",
    ],
    "dynamics": [
        "mid_return",
        "imbalance_change",
    ],
    "raw_plus_dynamics": [
        "mid_price",
        "spread",
        "bid_volume",
        "ask_volume",
        "imbalance",
        "mid_return",
        "imbalance_change",
    ],
    "flow_plus_dynamics": [
        "bid_volume",
        "ask_volume",
        "imbalance",
        "mid_return",
        "imbalance_change",
    ],
}


def main() -> None:
    snapshots = generate_market_dataset(
        count_per_regime=5000,
        initial_price=100.0,
        seed=42,
    )

    dataset = build_engineered_dataset(
        snapshots,
        horizon=1,
        threshold=0.0001,
    )

    X = dataset.X
    y = dataset.y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Feature Ablation")
    print("=" * 60)

    results = []

    for group_name, selected_features in FEATURE_GROUPS.items():
        indices = [
            FEATURE_NAMES.index(feature)
            for feature in selected_features
        ]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(
            X_train[:, indices],
            y_train,
        )

        predictions = model.predict(
            X_test[:, indices]
        )

        score = balanced_accuracy_score(
            y_test,
            predictions,
        )

        results.append(
            (
                group_name,
                score,
                selected_features,
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print(
        f"{'Group':<22}"
        f"{'Balanced Accuracy':<20}"
        f"Features"
    )
    print("-" * 60)

    for group_name, score, features in results:
        print(
            f"{group_name:<22}"
            f"{score:<20.4f}"
            f"{', '.join(features)}"
        )

    print("\nBest Feature Group")

    best_group, best_score, best_features = results[0]

    print(f"Group: {best_group}")
    print(f"Balanced Accuracy: {best_score:.4f}")
    print(f"Features: {', '.join(best_features)}")


if __name__ == "__main__":
    main()