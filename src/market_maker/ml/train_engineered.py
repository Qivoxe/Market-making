from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.market_maker.data.pipeline import (
    build_market_dataset,
)


FEATURE_NAMES = [
    "mid_price",
    "spread",
    "bid_volume",
    "ask_volume",
    "imbalance",
]


def main() -> None:
    dataset = build_market_dataset(
        count_per_regime=5000,
        horizon=1,
        threshold=0.0001,
        seed=42,
    )

    X = dataset.X
    y = dataset.y

    print("Dataset")
    print("=" * 70)

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature names: {FEATURE_NAMES}")

    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    labels = {
        -1: "DOWN",
        0: "FLAT",
        1: "UP",
    }

    print("\nTarget Distribution")

    for label, count in zip(
        unique,
        counts,
    ):
        print(
            f"{labels.get(int(label), str(label))}: "
            f"{count} "
            f"({count / len(y):.2%})"
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

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

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    print("\nModel Performance")

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[-1, 0, 1],
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[-1, 0, 1],
            zero_division=0,
        )
    )

    print("\nFeature Importance")

    for name, importance in sorted(
        zip(
            FEATURE_NAMES,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{name}: "
            f"{importance:.4f}"
        )


if __name__ == "__main__":
    main()