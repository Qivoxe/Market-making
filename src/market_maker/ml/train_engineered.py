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
]


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

    # The current production predictor uses these five features.
    if X.shape[1] < 5:
        raise ValueError(
            "Dataset does not contain the required five features."
        )

    X = X[:, :5]

    print("Dataset")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature names: {FEATURE_NAMES}")

    unique, counts = np.unique(y, return_counts=True)

    print("\nTarget Distribution")

    labels = {
        -1: "DOWN",
        0: "FLAT",
        1: "UP",
    }

    for label, count in zip(unique, counts):
        print(
            f"{labels.get(int(label), str(label))}: "
            f"{count} ({count / len(y):.2%})"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    print("\nModel Performance")
    print(f"Accuracy: {accuracy:.4f}")
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
            f"{name}: {importance:.4f}"
        )


if __name__ == "__main__":
    main()