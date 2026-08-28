from __future__ import annotations

import numpy as np

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            random_state=42,
        ),
    }

    results = []

    print("Model Comparison")
    print("=" * 60)
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    for name, model in models.items():
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

        results.append(
            (
                name,
                accuracy,
                balanced_accuracy,
            )
        )

    results.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    print("\nResults")
    print("-" * 60)
    print(
        f"{'Model':<25}"
        f"{'Accuracy':<15}"
        f"Balanced Accuracy"
    )
    print("-" * 60)

    for name, accuracy, balanced_accuracy in results:
        print(
            f"{name:<25}"
            f"{accuracy:<15.4f}"
            f"{balanced_accuracy:.4f}"
        )

    best_model, best_accuracy, best_balanced_accuracy = results[0]

    print("\nBest Model")
    print(f"Model: {best_model}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(
        f"Balanced Accuracy: "
        f"{best_balanced_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()