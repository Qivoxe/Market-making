from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

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
        threshold=0.001,
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
    probabilities = model.predict_proba(X_test)

    classes = model.classes_

    class_to_index = {
        int(value): index
        for index, value in enumerate(classes)
    }

    down_probability = probabilities[
        :,
        class_to_index.get(-1),
    ]

    flat_probability = probabilities[
        :,
        class_to_index.get(0),
    ]

    up_probability = probabilities[
        :,
        class_to_index.get(1),
    ]

    confidence = probabilities.max(axis=1)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    print("Signal Analysis")
    print("=" * 70)
    print(f"Samples: {len(X)}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    print("\nAverage Probabilities")
    print(f"DOWN: {down_probability.mean():.4f}")
    print(f"FLAT: {flat_probability.mean():.4f}")
    print(f"UP: {up_probability.mean():.4f}")

    print("\nConfidence Distribution")

    confidence_levels = [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    for level in confidence_levels:
        mask = confidence >= level

        if not np.any(mask):
            print(
                f">= {level:.2f}: 0 samples"
            )
            continue

        accuracy = np.mean(
            predictions[mask] == y_test[mask]
        )

        print(
            f">= {level:.2f}: "
            f"{mask.sum()} samples "
            f"({mask.mean():.2%}) | "
            f"accuracy={accuracy:.4f}"
        )

    print("\nDirectional Signals")

    up_mask = up_probability >= 0.60
    down_mask = down_probability >= 0.60

    if np.any(up_mask):
        up_accuracy = np.mean(
            y_test[up_mask] == 1
        )
        print(
            f"UP >= 0.60: "
            f"{up_mask.sum()} signals | "
            f"precision={up_accuracy:.4f}"
        )
    else:
        print("UP >= 0.60: 0 signals")

    if np.any(down_mask):
        down_accuracy = np.mean(
            y_test[down_mask] == -1
        )
        print(
            f"DOWN >= 0.60: "
            f"{down_mask.sum()} signals | "
            f"precision={down_accuracy:.4f}"
        )
    else:
        print("DOWN >= 0.60: 0 signals")

    strong_up = up_probability >= 0.70
    strong_down = down_probability >= 0.70

    print("\nStrong Signals")

    if np.any(strong_up):
        print(
            f"UP >= 0.70: "
            f"{strong_up.sum()} signals | "
            f"precision="
            f"{np.mean(y_test[strong_up] == 1):.4f}"
        )
    else:
        print("UP >= 0.70: 0 signals")

    if np.any(strong_down):
        print(
            f"DOWN >= 0.70: "
            f"{strong_down.sum()} signals | "
            f"precision="
            f"{np.mean(y_test[strong_down] == -1):.4f}"
        )
    else:
        print("DOWN >= 0.70: 0 signals")


if __name__ == "__main__":
    main()