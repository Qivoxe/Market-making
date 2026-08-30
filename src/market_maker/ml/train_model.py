from __future__ import annotations

from pathlib import Path

from src.market_maker.ml.model import (
    save_model,
    train_model,
)


MODEL_PATH = Path(
    "models/random_forest.joblib"
)


def main() -> None:
    print("Training Random Forest...")

    model = train_model(
        count_per_regime=5000,
        horizon=1,
        threshold=0.001,
        seed=42,
    )

    save_model(
        model,
        MODEL_PATH,
    )

    print()
    print("Model trained successfully.")
    print(f"Classes: {model.classes_}")
    print(f"Features: {model.n_features_in_}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()