from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

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


def train_model(
    count_per_regime: int = 1000,
    horizon: int = 1,
    threshold: float = 0.001,
    seed: int = 42,
) -> RandomForestClassifier:
    """
    Train the production Random Forest model.
    """

    dataset = build_market_dataset(
        count_per_regime=count_per_regime,
        horizon=horizon,
        threshold=threshold,
        seed=seed,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        dataset.X,
        dataset.y,
    )

    return model


def save_model(
    model: RandomForestClassifier,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )


def load_model(
    path: str | Path,
) -> RandomForestClassifier:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    model = joblib.load(path)

    if not isinstance(
        model,
        RandomForestClassifier,
    ):
        raise TypeError(
            "Saved object is not a RandomForestClassifier."
        )

    return model