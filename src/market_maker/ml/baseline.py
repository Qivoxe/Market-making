from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BaselineResult:
    model: LogisticRegression
    scaler: StandardScaler
    predictions: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray


class BaselineClassifier:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> BaselineResult:
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if len(X) < 2:
            raise ValueError("At least two samples are required.")

        split_index = int(len(X) * (1 - self.test_size))

        if split_index <= 0 or split_index >= len(X):
            raise ValueError("Invalid train/test split.")

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions,
        )

        confusion = confusion_matrix(
            y_test,
            predictions,
            labels=[-1, 0, 1],
        )

        return BaselineResult(
            model=self.model,
            scaler=self.scaler,
            predictions=predictions,
            probabilities=probabilities,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            confusion=confusion,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> str:
        predictions = self.predict(X)

        return classification_report(
            y,
            predictions,
            labels=[-1, 0, 1],
            zero_division=0,
        )