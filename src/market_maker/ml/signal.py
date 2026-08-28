from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingSignal:
    action: str
    confidence: float


def generate_signal(
    down_probability: float,
    flat_probability: float,
    up_probability: float,
    threshold: float = 0.60,
) -> TradingSignal:
    probabilities = (
        down_probability,
        flat_probability,
        up_probability,
    )

    if any(
        probability < 0.0 or probability > 1.0
        for probability in probabilities
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            "Threshold must be in the range (0, 1]."
        )

    total = sum(probabilities)

    if total < 0.8:
        raise ValueError(
            "Probabilities must sum to at least 0.8."
        )

    if up_probability >= threshold:
        return TradingSignal(
            action="BUY",
            confidence=up_probability,
        )

    if down_probability >= threshold:
        return TradingSignal(
            action="SELL",
            confidence=down_probability,
        )

    return TradingSignal(
        action="HOLD",
        confidence=flat_probability,
    )