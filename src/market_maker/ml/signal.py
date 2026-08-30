from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingSignal:
    """
    Trading signal generated from model probabilities.

    action:
        BUY  -> upward directional signal
        SELL -> downward directional signal
        HOLD -> neutral / insufficient confidence
    """

    action: str
    confidence: float


VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


def _validate_probability(
    probability: float,
    name: str,
) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def _validate_threshold(
    threshold: float,
) -> None:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must be greater than 0 and "
            "less than or equal to 1."
        )


def _validate_probabilities_sum(
    down_probability: float,
    flat_probability: float,
    up_probability: float,
) -> None:
    total = (
        down_probability
        + flat_probability
        + up_probability
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Probabilities must sum to 1."
        )


def generate_signal(
    *,
    down_probability: float,
    flat_probability: float,
    up_probability: float,
    threshold: float = 0.60,
) -> TradingSignal:
    """
    Convert model probabilities into a TradingSignal.

    Probabilities must be valid probabilities and must
    sum to 1.
    """

    _validate_probability(
        down_probability,
        "down_probability",
    )

    _validate_probability(
        flat_probability,
        "flat_probability",
    )

    _validate_probability(
        up_probability,
        "up_probability",
    )

    _validate_threshold(threshold)

    _validate_probabilities_sum(
        down_probability,
        flat_probability,
        up_probability,
    )

    probabilities = {
        "SELL": down_probability,
        "HOLD": flat_probability,
        "BUY": up_probability,
    }

    action = max(
        probabilities,
        key=probabilities.get,
    )

    confidence = float(probabilities[action])

    if confidence < threshold:
        action = "HOLD"

    return TradingSignal(
        action=action,
        confidence=confidence,
    )


def create_trading_signal(
    *,
    prediction: int | None = None,
    confidence: float | None = None,
    threshold: float = 0.60,
    down_probability: float | None = None,
    flat_probability: float | None = None,
    up_probability: float | None = None,
    minimum_confidence: float | None = None,
) -> TradingSignal:
    """
    Backwards-compatible signal factory.

    Supports either:

        create_trading_signal(
            prediction=1,
            confidence=0.8,
        )

    or:

        create_trading_signal(
            down_probability=0.1,
            flat_probability=0.1,
            up_probability=0.8,
        )
    """

    if minimum_confidence is not None:
        threshold = minimum_confidence

    _validate_threshold(threshold)

    probability_arguments = (
        down_probability,
        flat_probability,
        up_probability,
    )

    any_probability = any(
        value is not None
        for value in probability_arguments
    )

    if any_probability:
        if not all(
            value is not None
            for value in probability_arguments
        ):
            raise ValueError(
                "All three probabilities must be provided."
            )

        return generate_signal(
            down_probability=down_probability,
            flat_probability=flat_probability,
            up_probability=up_probability,
            threshold=threshold,
        )

    if prediction is None:
        raise ValueError(
            "Either prediction or probabilities "
            "must be provided."
        )

    if prediction not in (-1, 0, 1):
        raise ValueError(
            "prediction must be -1, 0, or 1."
        )

    if confidence is None:
        raise ValueError(
            "confidence must be provided with prediction."
        )

    _validate_probability(
        confidence,
        "confidence",
    )

    action_map = {
        -1: "SELL",
        0: "HOLD",
        1: "BUY",
    }

    action = action_map[prediction]

    if confidence < threshold:
        action = "HOLD"

    return TradingSignal(
        action=action,
        confidence=float(confidence),
    )