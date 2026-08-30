from __future__ import annotations


def calculate_inventory_skew(
    position: float,
    max_position: float,
    skew_strength: float = 0.25,
) -> float:
    """
    Positive value means the market maker should
    shift its fair value upward.

    Negative value means the market maker should
    shift its fair value downward.

    Since a short position should encourage buying,
    a negative position produces a positive skew.
    """

    if max_position <= 0:
        raise ValueError(
            "max_position must be greater than zero."
        )

    if skew_strength < 0:
        raise ValueError(
            "skew_strength cannot be negative."
        )

    normalized_position = (
        position / max_position
    )

    return (
        -normalized_position
        * skew_strength
    )