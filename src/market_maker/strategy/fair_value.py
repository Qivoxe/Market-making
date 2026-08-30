from __future__ import annotations


def calculate_fair_value(
    mid_price: float,
    directional_score: float,
    position: float,
    max_position: float,
    alpha: float = 0.50,
    inventory_skew_strength: float = 0.25,
) -> float:
    """
    Calculate a trading fair value.

    ML component:
        directional_score = P(UP) - P(DOWN)

    Inventory component:
        shifts fair value against existing inventory.
    """

    if mid_price <= 0:
        raise ValueError(
            "mid_price must be greater than zero."
        )

    if max_position <= 0:
        raise ValueError(
            "max_position must be greater than zero."
        )

    ml_skew = (
        alpha
        * directional_score
    )

    normalized_inventory = (
        position / max_position
    )

    inventory_skew = (
        -inventory_skew_strength
        * normalized_inventory
    )

    total_skew = (
        ml_skew
        + inventory_skew
    )

    return mid_price + total_skew