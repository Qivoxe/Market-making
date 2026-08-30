from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BacktestMetrics:
    total_pnl: float
    return_pct: float

    trade_count: int

    max_drawdown: float
    max_drawdown_pct: float

    sharpe_ratio: float

    final_position: float
    max_position: float
    inventory_utilization: float


def calculate_metrics(
    initial_cash: float,
    pnl: float,
    trades: Sequence,
    equity_curve: Sequence[float],
    final_position: float,
    max_position: float,
) -> BacktestMetrics:
    if initial_cash <= 0:
        raise ValueError(
            "Initial cash must be greater than zero."
        )

    if max_position <= 0:
        raise ValueError(
            "Max position must be greater than zero."
        )

    if len(equity_curve) == 0:
        raise ValueError(
            "Equity curve must not be empty."
        )

    equity = np.asarray(
        equity_curve,
        dtype=float,
    )


    return_pct = (
        pnl / initial_cash
    )


    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        running_max - equity
    )

    max_drawdown = float(
        np.max(drawdown)
    )

    drawdown_pct = np.divide(
        drawdown,
        running_max,
        out=np.zeros_like(drawdown),
        where=running_max != 0,
    )

    max_drawdown_pct = float(
        np.max(drawdown_pct)
    )


    if len(equity) < 3:
        sharpe_ratio = 0.0
    else:
        returns = np.diff(equity) / equity[:-1]

        mean_return = float(
            np.mean(returns)
        )

        std_return = float(
            np.std(
                returns,
                ddof=1,
            )
        )

        if std_return == 0.0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (
                mean_return
                / std_return
                * np.sqrt(len(returns))
            )


    inventory_utilization = (
        abs(final_position)
        / max_position
    )

    return BacktestMetrics(
        total_pnl=float(pnl),
        return_pct=float(return_pct),
        trade_count=len(trades),
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=float(sharpe_ratio),
        final_position=float(final_position),
        max_position=float(max_position),
        inventory_utilization=float(
            inventory_utilization
        ),
    )