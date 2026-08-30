from __future__ import annotations

from dataclasses import dataclass

from src.market_maker.backtest.metrics import (
    BacktestMetrics,
    calculate_metrics,
)


@dataclass(frozen=True)
class StrategyReport:
    name: str
    metrics: BacktestMetrics


@dataclass(frozen=True)
class StrategyComparison:
    baseline: StrategyReport
    ml: StrategyReport

    pnl_difference: float
    return_difference: float
    drawdown_difference: float
    sharpe_difference: float


def build_strategy_report(
    name: str,
    initial_cash: float,
    pnl: float,
    trades,
    equity_curve,
    final_position: float,
    max_position: float,
) -> StrategyReport:
    metrics = calculate_metrics(
        initial_cash=initial_cash,
        pnl=pnl,
        trades=trades,
        equity_curve=equity_curve,
        final_position=final_position,
        max_position=max_position,
    )

    return StrategyReport(
        name=name,
        metrics=metrics,
    )


def compare_strategies(
    baseline: StrategyReport,
    ml: StrategyReport,
) -> StrategyComparison:
    return StrategyComparison(
        baseline=baseline,
        ml=ml,
        pnl_difference=(
            ml.metrics.total_pnl
            - baseline.metrics.total_pnl
        ),
        return_difference=(
            ml.metrics.return_pct
            - baseline.metrics.return_pct
        ),
        drawdown_difference=(
            ml.metrics.max_drawdown
            - baseline.metrics.max_drawdown
        ),
        sharpe_difference=(
            ml.metrics.sharpe_ratio
            - baseline.metrics.sharpe_ratio
        ),
    )


def print_report(
    report: StrategyReport,
) -> None:
    metrics = report.metrics

    print()
    print(report.name)
    print("=" * 50)

    print(
        f"PnL:                    "
        f"{metrics.total_pnl:.2f}"
    )

    print(
        f"Return:                 "
        f"{metrics.return_pct * 100:.4f}%"
    )

    print(
        f"Trades:                 "
        f"{metrics.trade_count}"
    )

    print(
        f"Max Drawdown:           "
        f"{metrics.max_drawdown:.2f}"
    )

    print(
        f"Max Drawdown %:         "
        f"{metrics.max_drawdown_pct * 100:.4f}%"
    )

    print(
        f"Sharpe Ratio:           "
        f"{metrics.sharpe_ratio:.4f}"
    )

    print(
        f"Final Position:         "
        f"{metrics.final_position:.2f}"
    )

    print(
        f"Inventory Utilization:  "
        f"{metrics.inventory_utilization * 100:.2f}%"
    )


def print_comparison(
    comparison: StrategyComparison,
) -> None:
    print()
    print("Strategy Comparison")
    print("=" * 50)

    print(
        f"Baseline PnL:       "
        f"{comparison.baseline.metrics.total_pnl:.2f}"
    )

    print(
        f"ML PnL:             "
        f"{comparison.ml.metrics.total_pnl:.2f}"
    )

    print(
        f"PnL Difference:     "
        f"{comparison.pnl_difference:.2f}"
    )

    print(
        f"Baseline Return:    "
        f"{comparison.baseline.metrics.return_pct * 100:.4f}%"
    )

    print(
        f"ML Return:          "
        f"{comparison.ml.metrics.return_pct * 100:.4f}%"
    )

    print(
        f"Return Difference:  "
        f"{comparison.return_difference * 100:.4f}%"
    )

    print(
        f"Baseline Drawdown:  "
        f"{comparison.baseline.metrics.max_drawdown:.2f}"
    )

    print(
        f"ML Drawdown:        "
        f"{comparison.ml.metrics.max_drawdown:.2f}"
    )

    print(
        f"Drawdown Difference:"
        f" {comparison.drawdown_difference:.2f}"
    )

    print(
        f"Baseline Sharpe:    "
        f"{comparison.baseline.metrics.sharpe_ratio:.4f}"
    )

    print(
        f"ML Sharpe:          "
        f"{comparison.ml.metrics.sharpe_ratio:.4f}"
    )

    print(
        f"Sharpe Difference:  "
        f"{comparison.sharpe_difference:.4f}"
    )