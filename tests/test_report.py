import pytest

from src.market_maker.backtest.report import (
    build_strategy_report,
    compare_strategies,
)


def test_build_strategy_report():
    report = build_strategy_report(
        name="Baseline",
        initial_cash=10_000.0,
        pnl=100.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_050.0,
            10_100.0,
        ],
        final_position=5.0,
        max_position=100.0,
    )

    assert report.name == "Baseline"
    assert report.metrics.total_pnl == pytest.approx(
        100.0
    )
    assert report.metrics.return_pct == pytest.approx(
        0.01
    )


def test_compare_strategies():
    baseline = build_strategy_report(
        name="Baseline",
        initial_cash=10_000.0,
        pnl=100.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_050.0,
            10_100.0,
        ],
        final_position=5.0,
        max_position=100.0,
    )

    ml = build_strategy_report(
        name="ML",
        initial_cash=10_000.0,
        pnl=150.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_100.0,
            10_150.0,
        ],
        final_position=3.0,
        max_position=100.0,
    )

    comparison = compare_strategies(
        baseline=baseline,
        ml=ml,
    )

    assert comparison.pnl_difference == pytest.approx(
        50.0
    )

    assert comparison.return_difference == pytest.approx(
        0.005
    )


def test_report_inventory_utilization():
    report = build_strategy_report(
        name="ML",
        initial_cash=10_000.0,
        pnl=0.0,
        trades=[],
        equity_curve=[
            10_000.0,
            10_001.0,
            10_002.0,
        ],
        final_position=25.0,
        max_position=100.0,
    )

    assert report.metrics.inventory_utilization == pytest.approx(
        0.25
    )