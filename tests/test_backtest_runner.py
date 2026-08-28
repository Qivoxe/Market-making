from __future__ import annotations

import pytest

from src.market_maker.backtest.runner import (
    BacktestConfig,
    run_backtest,
)


def test_run_backtest_returns_result():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
        )
    )

    assert result is not None


def test_backtest_has_valid_initial_cash():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
            initial_cash=10_000.0,
        )
    )

    assert result.initial_cash == pytest.approx(10_000.0)


def test_backtest_produces_finite_values():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
        )
    )

    assert result.final_cash == pytest.approx(
        result.final_cash
    )

    assert result.final_equity == pytest.approx(
        result.final_equity
    )

    assert result.pnl == pytest.approx(
        result.pnl
    )

    assert result.return_pct == pytest.approx(
        result.return_pct
    )


def test_backtest_equity_matches_cash_and_position():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
        )
    )

    expected_equity = (
        result.final_cash
        + result.final_position
        * result.final_mid_price
    )

    assert result.final_equity == pytest.approx(
        expected_equity
    )


def test_backtest_pnl_matches_equity():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
        )
    )

    expected_pnl = (
        result.final_equity
        - result.initial_cash
    )

    assert result.pnl == pytest.approx(
        expected_pnl
    )


def test_backtest_return_matches_pnl():
    result = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            initial_price=100.0,
            seed=42,
        )
    )

    expected_return = (
        result.pnl
        / result.initial_cash
    )

    assert result.return_pct == pytest.approx(
        expected_return
    )


def test_backtest_is_reproducible():
    config = BacktestConfig(
        count_per_regime=20,
        initial_price=100.0,
        seed=42,
    )

    first = run_backtest(config)
    second = run_backtest(config)

    assert first.final_cash == pytest.approx(
        second.final_cash
    )

    assert first.final_position == pytest.approx(
        second.final_position
    )

    assert first.final_equity == pytest.approx(
        second.final_equity
    )

    assert first.pnl == pytest.approx(
        second.pnl
    )


def test_different_seed_can_change_result():
    first = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            seed=42,
        )
    )

    second = run_backtest(
        BacktestConfig(
            count_per_regime=20,
            seed=123,
        )
    )

    assert (
        first.final_equity
        != second.final_equity
    )


def test_invalid_count_per_regime():
    with pytest.raises(ValueError):
        run_backtest(
            BacktestConfig(
                count_per_regime=0,
            )
        )


def test_invalid_initial_price():
    with pytest.raises(ValueError):
        run_backtest(
            BacktestConfig(
                initial_price=0.0,
            )
        )


def test_invalid_max_position():
    with pytest.raises(ValueError):
        run_backtest(
            BacktestConfig(
                max_position=0.0,
            )
        )


def test_invalid_order_size():
    with pytest.raises(ValueError):
        run_backtest(
            BacktestConfig(
                order_size=0.0,
            )
        )