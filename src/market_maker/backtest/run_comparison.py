from __future__ import annotations

from src.market_maker.backtest.compare import (
    ComparisonConfig,
    run_comparison,
)
from src.market_maker.backtest.report import (
    print_comparison,
    print_report,
)


def main() -> None:
    config = ComparisonConfig(
        count_per_regime=100,
        initial_price=100.0,
        seed=42,
    )

    comparison = run_comparison(config)

    print_report(
        comparison.baseline
    )

    print_report(
        comparison.ml
    )

    print_comparison(
        comparison
    )


if __name__ == "__main__":
    main()