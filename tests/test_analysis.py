from src.market_maker.data.analysis import analyze_dataset


def test_analyze_dataset():
    stats = analyze_dataset(
        count_per_regime=100,
        horizon=1,
        seed=42,
    )

    assert stats["samples"] == 499.0
    assert stats["features"] == 5.0
    assert stats["mid_price_mean"] > 0
    assert stats["mid_price_std"] >= 0
    assert stats["spread_mean"] >= 0
    assert stats["spread_std"] >= 0
    assert -1 <= stats["imbalance_mean"] <= 1
    assert stats["imbalance_std"] >= 0