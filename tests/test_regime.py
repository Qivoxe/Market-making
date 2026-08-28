from src.market_maker.simulation.regime import MarketRegime


def test_market_regimes_exist():
    assert MarketRegime.NORMAL.value == "normal"
    assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"
    assert MarketRegime.TRENDING_UP.value == "trending_up"
    assert MarketRegime.TRENDING_DOWN.value == "trending_down"
    assert MarketRegime.MEAN_REVERTING.value == "mean_reverting"