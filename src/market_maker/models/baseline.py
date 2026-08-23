"""Baseline model."""

import numpy as np


class BaselineModel:

    def __init__(self, price_low=1, price_high=100, signal_noise=5.0):
        self.price_low = price_low
        self.price_high = price_high
        self.signal_noise = signal_noise

    def simulate(self, spread, n_rounds=10_000, rng=None):
        """
        Returns dict with pnl array, trade_mask, and summary stats.
        """
        rng = rng or np.random.default_rng()
        half = spread / 2

        fair = rng.uniform(self.price_low, self.price_high, n_rounds)
        signal = fair + rng.normal(0, self.signal_noise, n_rounds)
        mid = (self.price_low + self.price_high) / 2  # MM quotes around mid

        bid = mid - half
        ask = mid + half

        # Trader buys at ask when signal > ask → MM sells below fair → MM loses
        # Trader sells at bid when signal < bid → MM buys above fair → MM loses
        # Otherwise no trade
        buy_trade  = signal > ask
        sell_trade = signal < bid
        trade      = buy_trade | sell_trade

        pnl = np.where(buy_trade,  fair - ask,   # MM sold at ask, true value = fair
              np.where(sell_trade, bid  - fair,   # MM bought at bid, true value = fair
              0.0))

        return {
            "pnl": pnl,
            "trade_rate": trade.mean(),
            "mean_pnl": pnl[trade].mean() if trade.any() else 0,
            "total_pnl": pnl.sum(),
        }

    def optimal_spread_sweep(self, spreads, n_rounds=10_000, rng=None):
        """Sweep spreads and return mean P&L per trade for each."""
        rng = rng or np.random.default_rng(42)
        results = [self.simulate(s, n_rounds, rng) for s in spreads]
        return np.array([r["mean_pnl"] for r in results])
