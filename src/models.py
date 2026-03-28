"""
Market Making Models
====================
1. BaselineModel   - Fixed spread with adverse selection (original)
2. AvellanedaStoikov - Optimal dynamic spread from A-S (2008) paper
"""

import numpy as np


# ──────────────────────────────────────────────
# 1. BASELINE MODEL (your original logic, cleaned up)
# ──────────────────────────────────────────────

class BaselineModel:
    """
    Fixed-spread market maker vs informed trader.
    
    Setup:
        - True fair price V ~ Uniform[price_low, price_high]
        - Informed trader sees signal: S = V + noise, noise ~ N(0, signal_noise²)
        - Trader buys if S > ask, sells if S < bid
        - MM profit per trade = half-spread, loss when adverse selection hits
    """

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


# ──────────────────────────────────────────────
# 2. AVELLANEDA-STOIKOV MODEL
# ──────────────────────────────────────────────

class AvellanedaStoikov:
    """
    Optimal market-making spread from Avellaneda & Stoikov (2008).
    
    The MM maximises expected utility of terminal wealth subject to
    inventory risk. The key results:

        Reservation price:  r(s,q,t) = s - q * γ * σ² * (T - t)
        Optimal spread:     δ* = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/κ)

    Parameters
    ----------
    gamma  : risk-aversion coefficient (higher → wider spread)
    sigma  : volatility of the mid-price
    kappa  : order-arrival intensity (higher → tighter spread)
    T      : trading horizon (normalised, e.g. 1.0)
    dt     : time step size
    """

    def __init__(self, gamma=0.1, sigma=2.0, kappa=1.5, T=1.0, dt=0.005):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.dt = dt

    def optimal_spread(self, t):
        """Compute A-S optimal full spread at time t."""
        time_left = self.T - t
        inventory_term = self.gamma * self.sigma**2 * time_left
        arrival_term   = (2 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return inventory_term + arrival_term

    def reservation_price(self, mid, inventory, t):
        """Skew the mid based on inventory exposure."""
        return mid - inventory * self.gamma * self.sigma**2 * (self.T - t)

    def simulate(self, n_paths=1000, rng=None):
        """
        Full path simulation with dynamic spread and inventory tracking.

        Mid-price follows GBM: dS = σ dW
        Returns per-path: final_pnl, inventory_path, spread_path, mid_path
        """
        rng = rng or np.random.default_rng(42)
        steps = int(self.T / self.dt)
        times = np.linspace(0, self.T, steps)

        all_pnl       = np.zeros(n_paths)
        all_inventory = np.zeros((n_paths, steps))
        all_spreads   = np.zeros(steps)
        mid_path      = np.zeros((n_paths, steps))

        # Pre-compute spreads (deterministic given time)
        for i, t in enumerate(times):
            all_spreads[i] = self.optimal_spread(t)

        for path in range(n_paths):
            mid       = 50.0  # starting mid-price
            inventory = 0
            cash      = 0.0

            for i, t in enumerate(times):
                mid_path[path, i] = mid

                # Price evolves as random walk
                mid += self.sigma * np.sqrt(self.dt) * rng.standard_normal()

                spread = all_spreads[i]
                r      = self.reservation_price(mid, inventory, t)
                bid    = r - spread / 2
                ask    = r + spread / 2

                # Order arrivals: Poisson-like Bernoulli at each step
                lam_b = self.kappa * np.exp(-self.kappa * (mid - bid))
                lam_a = self.kappa * np.exp(-self.kappa * (ask - mid))
                lam_b = np.clip(lam_b, 0, 1)
                lam_a = np.clip(lam_a, 0, 1)

                if rng.random() < lam_b * self.dt:   # buy order hit our bid
                    inventory += 1
                    cash      -= bid
                if rng.random() < lam_a * self.dt:   # sell order hit our ask
                    inventory -= 1
                    cash      += ask

                all_inventory[path, i] = inventory

            # Mark to market at final mid
            all_pnl[path] = cash + inventory * mid

        return {
            "pnl":            all_pnl,
            "mean_pnl":       all_pnl.mean(),
            "std_pnl":        all_pnl.std(),
            "sharpe":         all_pnl.mean() / (all_pnl.std() + 1e-9),
            "inventory_paths": all_inventory,
            "spread_path":    all_spreads,
            "times":          times,
            "mid_path":       mid_path,
        }

    def spread_sensitivity(self, gammas=None, sigmas=None):
        """
        Return optimal spread at t=0 across a range of γ or σ values.
        Useful for sensitivity analysis.
        """
        if gammas is not None:
            return np.array([
                AvellanedaStoikov(gamma=g, sigma=self.sigma,
                                  kappa=self.kappa, T=self.T).optimal_spread(0)
                for g in gammas
            ])
        if sigmas is not None:
            return np.array([
                AvellanedaStoikov(gamma=self.gamma, sigma=s,
                                  kappa=self.kappa, T=self.T).optimal_spread(0)
                for s in sigmas
            ])