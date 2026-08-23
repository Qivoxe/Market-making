"""Avellaneda-Stoikov model."""

import numpy as np


class AvellanedaStoikov:

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
