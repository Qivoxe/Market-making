"""
Monte Carlo Simulation Engine
==============================
Runs parameter sweeps across signal quality (α), spread width, 
risk aversion (γ), and volatility (σ). Returns structured results
for downstream visualisation.
"""

import numpy as np
from src.models import BaselineModel, AvellanedaStoikov


def sweep_signal_quality(alphas, n_rounds=10_000, seed=42):
    """
    Sweep over informed-trader signal noise (lower α → sharper signal).
    Returns optimal spread and mean P&L for each alpha.
    """
    rng = np.random.default_rng(seed)
    spreads = np.linspace(0.5, 20, 60)
    results = []

    for alpha in alphas:
        model = BaselineModel(signal_noise=alpha)
        pnls  = model.optimal_spread_sweep(spreads, n_rounds=n_rounds, rng=rng)
        best_idx   = np.argmax(pnls)
        results.append({
            "alpha":         alpha,
            "optimal_spread": spreads[best_idx],
            "max_pnl":        pnls[best_idx],
            "pnl_curve":      pnls,
        })

    return results, spreads


def run_as_monte_carlo(n_paths=2000, gamma=0.1, sigma=2.0,
                       kappa=1.5, T=1.0, dt=0.005, seed=42):
    """Run a single A-S simulation and return full results."""
    model = AvellanedaStoikov(gamma=gamma, sigma=sigma,
                               kappa=kappa, T=T, dt=dt)
    return model.simulate(n_paths=n_paths, rng=np.random.default_rng(seed))


def sweep_risk_aversion(gammas, n_paths=500, seed=42):
    """Compare A-S performance across different γ values."""
    results = []
    for g in gammas:
        model = AvellanedaStoikov(gamma=g)
        res   = model.simulate(n_paths=n_paths,
                               rng=np.random.default_rng(seed))
        results.append({
            "gamma":    g,
            "mean_pnl": res["mean_pnl"],
            "std_pnl":  res["std_pnl"],
            "sharpe":   res["sharpe"],
            "spread_at_t0": model.optimal_spread(0),
        })
    return results


def sweep_volatility(sigmas, n_paths=500, seed=42):
    """Compare A-S optimal spreads across volatility regimes."""
    results = []
    for s in sigmas:
        model = AvellanedaStoikov(sigma=s)
        res   = model.simulate(n_paths=n_paths,
                               rng=np.random.default_rng(seed))
        results.append({
            "sigma":    s,
            "mean_pnl": res["mean_pnl"],
            "std_pnl":  res["std_pnl"],
            "sharpe":   res["sharpe"],
            "spread_at_t0": model.optimal_spread(0),
        })
    return results