"""
Market Making Research Suite
=============================
Run all simulations and generate plots.

Usage:
    python main.py              # full run
    python main.py --fast       # quick run (fewer paths, for testing)
"""

import sys
import time
import numpy as np

from src.simulator import (
    sweep_signal_quality,
    run_as_monte_carlo,
    sweep_risk_aversion,
    sweep_volatility,
)
from src.visualizer import (
    plot_adverse_selection,
    plot_as_simulation,
    plot_sensitivity,
)

FAST = "--fast" in sys.argv

N_ROUNDS  = 5_000  if FAST else 15_000
N_PATHS   = 300    if FAST else 1_500
N_SWEEP   = 200    if FAST else 500


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Market Making Research Suite                        ║")
    print("║  Adverse Selection + Avellaneda-Stoikov (2008)       ║")
    print("╚══════════════════════════════════════════════════════╝")
    if FAST:
        print("  [fast mode: reduced paths]\n")

    # ── 1. Adverse Selection Sweep ──────────────────────────────
    section("1/3  Adverse Selection — Signal Quality Sweep")
    t0 = time.time()
    alphas = np.linspace(1.0, 15.0, 10)
    sweep_results, spreads = sweep_signal_quality(alphas, n_rounds=N_ROUNDS)

    for r in sweep_results:
        print(f"  σ_signal={r['alpha']:5.1f}  |  "
              f"optimal spread={r['optimal_spread']:5.2f}  |  "
              f"max P&L={r['max_pnl']:+.4f}")

    plot_adverse_selection(sweep_results, spreads)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 2. Avellaneda-Stoikov Monte Carlo ────────────────────────
    section("2/3  Avellaneda-Stoikov — Monte Carlo Simulation")
    t0 = time.time()
    sim = run_as_monte_carlo(n_paths=N_PATHS)

    print(f"  Mean P&L  : {sim['mean_pnl']:+.4f}")
    print(f"  Std P&L   : {sim['std_pnl']:.4f}")
    print(f"  Sharpe    : {sim['sharpe']:.4f}")
    print(f"  % > 0     : {(sim['pnl']>0).mean()*100:.1f}%")

    plot_as_simulation(sim)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3. Sensitivity Analysis ──────────────────────────────────
    section("3/3  Sensitivity — Risk Aversion & Volatility Sweep")
    t0 = time.time()
    gammas = np.linspace(0.02, 0.5, 12)
    sigmas = np.linspace(0.5, 5.0, 12)

    gamma_results = sweep_risk_aversion(gammas, n_paths=N_SWEEP)
    sigma_results = sweep_volatility(sigmas,  n_paths=N_SWEEP)

    print("  γ sweep results:")
    for r in gamma_results[::3]:
        print(f"    γ={r['gamma']:.3f}  spread={r['spread_at_t0']:.3f}  "
              f"Sharpe={r['sharpe']:.3f}")

    plot_sensitivity(gamma_results, sigma_results)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Summary ──────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  All results saved to results/                       ║")
    print("║  ├── adverse_selection.png                           ║")
    print("║  ├── avellaneda_stoikov.png                          ║")
    print("║  └── sensitivity.png                                 ║")
    print("╚══════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()