"""
Visualizer
==========
All plots use a dark terminal aesthetic (fits the quant/HFT vibe).
Saves PNGs to results/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Style ──────────────────────────────────────────────────────────────────
DARK_BG   = "#0d0f14"
PANEL_BG  = "#13161e"
ACCENT1   = "#00e5ff"   # cyan  – primary lines
ACCENT2   = "#ff6b6b"   # red   – losses / risk
ACCENT3   = "#69ff82"   # green – profit / positive
ACCENT4   = "#f5a623"   # amber – secondary
GRID_COL  = "#1e2333"
TEXT_COL  = "#c8d0e0"

def _base_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   TEXT_COL,
        "axes.titlecolor":   TEXT_COL,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "grid.color":        GRID_COL,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "text.color":        TEXT_COL,
        "font.family":       "monospace",
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COL,
        "legend.labelcolor": TEXT_COL,
    })

os.makedirs("results", exist_ok=True)


# ── Plot 1: Adverse Selection – P&L vs Spread for multiple signal qualities ──

def plot_adverse_selection(sweep_results, spreads, save=True):
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Adverse Selection Model — Signal Quality Analysis",
                 fontsize=14, fontweight="bold", color=ACCENT1, y=1.02)

    # Left: P&L curves per alpha
    ax = axes[0]
    cmap = plt.cm.get_cmap("cool", len(sweep_results))
    for i, res in enumerate(sweep_results):
        alpha = res["alpha"]
        ax.plot(spreads, res["pnl_curve"],
                color=cmap(i), linewidth=1.8,
                label=f"σ_signal = {alpha:.1f}")
        ax.axvline(res["optimal_spread"], color=cmap(i),
                   linestyle=":", linewidth=1, alpha=0.5)

    ax.axhline(0, color=ACCENT2, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("Bid-Ask Spread", fontsize=11)
    ax.set_ylabel("Mean P&L per Trade", fontsize=11)
    ax.set_title("P&L Curve per Signal Noise Level", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.4)

    # Right: Optimal spread vs signal quality
    ax2 = axes[1]
    alphas    = [r["alpha"] for r in sweep_results]
    opt_spreads = [r["optimal_spread"] for r in sweep_results]
    max_pnls  = [r["max_pnl"] for r in sweep_results]

    ax2b = ax2.twinx()
    ax2.plot(alphas, opt_spreads, color=ACCENT1, linewidth=2.2,
             marker="o", markersize=5, label="Optimal Spread")
    ax2b.plot(alphas, max_pnls, color=ACCENT3, linewidth=2.2,
              marker="s", markersize=5, linestyle="--", label="Max P&L")

    ax2.set_xlabel("Signal Noise (σ_signal)", fontsize=11)
    ax2.set_ylabel("Optimal Spread", fontsize=11, color=ACCENT1)
    ax2b.set_ylabel("Max P&L per Trade", fontsize=11, color=ACCENT3)
    ax2.tick_params(axis="y", colors=ACCENT1)
    ax2b.tick_params(axis="y", colors=ACCENT3)
    ax2b.set_facecolor(PANEL_BG)
    ax2.set_title("Optimal Spread Widens with Informed Traders", fontsize=12)
    ax2.grid(True, alpha=0.4)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig("results/adverse_selection.png", dpi=150,
                    bbox_inches="tight", facecolor=DARK_BG)
        print("  ✓ Saved results/adverse_selection.png")
    plt.close()


# ── Plot 2: A-S Monte Carlo Dashboard ────────────────────────────────────────

def plot_as_simulation(sim_results, save=True):
    _base_style()
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    pnl        = sim_results["pnl"]
    inv_paths  = sim_results["inventory_paths"]
    spreads    = sim_results["spread_path"]
    times      = sim_results["times"]
    mid_paths  = sim_results["mid_path"]

    fig.suptitle("Avellaneda-Stoikov Optimal Market Making — Monte Carlo",
                 fontsize=14, fontweight="bold", color=ACCENT1)

    # 1. P&L Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pnl, bins=60, color=ACCENT1, alpha=0.8, edgecolor="none")
    ax1.axvline(pnl.mean(), color=ACCENT3, linewidth=2, label=f"Mean={pnl.mean():.2f}")
    ax1.axvline(np.percentile(pnl, 5), color=ACCENT2, linewidth=1.5,
                linestyle="--", label=f"5th pct={np.percentile(pnl,5):.2f}")
    ax1.set_title("Terminal P&L Distribution", fontsize=11)
    ax1.set_xlabel("P&L")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Optimal Spread over Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, spreads, color=ACCENT4, linewidth=2.5)
    ax2.fill_between(times, spreads, alpha=0.2, color=ACCENT4)
    ax2.set_title("Optimal Spread Decays to Zero", fontsize=11)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Bid-Ask Spread δ*")
    ax2.grid(True, alpha=0.3)

    # 3. Sample mid-price paths
    ax3 = fig.add_subplot(gs[0, 2])
    sample_n = min(80, mid_paths.shape[0])
    for i in range(sample_n):
        ax3.plot(times, mid_paths[i], color=ACCENT1, alpha=0.08, linewidth=0.7)
    ax3.plot(times, mid_paths.mean(axis=0), color=ACCENT3,
             linewidth=2, label="Mean path")
    ax3.set_title("Mid-Price Paths (GBM)", fontsize=11)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Mid Price")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Inventory distribution over time (heatmap-style)
    ax4 = fig.add_subplot(gs[1, 0])
    inv_mean = inv_paths.mean(axis=0)
    inv_std  = inv_paths.std(axis=0)
    ax4.plot(times, inv_mean, color=ACCENT3, linewidth=2, label="Mean inventory")
    ax4.fill_between(times, inv_mean - inv_std, inv_mean + inv_std,
                     alpha=0.25, color=ACCENT3, label="±1 std")
    ax4.fill_between(times, inv_mean - 2*inv_std, inv_mean + 2*inv_std,
                     alpha=0.12, color=ACCENT3, label="±2 std")
    ax4.axhline(0, color=ACCENT2, linestyle="--", linewidth=1.2)
    ax4.set_title("Inventory Risk Over Time", fontsize=11)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Net Inventory (shares)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Cumulative P&L paths (fan chart)
    ax5 = fig.add_subplot(gs[1, 1])
    n_show = min(200, len(pnl))
    # Simulate cumulative pnl proxy via sorted final pnl
    sorted_pnl = np.sort(pnl)
    percentiles = [5, 25, 50, 75, 95]
    pct_vals    = np.percentile(pnl, percentiles)
    colors_pct  = [ACCENT2, ACCENT4, ACCENT3, ACCENT4, ACCENT2]
    for pct, val, c in zip(percentiles, pct_vals, colors_pct):
        ax5.axvline(val, color=c, linewidth=1.5, linestyle="--",
                    label=f"p{pct}={val:.1f}")
    ax5.hist(pnl, bins=50, color=ACCENT1, alpha=0.3, edgecolor="none")
    ax5.set_title("P&L Percentile Breakdown", fontsize=11)
    ax5.set_xlabel("Terminal P&L")
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # 6. Key Stats Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    sharpe  = sim_results["sharpe"]
    stats = [
        ["Metric",              "Value"],
        ["N Paths",             f"{len(pnl):,}"],
        ["Mean P&L",            f"{pnl.mean():.3f}"],
        ["Std P&L",             f"{pnl.std():.3f}"],
        ["Sharpe Ratio",        f"{sharpe:.3f}"],
        ["5th Percentile",      f"{np.percentile(pnl,5):.3f}"],
        ["95th Percentile",     f"{np.percentile(pnl,95):.3f}"],
        ["% Profitable",        f"{(pnl>0).mean()*100:.1f}%"],
        ["Max Drawdown (proxy)",f"{pnl.min():.3f}"],
    ]
    table = ax6.table(cellText=stats[1:], colLabels=stats[0],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if r > 0 else DARK_BG)
        cell.set_edgecolor(GRID_COL)
        cell.set_text_props(color=ACCENT1 if r == 0 else TEXT_COL)
    ax6.set_title("Simulation Summary", fontsize=11, color=TEXT_COL)

    if save:
        plt.savefig("results/avellaneda_stoikov.png", dpi=150,
                    bbox_inches="tight", facecolor=DARK_BG)
        print("  ✓ Saved results/avellaneda_stoikov.png")
    plt.close()


# ── Plot 3: Sensitivity — γ and σ Analysis ───────────────────────────────────

def plot_sensitivity(gamma_results, sigma_results, save=True):
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("A-S Sensitivity Analysis — Risk Aversion & Volatility",
                 fontsize=14, fontweight="bold", color=ACCENT1)

    # Gamma sweep
    ax = axes[0]
    gammas     = [r["gamma"]        for r in gamma_results]
    g_spreads  = [r["spread_at_t0"] for r in gamma_results]
    g_sharpes  = [r["sharpe"]       for r in gamma_results]

    ax2 = ax.twinx()
    ax.plot(gammas, g_spreads, color=ACCENT1, linewidth=2.2,
            marker="o", markersize=6, label="Optimal Spread")
    ax2.plot(gammas, g_sharpes, color=ACCENT3, linewidth=2.2,
             marker="s", markersize=6, linestyle="--", label="Sharpe Ratio")
    ax.set_xlabel("Risk Aversion γ", fontsize=11)
    ax.set_ylabel("Optimal Spread at t=0", fontsize=11, color=ACCENT1)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11, color=ACCENT3)
    ax.tick_params(axis="y", colors=ACCENT1)
    ax2.tick_params(axis="y", colors=ACCENT3)
    ax2.set_facecolor(PANEL_BG)
    ax.set_title("Higher Risk Aversion → Wider Spread", fontsize=12)
    ax.grid(True, alpha=0.4)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9)

    # Sigma sweep
    ax3 = axes[1]
    sigmas     = [r["sigma"]        for r in sigma_results]
    s_spreads  = [r["spread_at_t0"] for r in sigma_results]
    s_pnls     = [r["mean_pnl"]     for r in sigma_results]

    ax4 = ax3.twinx()
    ax3.plot(sigmas, s_spreads, color=ACCENT4, linewidth=2.2,
             marker="o", markersize=6, label="Optimal Spread")
    ax4.plot(sigmas, s_pnls, color=ACCENT2, linewidth=2.2,
             marker="s", markersize=6, linestyle="--", label="Mean P&L")
    ax3.set_xlabel("Mid-Price Volatility σ", fontsize=11)
    ax3.set_ylabel("Optimal Spread at t=0", fontsize=11, color=ACCENT4)
    ax4.set_ylabel("Mean Terminal P&L", fontsize=11, color=ACCENT2)
    ax3.tick_params(axis="y", colors=ACCENT4)
    ax4.tick_params(axis="y", colors=ACCENT2)
    ax4.set_facecolor(PANEL_BG)
    ax3.set_title("Higher Volatility → Wider Spread Required", fontsize=12)
    ax3.grid(True, alpha=0.4)
    l3, lb3 = ax3.get_legend_handles_labels()
    l4, lb4 = ax4.get_legend_handles_labels()
    ax3.legend(l3+l4, lb3+lb4, fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig("results/sensitivity.png", dpi=150,
                    bbox_inches="tight", facecolor=DARK_BG)
        print("  ✓ Saved results/sensitivity.png")
    plt.close()