import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from src.config import SPREADS

def plot_results(results):
    """
    Plots Expected P&L vs Spread for different signal qualities
    """
    plt.figure()

    for sigma, pnl_values in results.items():
        plt.plot(SPREADS, pnl_values, label=f"Signal noise σ = {sigma}")

    plt.axhline(0)
    plt.xlabel("Bid-Ask Spread")
    plt.ylabel("Expected P&L")
    plt.title("Adverse Selection: Optimal Spread vs Information Quality")
    plt.legend()
    plt.tight_layout()
    plt.show()
