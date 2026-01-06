from src.simulation import run_simulation
from src.plots import plot_results

def main():
    print("Running market-making simulation...")
    results = run_simulation()
    plot_results(results)

if __name__ == "__main__":
    main()
