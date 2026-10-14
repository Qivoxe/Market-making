# Market Making Simulator

A modular Python framework for simulating market activity, managing an order book, generating machine-learning signals, and evaluating market-making strategies through backtesting.

## Overview

This project explores the engineering and quantitative concepts behind algorithmic market making. It combines an exchange simulation, order-flow features, market-regime modeling, and a backtesting workflow to evaluate strategy behavior.

## Key Components

- **Order Book and Exchange Engine** — manages simulated orders and exchange interactions.
- **Market Simulation** — generates market events under configurable market regimes.
- **Order-Flow Features** — calculates market features such as mid-price, spread, bid and ask volume, and order-book imbalance.
- **Machine-Learning Signals** — uses model probabilities to produce directional signals.
- **Market-Making Strategy** — makes quoting decisions and manages inventory through the backtest engine.
- **Backtesting** — tracks trades, equity, and profit and loss (PnL).

## Technology Stack

- Python
- NumPy
- pytest
- Machine-learning and quantitative strategy components

## Project Structure

```text
src/
└── market_maker/
    ├── backtest/
    ├── features/
    ├── ml/
    ├── orderbook/
    └── simulation/
tests/
```

The directory layout above is a high-level overview; individual files and responsibilities may vary as the project evolves.

## Installation

Clone the repository:

```bash
git clone https://github.com/Qivoxe/Market-making.git
cd Market-making
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the project dependencies using the repository's dependency file, if provided. For example:

```powershell
pip install -r requirements.txt
```

## Run Tests

Run the test suite from the project root:

```powershell
pytest -q
```

## Run a Backtest

The repository includes a market-making backtest runner. Its configuration supports parameters such as the number of simulation steps and random seed.

Use the project's runner or comparison module to execute the backtest. Check the current module interfaces before adapting example commands.

## Example Experiment

One local event-driven simulation with a fixed seed produced:

| Metric              | Observed result |
| ------------------- | --------------: |
| Simulation steps    |             100 |
| Equity observations |             100 |
| Trades              |              40 |
| Reported PnL        |         +0.9748 |

These figures represent a single simulation run, not evidence of consistent profitability. Results depend on the configuration, random seed, market assumptions, and strategy behavior. They should be validated across multiple runs and against appropriate baselines.

## Future Improvements

- Evaluate performance across multiple random seeds and market regimes.
- Compare strategy results against baseline approaches.
- Analyze inventory risk, drawdown, and risk-adjusted returns.
- Add reproducible experiment reports and visualizations.
- Improve realism in execution, transaction costs, and market dynamics.

## Disclaimer

This project is for educational and research purposes. Simulated performance does not guarantee real-world trading performance and should not be interpreted as investment advice.

## Author

**Shivam Roy**

GitHub: [Qivoxe](https://github.com/Qivoxe)
