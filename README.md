# Adverse Selection & Optimal Spread in Market Making

## Motivation
In real markets, tight bid-ask spreads can lead to losses when trading
against informed participants. This project simulates how information
asymmetry affects optimal market-making strategies.

## Model
- Hidden fair price ~ Uniform[1, 100]
- Market maker posts fixed bid-ask spread
- Informed trader observes noisy signal
- Trader trades only when profitable (adverse selection)

## Key Results
- Tight spreads lead to negative expected P&L
- Better trader information ⇒ wider optimal spread
- Explains why spreads exist in real markets

## Concepts Used
- Expected value
- Conditional reasoning
- Market microstructure
- Adverse selection

## How to Run
```bash
pip install -r requirements.txt
python main.py
