import random
import numpy as np
from src.config import PRICE_LOW, PRICE_HIGH, TRUE_MEAN

def adverse_selection_trade(spread, sigma, trials):
    """
    Simulates market making against an informed trader.
    Market maker is uninformed and posts symmetric bid-ask.
    Trader trades only when profitable (adverse selection).
    """
    pnl = []
    mid = TRUE_MEAN
    bid = mid - spread / 2
    ask = mid + spread / 2

    for _ in range(trials):
        true_price = random.uniform(PRICE_LOW, PRICE_HIGH)
        signal = true_price + np.random.normal(0, sigma)

        if signal > ask:
            # Trader buys
            profit = ask - true_price
            pnl.append(profit)

        elif signal < bid:
            # Trader sells
            profit = true_price - bid
            pnl.append(profit)

        # else: no trade

    if len(pnl) == 0:
        return 0.0

    return float(np.mean(pnl))
