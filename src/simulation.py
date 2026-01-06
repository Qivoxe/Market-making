from src.models import adverse_selection_trade
from src.config import TRIALS, SIGMAS, SPREADS

def run_simulation():
    """
    Runs experiments across signal qualities and spreads.
    Returns a dictionary: sigma -> expected pnl list
    """
    results = {}

    for sigma in SIGMAS:
        avg_pnls = []
        for spread in SPREADS:
            pnl = adverse_selection_trade(
                spread=spread,
                sigma=sigma,
                trials=TRIALS
            )
            avg_pnls.append(pnl)
        results[sigma] = avg_pnls

    return results
