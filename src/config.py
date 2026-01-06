# Global configuration parameters

PRICE_LOW = 1
PRICE_HIGH = 100

TRUE_MEAN = (PRICE_LOW + PRICE_HIGH) / 2

TRIALS = 20000

# Signal noise levels (lower = better information)
SIGMAS = [1000]

# Bid-ask spreads to test
SPREADS = range(2, 51, 2)
