import datetime

# =============================================================================
# 1. SHARED MACROECONOMIC TAIL RISK (Merton Jump-Diffusion Crash Scenario)
# =============================================================================
# Both models evaluate survival against the EXACT same macro crash distribution:
JUMP_FREQUENCY_PER_YEAR = 0.07      # Arrival frequency
JUMP_MEAN_SIZE = -0.35              # Mean log-crash shock
JUMP_VOLATILITY = 0.07              # Jump size uncertainty / standard deviation


# =============================================================================
# 2. SHARED RETURN & DRIFT ASSUMPTIONS
# =============================================================================
DRIFT_CAP = 0.09                    # simple/arithmetic return ceiling
MARGIN_INTEREST_RATE = 0.015        # margin borrowing rate
HISTORICAL_LOOKBACK_YEARS = 20      # Lookback window for market ingestion


# =============================================================================
# 3. DYNAMIC METHOD PARAMETERS (Bayesian HMM Factor Calibration)
# =============================================================================
NUM_FACTORS = 5                     # Latent PCA factors
MCMC_DRAWS = 1000                   # Posterior samples per chain
MCMC_TUNE = 1000                    # Adaptation steps for NUTS
MCMC_CORES = 8                      # Parallel CPU chains


# =============================================================================
# 4. STATIC METHOD PARAMETERS (Heston Stochastic Volatility - Archive)
# =============================================================================
HESTON_KAPPA = 5.0                  # Mean-reversion speed of variance
HESTON_XI = 0.5                     # Volatility of volatility
HESTON_RHO = -0.7                   # Leverage correlation


# =============================================================================
# 5. SHARED MONTE CARLO & OPTIMIZER SETTINGS
# =============================================================================
NUM_PATHS = 250000                  # Simulated paths
MAX_MARGIN_CALL_PROBABILITY = 0.03  # Maximum allowable ruin budget
MAX_TARGET_LEVERAGE = 2.0           # Maximum leverage to test
OPTIMIZER_TOLERANCE = 0.01          # Grid search increment
RNG_SEED = 142                      # Reproducible random seed
HISTORY_INTERVAL_DAYS = 126         # Quantile extraction interval


# =============================================================================
# 6. PORTFOLIO HOLDINGS & BROKER RULES
# =============================================================================
BASE_CURRENCY = "CHF"
ACTIVE_ASSET = "VT"

CURRENT_DATE = datetime.date.today()
CURRENT_DEBT = 38050.69
CURRENT_SMA = 16836.03

HOLDINGS = {
    "VT": 345,
    "ACWI.SW": 84,
    "CHDVD.SW": 51,
    "MEUD.PA": 32.6,
    "VNA.DE": 251,
    "FLIN": 132,
    "GIVN.SW": 1,
    "MC.PA": 6,
    "PGHN.SW": 3,
    "SAP.DE": 12,
    "XS1970549561": 2,
    "DGE.L": 56
}

MM_REQUIREMENTS = {
    "VT": 0.25,
    "ACWI.SW": 0.25,
    "CHDVD.SW": 0.25,
    "MEUD.PA": 0.25,
    "VNA.DE": 0.25,
    "FLIN": 0.25,
    "GIVN.SW": 0.25,
    "MC.PA": 0.25,
    "PGHN.SW": 0.25,
    "SAP.DE": 0.25,
    "XS1970549561": 0.15,
    "DGE.L": 0.25
}

IM_REQUIREMENTS = {
    "VT": 0.25,
    "ACWI.SW": 0.2875,
    "CHDVD.SW": 0.2875,
    "MEUD.PA": 0.2875,
    "VNA.DE": 0.2875,
    "FLIN": 0.2875,
    "GIVN.SW": 0.2875,
    "MC.PA": 0.2875,
    "PGHN.SW": 0.2875,
    "SAP.DE": 0.2875,
    "XS1970549561": 0.15,
    "DGE.L": 0.2875
}

ASSET_CURRENCIES = {
    "VT": "USD",
    "ACWI.SW": "CHF",
    "CHDVD.SW": "CHF",
    "MEUD.PA": "EUR",
    "VNA.DE": "EUR",
    "FLIN": "USD",
    "GIVN.SW": "CHF",
    "MC.PA": "EUR",
    "PGHN.SW": "CHF",
    "SAP.DE": "EUR",
    "XS1970549561": "EUR",
    "DGE.L": "GBX"
}

OTC_REGISTRY = {
    "XS1970549561": {
        "live_price_local": 876.52,
        "proxy_ticker": "IS3C.DE"
    }
}


# =============================================================================
# 7. CASH FLOW SCHEDULES
# =============================================================================

DEFAULT_MONTHLY_DEPOSIT_FUTURE = 1000.00
DEPOSIT_SCHEDULE = [
    (datetime.date(2026, 1, 1), datetime.date(2026, 12, 31), 800.00),
    (datetime.date(2027, 1, 1), datetime.date(2027, 12, 31), 1200.00),
    (datetime.date(2028, 1, 1), datetime.date(2028, 6, 30), 1500.00),
    (datetime.date(2028, 7, 1), datetime.date(2028, 12, 31), 1800.00),
    (datetime.date(2029, 1, 1), datetime.date(2040, 12, 31), 2000.00)
]

POST_LAST_WITHDRAWAL_BUFFER_MONTHS = 11
POST_LAST_WITHDRAWAL_BUFFER_DAYS = 15

WITHDRAWAL_SCHEDULE = [
    {"date": datetime.date(2027, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2028, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2029, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 1), "amount": 10774.00},
    {"date": datetime.date(2031, 12, 30), "amount": 5000.00}
]