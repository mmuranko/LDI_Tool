import datetime

# =============================================================================
# 1. SHARED MACROECONOMIC TAIL RISK (Merton Jump-Diffusion Crash Scenario)
# =============================================================================
# Both models evaluate survival against the EXACT same macro crash distribution:
JUMP_FREQUENCY_PER_YEAR = 0.08      # Arrival frequency
JUMP_MEAN_SIZE = -0.30              # Mean log-crash shock
JUMP_VOLATILITY = 0.08              # Jump size uncertainty / standard deviation

# =============================================================================
# 2. SHARED RETURN, DIVIDEND, BOND & DRIFT ASSUMPTIONS
# =============================================================================
DRIFT_CAP = 0.08                    # Simple/arithmetic annual return ceiling
MARGIN_INTEREST_RATE = 0.015        # Annual margin borrowing rate
HISTORICAL_LOOKBACK_YEARS = 6       # Lookback window for market ingestion

DIV_MEAN_REVERSION_SPEED = 0.2      # CIR mean-reversion speed for equity dividend yields
DIV_PAYOUT_DAY = 15                 # Day of month when dividends are credited
BOND_YIELD_MR_SPEED = 0.25          # Ornstein-Uhlenbeck mean-reversion speed for bond YTM
BOND_YIELD_FLOOR = 0.005            # Minimum permissible bond yield

# =============================================================================
# 3. DYNAMIC METHOD PARAMETERS (Bayesian HMM Factor Calibration)
# =============================================================================
NUM_FACTORS = 4                     # Latent PCA factors
MCMC_TUNE = 1000                    # Adaptation steps for NUTS
MCMC_DRAWS = 500                    # Posterior samples per chain
MCMC_CORES = 4                      # Parallel CPU chains
HMM_STICKY_PRIOR = (95.0, 5.0)      # Dirichlet prior concentrations (stay_prob, switch_prob)
MAX_ACCEPTABLE_RHAT = 1.05          # Convergence diagnostic threshold

# =============================================================================
# 4. SHARED MONTE CARLO & OPTIMIZER SETTINGS
# =============================================================================
NUM_PATHS = 250000
MAX_MARGIN_CALL_PROBABILITY = 0.035
MAX_TARGET_LEVERAGE = 1.750
OPTIMIZER_TOLERANCE = 0.002
COARSE_GRID_STEP = 0.050            # Stage-1 coarse step before fine 0.005 step refinement
RNG_SEED = 42
HISTORY_INTERVAL_DAYS = 21          # Sub-sampling interval (days) for diagnostic trajectory plots
REBALANCE_INTERVAL_DAYS = 1         # 1 = daily worst-case high-water-mark rebalance stress test

# =============================================================================
# 5. PORTFOLIO HOLDINGS & BROKER RULES
# =============================================================================
BASE_CURRENCY = "CHF"
ACTIVE_ASSET = "VT"

CURRENT_DATE = datetime.date.today()
CURRENT_DEBT = 42800.89
CURRENT_SMA = 15782.59

HOLDINGS = {
    "VT": 529,
    "CHDVD.SW": 64,
    "MEUD.PA": 44,
    "VNA.DE": 251,
    "FLIN": 132,
    "MC.PA": 6,
    "SAP.DE": 12,
    "DGE.L": 56,
    "XS1970549561": 2,
    "XS3114901336": 4,
    "US925524AH30": 2
}

MM_REQUIREMENTS = {
    "VT": 0.25,
    "CHDVD.SW": 0.25,
    "MEUD.PA": 0.25,
    "VNA.DE": 0.25,
    "FLIN": 0.25,
    "MC.PA": 0.25,
    "SAP.DE": 0.25,
    "DGE.L": 0.25,
    "XS1970549561": 0.15,
    "XS3114901336": 0.15,
    "US925524AH30": 0.2972
}

IM_REQUIREMENTS = {
    "VT": 0.25,
    "CHDVD.SW": 0.2875,
    "MEUD.PA": 0.2875,
    "VNA.DE": 0.2875,
    "FLIN": 0.2875,
    "MC.PA": 0.2875,
    "SAP.DE": 0.2875,
    "DGE.L": 0.2875,
    "XS1970549561": 0.1875,
    "XS3114901336": 0.1875,
    "US925524AH30": 0.3715
}

ASSET_CURRENCIES = {
    "VT": "USD",
    "CHDVD.SW": "CHF",
    "MEUD.PA": "EUR",
    "VNA.DE": "EUR",
    "FLIN": "USD",
    "MC.PA": "EUR",
    "SAP.DE": "EUR",
    "DGE.L": "GBX",
    "XS1970549561": "EUR",
    "XS3114901336": "USD",
    "US925524AH30": "USD"
}

ASSET_PROXIES = {
    "MEUD.PA": ("EXSA.DE", "EUR"),      # iShares STOXX Europe 600 UCITS ETF (DE)
    "FLIN": ("INDA", "USD"),            # iShares MSCI India ETF
    "XS1970549561": ("IS3C.DE", "EUR"),
    "XS3114901336": ("EMB", "USD"),
    "US925524AH30": ("HYG", "USD"),
}

BOND_REGISTRY = {
    "XS1970549561": {
        "current_price_local": 845.79,
        "maturity_date": datetime.date(2034, 4, 3),
        "coupon_rate": 0.035,
        "coupon_frequency": 1,
        "face_value_local": 1000.0,
        "proxy_ticker": "IS3C.DE",
        "proxy_duration": 6.39
    },
    "XS3114901336": {
        "current_price_local": 973.63,
        "maturity_date": datetime.date(2030, 9, 16),
        "coupon_rate": 0.0575,
        "coupon_frequency": 2,
        "face_value_local": 1000.0,
        "proxy_ticker": "EMB",
        "proxy_duration": 6.28
    },
    "US925524AH30": {
        "current_price_local": 940.23,
        "maturity_date": datetime.date(2030, 7, 30),
        "coupon_rate": 0.07875,
        "coupon_frequency": 2,
        "face_value_local": 1000.0,
        "proxy_ticker": "HYG",
        "proxy_duration": 3.18
    }
}

# =============================================================================
# 6. CASH FLOW SCHEDULES
# =============================================================================
DEFAULT_MONTHLY_DEPOSIT_FUTURE = 1000.00

DEPOSIT_SCHEDULE = [
    (datetime.date(2026, 1, 1), datetime.date(2026, 12, 31), 300.00),
    (datetime.date(2027, 1, 1), datetime.date(2027, 12, 31), 0.00),
    (datetime.date(2028, 1, 1), datetime.date(2028, 6, 30), 0.00),
    (datetime.date(2028, 7, 1), datetime.date(2028, 12, 31), 300.00),
    (datetime.date(2029, 1, 1), datetime.date(2040, 12, 31), 1800.00)
]

'''
DEPOSIT_SCHEDULE = [
    (datetime.date(2026, 1, 1), datetime.date(2026, 12, 31), 300.00),
    (datetime.date(2027, 1, 1), datetime.date(2027, 12, 31), 1800.00),
    (datetime.date(2028, 1, 1), datetime.date(2028, 6, 30), 2000.00),
    (datetime.date(2028, 7, 1), datetime.date(2028, 12, 31), 2100.00),
    (datetime.date(2029, 1, 1), datetime.date(2040, 12, 31), 2200.00)
]
'''

POST_LAST_WITHDRAWAL_BUFFER_MONTHS = 11
POST_LAST_WITHDRAWAL_BUFFER_DAYS = 15

WITHDRAWAL_SCHEDULE = [
    {"date": datetime.date(2027, 12, 30), "amount": 1.00},
    {"date": datetime.date(2028, 12, 30), "amount": 1.00},
    {"date": datetime.date(2029, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 10774.00},
    {"date": datetime.date(2031, 12, 30), "amount": 5000.00}
]

'''
WITHDRAWAL_SCHEDULE = [
    {"date": datetime.date(2027, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2028, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2029, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 10774.00},
    {"date": datetime.date(2031, 12, 30), "amount": 5000.00}
]
'''