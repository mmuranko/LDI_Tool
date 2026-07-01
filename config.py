import datetime

# --- Tail Risk (Merton Jump-Diffusion Parameters) ---
JUMP_FREQUENCY_PER_YEAR = 0.06   # Expected number of macro shocks per year (e.g., 0.50 = one every 2 years)
JUMP_MEAN_SIZE = -0.40           # Mean log-jump size. Median simple jump is exp(-0.15)-1, about -14%.
JUMP_VOLATILITY = 0.06           # Standard deviation of log-jump size, not simple-return volatility.

# --- Heston Stochastic Volatility Parameters ---
HESTON_KAPPA = 5.0       # Mean-reversion speed (e.g., 3.0 means vol shocks decay over ~4 months)
HESTON_XI = 0.5          # Volatility of Volatility (How aggressively the VIX itself swings)
HESTON_RHO = -0.7        # Leverage Effect: When equities drop, variance violently spikes (highly negative correlation)

# --- Simulation Risk Limits ---
NUM_PATHS = 500000
MAX_MARGIN_CALL_PROBABILITY = 0.03

# --- Optimizer Bounds & Tolerance ---
MAX_TARGET_LEVERAGE = 2.0
OPTIMIZER_TOLERANCE = 0.01

# Unified Drift Limit
DRIFT_CAP = 0.09

# --- Optimizer Controls ---
RNG_SEED = 42
HISTORY_INTERVAL_DAYS = 126

# --- Currency & FX Configuration ---
BASE_CURRENCY = "CHF"
ACTIVE_ASSET = "VT"

# --- Holdings Inventory ---
HOLDINGS = {
    "VT": 243,
    "ACWI.SW": 84,
    "CHDVD.SW": 51,
    "MEUD.PA": 32.6,
    "VNA.DE": 251,
    "FLIN": 132,
    "GIVN.SW": 1,
    "PGHN.SW": 3,
    "MC.PA": 6,
    "SAP.DE": 12,
    "FONC.SW": 28,
    "ANFO.SW": 40,
    "DGE.L": 56,
    "XS1970549561": 2
}

# --- Broker Margin Rules ---
MM_REQUIREMENTS = {
    "VT": 0.25,
    "ACWI.SW": 0.25,
    "CHDVD.SW": 0.25,
    "MEUD.PA": 0.25,
    "VNA.DE": 0.25,
    "FLIN": 0.25,
    "GIVN.SW": 0.25,
    "PGHN.SW": 0.25,
    "MC.PA": 0.25,
    "SAP.DE": 0.25,
    "FONC.SW": 0.25,
    "ANFO.SW": 0.25,
    "DGE.L": 0.25,
    "XS1970549561": 0.15
}

IM_REQUIREMENTS = {
    "VT": 0.25,
    "ACWI.SW": 0.2875,
    "CHDVD.SW": 0.2875,
    "MEUD.PA": 0.2875,
    "VNA.DE": 0.2875,
    "FLIN": 0.2875,
    "GIVN.SW": 0.2875,
    "PGHN.SW": 0.2875,
    "MC.PA": 0.2875,
    "SAP.DE": 0.2875,
    "FONC.SW": 0.2875,
    "ANFO.SW": 0.2875,
    "DGE.L": 0.2875,
    "XS1970549561": 0.15
}

# Explicitly declare the trading currency for every asset
ASSET_CURRENCIES = {
    "VT": "USD",
    "ACWI.SW": "CHF",
    "CHDVD.SW": "CHF",
    "MEUD.PA": "EUR",
    "VNA.DE": "EUR",
    "FLIN": "USD",
    "GIVN.SW": "CHF",
    "PGHN.SW": "CHF",
    "MC.PA": "EUR",
    "SAP.DE": "EUR",
    "FONC.SW": "CHF",
    "ANFO.SW": "CHF",
    "DGE.L": "GBX",
    "XS1970549561": "EUR"
}

# --- OTC & Fixed Income Registry ---
OTC_REGISTRY = {
    "XS1970549561": {
        "live_price_local": 894.70,
        "proxy_ticker": "IS3C.DE"
    }
}

# --- Balance Sheet Cash State ---
CURRENT_DATE = datetime.date.today()
CURRENT_DEBT = 35049.63
CURRENT_SMA = 15290.52
TODAY_DEPOSIT = 0

# --- Future Projections Config ---
DEFAULT_MONTHLY_DEPOSIT_2026 = 1000.00
DEFAULT_MONTHLY_DEPOSIT_FUTURE = 2000.00
MARGIN_INTEREST_RATE = 0.015
HISTORICAL_LOOKBACK_YEARS = 20

# --- Simulation Horizon Buffer ---
POST_LAST_WITHDRAWAL_BUFFER_MONTHS = 11
POST_LAST_WITHDRAWAL_BUFFER_DAYS = 15

# --- Strict Liability Ledger ---
WITHDRAWAL_SCHEDULE = [
    {"date": datetime.date(2027, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2028, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2029, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 1), "amount": 10774.00},
    {"date": datetime.date(2031, 12, 30), "amount": 5000.00}
]