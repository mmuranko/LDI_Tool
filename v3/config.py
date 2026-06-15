import datetime

# --- Tail Risk (Merton Jump-Diffusion Parameters) ---
# Simulates sudden macroeconomic shocks (fat negative tails)
JUMP_FREQUENCY_PER_YEAR = 0.06   # Expected number of macro shocks per year (e.g., 0.50 = one every 2 years)
JUMP_MEAN_SIZE = -0.15           # Mean log-jump size. Median simple jump is exp(-0.15)-1, about -14%.
JUMP_VOLATILITY = 0.06           # Standard deviation of log-jump size, not simple-return volatility.

# --- Heston Stochastic Volatility Parameters ---
HESTON_KAPPA = 5.0       # Mean-reversion speed (e.g., 3.0 means vol shocks decay over ~4 months)
HESTON_XI = 0.5          # Volatility of Volatility (How aggressively the VIX itself swings)
HESTON_RHO = -0.7        # Leverage Effect: When equities drop, variance violently spikes (highly negative correlation)

# --- Simulation Risk Limits ---
NUM_PATHS = 50000
MAX_MARGIN_CALL_PROBABILITY = 0.05

# Contribution-leverage policy search range.
# The optimizer searches L in [1.0, MAX_CONTRIBUTION_LEVERAGE].
MAX_CONTRIBUTION_LEVERAGE = 1.5

# Portfolio-leverage guardrails for applying monthly contributions.
#
# Rule:
#   leverage <= X      -> invest contribution at CONTRIBUTION_LEVERAGE
#   X < leverage < Y   -> invest contribution unlevered, i.e. 1.0x
#   leverage >= Y      -> deposit only; do not buy target asset
#
# Portfolio leverage is defined as:
#   gross_assets / equity
# where gross_assets includes target assets, legacy assets, and cash.
CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX = 1.7   # X
CONTRIBUTION_POLICY_NO_INVEST_MIN = 2.0       # Y

TARGET_DRIFT_CAP = 0.09
LEGACY_DRIFT_CAP = 0.09

# --- Optimizer Controls ---
RNG_SEED = 42
OPTIMIZER_GRID_POINTS = 25
OPTIMIZER_REFINEMENT_POINTS = 25
MONOTONICITY_TOLERANCE = 0.002

# --- Currency & FX Configuration ---
BASE_CURRENCY = "CHF"
TARGET_ASSET = "VT"

# --- Holdings Inventory ---
HOLDINGS = {
    "VT": 90,
    "ACWI.SW": 84,
    "CHDVD.SW": 51,
    "MEUD.PA": 32.6,
    "VTI": 28.6,
    "VNA.DE": 251,
    "FLIN": 132,
    "GIVN.SW": 1,
    "MC.PA": 6,
    "SAP.DE": 12,
    "FONC.SW": 28,
    "ANFO.SW": 40,
    "DGE.L": 56,
    "XS1970549561": 2  # Romanian Gov Bond
}

# --- Broker Margin Rules ---
MARGIN_REQUIREMENTS = {
    "VT": 0.25,
    "ACWI.SW": 0.25,
    "CHDVD.SW": 0.25,
    "MEUD.PA": 0.25,
    "VTI": 0.25,
    "VNA.DE": 0.25,
    "FLIN": 0.25,
    "GIVN.SW": 0.25,
    "MC.PA": 0.25,
    "SAP.DE": 0.25,
    "FONC.SW": 0.25,
    "ANFO.SW": 0.25,
    "DGE.L": 0.25,
    "XS1970549561": 0.15
}

# Explicitly declare the trading currency for every asset in HOLDINGS
ASSET_CURRENCIES = {
    "VT": "USD", 
    "ACWI.SW": "CHF", 
    "CHDVD.SW": "CHF", 
    "MEUD.PA": "EUR",
    "VTI": "USD", 
    "VNA.DE": "EUR", 
    "FLIN": "USD",
    "GIVN.SW": "CHF", 
    "MC.PA": "EUR", 
    "SAP.DE": "EUR",
    "FONC.SW": "CHF", 
    "ANFO.SW": "CHF", 
    "DGE.L": "GBX",
    "XS1970549561": "EUR"
}

# --- OTC & Fixed Income Registry ---
# Now accepts the native local price. The engine will handle the FX conversion.
OTC_REGISTRY = {
    "XS1970549561": {
        "live_price_local": 887.21, # Native EUR price
        "proxy_ticker": "EUN3.DE"  # EUR-denominated proxy ETF
    }
}

# --- Balance Sheet Cash State ---
CURRENT_DATE = datetime.date.today()
CURRENT_DEBT = 23282.32
TODAY_DEPOSIT = 0

# --- Future Projections Config ---
DEFAULT_MONTHLY_DEPOSIT_2026 = 1000.00
DEFAULT_MONTHLY_DEPOSIT_FUTURE = 2000.00
MARGIN_INTEREST_RATE = 0.015
HISTORICAL_LOOKBACK_YEARS = 20

# --- Simulation Horizon Buffer ---
# Simulate six calendar months plus roughly half a month after the last withdrawal.
POST_LAST_WITHDRAWAL_BUFFER_MONTHS = 6
POST_LAST_WITHDRAWAL_BUFFER_DAYS = 15

# --- Strict Liability Ledger ---
WITHDRAWAL_SCHEDULE = [
    {"date": datetime.date(2027, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2028, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2029, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2030, 12, 30), "amount": 5000.00},
    {"date": datetime.date(2031, 12, 30), "amount": 5000.00}
]