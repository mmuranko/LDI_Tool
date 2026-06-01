import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from config import (HOLDINGS, MARGIN_REQUIREMENTS, HISTORICAL_LOOKBACK_YEARS, 
                    OTC_REGISTRY, ASSET_CURRENCIES, BASE_CURRENCY, TARGET_ASSET, 
                    ACWI_DRIFT_CAP, LEGACY_DRIFT_CAP)

class DataEngine:
    """Handles data ingestion, FX conversion, and parameter estimation."""
    
    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = datetime.date.today()
        self.start_date = self.end_date - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)
        self.data = pd.DataFrame()
        self.current_prices_chf = {}

    def fetch_data(self) -> None:
            """Downloads historical data, spot prices, and applies vectorized FX conversion to CHF."""
            
            # 1. Identify required FX pairs (Map GBX to GBP for the yfinance query)
            foreign_ccys = set(ASSET_CURRENCIES.values()) - {BASE_CURRENCY}
            fx_tickers = {}
            for ccy in foreign_ccys:
                if ccy == "GBX":
                    fx_tickers["GBX"] = f"GBP{BASE_CURRENCY}=X"
                else:
                    fx_tickers[ccy] = f"{ccy}{BASE_CURRENCY}=X"
            
            # 2. Build the unified yfinance query list 
            yf_query_list = list(set(fx_tickers.values()))
            for ticker in self.tickers:
                if ticker in OTC_REGISTRY:
                    yf_query_list.append(OTC_REGISTRY[ticker]["proxy_ticker"])
                else:
                    yf_query_list.append(ticker)
                    
            # Deduplicate to prevent API request warnings
            yf_query_list = list(set(yf_query_list))
                    
            print(f"[*] Fetching market data and {list(fx_tickers.values())} FX rates...")
            
            # Silence the default yfinance progress bar with progress=False
            raw_data = yf.download(yf_query_list, start=self.start_date, end=self.end_date, 
                                   auto_adjust=False, progress=False)
            
            # 3. Safe Extraction of the Price Matrix
            if isinstance(raw_data.columns, pd.MultiIndex):
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    df = raw_data.xs('Adj Close', level=0, axis=1).copy()
                else:
                    df = raw_data.xs('Close', level=0, axis=1).copy()
            else:
                price_col = 'Adj Close' if 'Adj Close' in raw_data.columns else 'Close'
                df = pd.DataFrame(raw_data[price_col], columns=yf_query_list)
                
            # Forward-fill FX rates first to bridge weekend/holiday gaps
            for fx in set(fx_tickers.values()):
                if fx in df.columns:
                    df[fx] = df[fx].ffill()

            self.raw_local_data = df.copy() # The pure asset prices
            self.raw_fx_data = df[list(fx_tickers.values())].copy() # The pure FX rates

            # 4. Process Live Prices & Historical DataFrame with Tripwires
            # Wrap the ticker loop in a beautiful tqdm progress bar
            for ticker in tqdm(self.tickers, desc="Processing Asset Data", bar_format="{l_bar}{bar:30}{r_bar}", colour="blue"):
                ccy = ASSET_CURRENCIES[ticker]
                fx_ticker = fx_tickers.get(ccy)
                
                # --- Tripwire: Ensure ticker exists in the downloaded data ---
                query_target = OTC_REGISTRY[ticker]["proxy_ticker"] if ticker in OTC_REGISTRY else ticker
                if query_target not in df.columns:
                    raise ValueError(f"\n[!] SYSTEM HALT: '{query_target}' is completely missing from the API response.\n"
                                    f"    Check Yahoo Finance to see if the ticker is delisted or mistyped.")
                
                clean_series = df[query_target].dropna()
                if clean_series.empty:
                    raise ValueError(f"\n[!] SYSTEM HALT: '{query_target}' returned empty price data.\n"
                                    f"    It may be temporarily unavailable or delisted.")
                
                # --- Handle Live Spot Prices ---
                if ticker in OTC_REGISTRY:
                    local_spot = OTC_REGISTRY[ticker]["live_price_local"]
                    self.data[ticker] = clean_series # Route proxy history to the native ticker
                else:
                    local_spot = clean_series.iloc[-1]
                    self.data[ticker] = clean_series
                
                # --- Apply FX Conversion to Live Spot ---
                if ccy == BASE_CURRENCY:
                    self.current_prices_chf[ticker] = local_spot
                else:
                    if fx_ticker not in df.columns or df[fx_ticker].dropna().empty:
                        raise ValueError(f"\n[!] SYSTEM HALT: FX Rate '{fx_ticker}' failed to download.")
                        
                    live_fx_rate = df[fx_ticker].dropna().iloc[-1]
                    
                    # Intercept the GBX fractional requirement
                    if ccy == "GBX":
                        self.current_prices_chf[ticker] = local_spot * (live_fx_rate / 100.0)
                    else:
                        self.current_prices_chf[ticker] = local_spot * live_fx_rate
                    
                # --- Handle Historical Time-Series FX Conversion ---
                if ccy != BASE_CURRENCY:
                    if ccy == "GBX":
                        self.data[ticker] = self.data[ticker] * (df[fx_ticker] / 100.0)
                    else:
                        self.data[ticker] = self.data[ticker] * df[fx_ticker]

            # Clean memory
            self.data.dropna(how='all', inplace=True)

    def build_current_state(self) -> dict:
        """Calculates today's explicit initial values using CHF spot prices."""
        v_acwi_0 = HOLDINGS[TARGET_ASSET] * self.current_prices_chf[TARGET_ASSET]
        v_legacy_0 = sum(HOLDINGS[t] * self.current_prices_chf[t] for t in self.tickers if t != TARGET_ASSET)
        
        leg_mm_total = sum(HOLDINGS[t] * self.current_prices_chf[t] * MARGIN_REQUIREMENTS[t] 
                           for t in self.tickers if t != TARGET_ASSET)
        m_legacy = leg_mm_total / v_legacy_0 if v_legacy_0 > 0 else 0.0
        
        return {
            "v_acwi_0": v_acwi_0,
            "v_legacy_0": v_legacy_0,
            "m_acwi": MARGIN_REQUIREMENTS[TARGET_ASSET],
            "m_legacy": m_legacy
        }

    def estimate_parameters(self) -> dict:
        """Computes parameters for a 3-Factor Model: Target, Legacy Local, and Legacy FX."""
        
        # --- 1. Target Asset (ACWI in CHF) ---
        acwi_simple = self.data[TARGET_ASSET].pct_change().dropna()
        acwi_ret = np.log(1 + acwi_simple)
        
        sigma_acwi = acwi_ret.std() * np.sqrt(252)
        mu_acwi = min((acwi_ret.mean() * 252) + (sigma_acwi**2 / 2), ACWI_DRIFT_CAP)
        
        # --- 2. Legacy Basket (LOCAL Asset Returns) ---
        legacy_tickers = [t for t in self.tickers if t != TARGET_ASSET]
        
        # Weights based on today's exposure
        legacy_weights = np.array([HOLDINGS[t] * self.current_prices_chf[t] for t in legacy_tickers])
        legacy_weights = legacy_weights / legacy_weights.sum()
        
        # [FIX]: Map holding tickers to their yfinance proxies to read from the raw downloaded data
        legacy_query_targets = [OTC_REGISTRY[t]["proxy_ticker"] if t in OTC_REGISTRY else t for t in legacy_tickers]
        
        # Extract purely local returns (NO FX influence)
        local_simple = self.raw_local_data[legacy_query_targets].pct_change().dropna()
        local_simple.columns = legacy_tickers # Rename columns back to internal holding keys for safety
        
        legacy_local_simple = (local_simple * legacy_weights).sum(axis=1)
        legacy_local_ret = np.log(1 + legacy_local_simple)
        
        sigma_legacy_loc = legacy_local_ret.std() * np.sqrt(252)
        mu_legacy_loc = min((legacy_local_ret.mean() * 252) + (sigma_legacy_loc**2 / 2), LEGACY_DRIFT_CAP)

        # --- 3. Legacy Basket (FX Returns) ---
        # Map each legacy asset to its respective FX series
        fx_series_list = []
        for t in legacy_tickers:
            ccy = ASSET_CURRENCIES[t]
            if ccy == BASE_CURRENCY:
                fx_series_list.append(pd.Series(0, index=local_simple.index)) # No FX risk for CHF assets
            else:
                fx_col = f"GBP{BASE_CURRENCY}=X" if ccy == "GBX" else f"{ccy}{BASE_CURRENCY}=X"
                fx_series_list.append(self.raw_fx_data[fx_col].pct_change())
                
        fx_simple_df = pd.concat(fx_series_list, axis=1).dropna()
        legacy_fx_simple = (fx_simple_df.values * legacy_weights).sum(axis=1)
        legacy_fx_ret = pd.Series(np.log(1 + legacy_fx_simple), index=fx_simple_df.index)
        
        sigma_fx = legacy_fx_ret.std() * np.sqrt(252)
        # mu_fx = (legacy_fx_ret.mean() * 252) + (sigma_fx**2 / 2)
        mu_fx = 0.00 # Assumes FX is a pure random walk, otherwise CHF value grows indefinitely

        # --- 4. The 3x3 Correlation Matrix ---
        # Align dates mathematically
        aligned_df = pd.concat([acwi_ret, legacy_local_ret, legacy_fx_ret], axis=1).dropna()
        corr_matrix = aligned_df.corr().values

        return {
            "mu_acwi": mu_acwi, "sigma_acwi": sigma_acwi,
            "mu_legacy_loc": mu_legacy_loc, "sigma_legacy_loc": sigma_legacy_loc,
            "mu_fx": mu_fx, "sigma_fx": sigma_fx,
            "corr_matrix": corr_matrix # Replaces the scalar 'rho'
        }