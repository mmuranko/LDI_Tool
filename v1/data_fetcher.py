import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from config import (
    HOLDINGS, MARGIN_REQUIREMENTS, HISTORICAL_LOOKBACK_YEARS,
    OTC_REGISTRY, ASSET_CURRENCIES, BASE_CURRENCY, TARGET_ASSET,
    TARGET_DRIFT_CAP, LEGACY_DRIFT_CAP, CURRENT_DATE
)

class DataEngine:
    """Handles data ingestion, FX conversion, and parameter estimation."""
    
    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)
        self.data = pd.DataFrame()
        self.current_prices_chf = {}

    @staticmethod
    def _extract_price_matrix(raw_data: pd.DataFrame, yf_query_list: list) -> pd.DataFrame:
        """Normalizes yfinance output into a DataFrame with one column per ticker."""
        if raw_data is None or raw_data.empty:
            raise ValueError("[!] SYSTEM HALT: yfinance returned no data at all.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            fields = raw_data.columns.get_level_values(0)
            price_field = "Adj Close" if "Adj Close" in fields else "Close"
            df = raw_data.xs(price_field, level=0, axis=1).copy()
        else:
            price_field = "Adj Close" if "Adj Close" in raw_data.columns else "Close"
            price_obj = raw_data[price_field]

            if isinstance(price_obj, pd.Series):
                if len(yf_query_list) != 1:
                    raise ValueError(
                        "[!] SYSTEM HALT: Single-series yfinance result received "
                        "for a multi-ticker query."
                    )
                df = price_obj.rename(yf_query_list[0]).to_frame()
            else:
                df = price_obj.copy()

        df.columns = [str(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df

    @staticmethod
    def _cap_drift(raw_mu: float, cap: float) -> float:
        """Caps only the upside drift estimate and protects against NaN/inf."""
        if not np.isfinite(raw_mu):
            return 0.0
        return min(float(raw_mu), cap)

    @staticmethod
    def _sanitize_corr_matrix(corr_matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Converts an empirical correlation matrix into a symmetric,
        finite, positive-definite correlation matrix suitable for Cholesky.
        """
        corr = np.asarray(corr_matrix, dtype=float)

        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError(f"[!] Invalid correlation matrix shape: {corr.shape}")

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, epsilon)

        repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
        repaired = (repaired + repaired.T) / 2.0

        scale = np.sqrt(np.clip(np.diag(repaired), epsilon, None))
        repaired = repaired / np.outer(scale, scale)

        repaired = np.clip(repaired, -1.0, 1.0)
        np.fill_diagonal(repaired, 1.0)

        return repaired

    def fetch_data(self) -> None:
            """Downloads historical data, spot prices, and applies vectorized FX conversion to CHF."""
            
            # 1. Validate configuration and identify required FX pairs.
            missing_currency = [t for t in self.tickers if t not in ASSET_CURRENCIES]
            if missing_currency:
                raise ValueError(f"[!] Missing ASSET_CURRENCIES entries: {missing_currency}")

            missing_margin = [t for t in self.tickers if t not in MARGIN_REQUIREMENTS]
            if missing_margin:
                raise ValueError(f"[!] Missing MARGIN_REQUIREMENTS entries: {missing_margin}")

            foreign_ccys = {
                ASSET_CURRENCIES[t]
                for t in self.tickers
                if ASSET_CURRENCIES[t] != BASE_CURRENCY
            }

            fx_tickers = {}
            for ccy in sorted(foreign_ccys):
                if ccy == "GBX":
                    fx_tickers["GBX"] = f"GBP{BASE_CURRENCY}=X"
                else:
                    fx_tickers[ccy] = f"{ccy}{BASE_CURRENCY}=X"

            # 2. Build the unified yfinance query list.
            yf_query_list = list(fx_tickers.values())

            for ticker in self.tickers:
                if ticker in OTC_REGISTRY:
                    yf_query_list.append(OTC_REGISTRY[ticker]["proxy_ticker"])
                else:
                    yf_query_list.append(ticker)

            yf_query_list = list(dict.fromkeys(yf_query_list))

            if not yf_query_list:
                raise ValueError("[!] SYSTEM HALT: No tickers available for yfinance download.")

            print(f"[*] Fetching market data and {list(fx_tickers.values())} FX rates...")

            download_query = yf_query_list[0] if len(yf_query_list) == 1 else yf_query_list

            raw_data = yf.download(
                download_query,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )

            df = self._extract_price_matrix(raw_data, yf_query_list)

            # 3. Validate FX columns before using them.
            fx_columns = list(dict.fromkeys(fx_tickers.values()))
            missing_fx = [fx for fx in fx_columns if fx not in df.columns]

            if missing_fx:
                raise ValueError(
                    "\n[!] SYSTEM HALT: Required FX series missing from API response.\n"
                    f"    Missing FX tickers: {missing_fx}\n"
                    f"    Received columns: {list(df.columns)}"
                )

            # Forward-fill FX rates to bridge weekend/holiday gaps.
            for fx in fx_columns:
                df[fx] = df[fx].ffill()

            self.raw_local_data = df.copy()
            self.raw_fx_data = df[fx_columns].copy() if fx_columns else pd.DataFrame(index=df.index)

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
        v_target_0 = HOLDINGS[TARGET_ASSET] * self.current_prices_chf[TARGET_ASSET]

        legacy_by_currency = {}
        v_legacy_0 = 0.0
        leg_mm_total = 0.0

        for ticker in self.tickers:
            if ticker == TARGET_ASSET:
                continue

            value_chf = HOLDINGS[ticker] * self.current_prices_chf[ticker]
            margin_chf = value_chf * MARGIN_REQUIREMENTS[ticker]
            ccy = ASSET_CURRENCIES[ticker]

            v_legacy_0 += value_chf
            leg_mm_total += margin_chf

            if ccy not in legacy_by_currency:
                legacy_by_currency[ccy] = {
                    "v0": 0.0,
                    "maintenance_margin_chf": 0.0,
                    "tickers": []
                }

            legacy_by_currency[ccy]["v0"] += value_chf
            legacy_by_currency[ccy]["maintenance_margin_chf"] += margin_chf
            legacy_by_currency[ccy]["tickers"].append(ticker)

        for bucket in legacy_by_currency.values():
            bucket["m"] = (
                bucket["maintenance_margin_chf"] / bucket["v0"]
                if bucket["v0"] > 0
                else 0.0
            )

        m_legacy = leg_mm_total / v_legacy_0 if v_legacy_0 > 0 else 0.0

        return {
            "v_target_0": v_target_0,
            "v_legacy_0": v_legacy_0,
            "m_target": MARGIN_REQUIREMENTS[TARGET_ASSET],
            "m_legacy": m_legacy,
            "legacy_by_currency": legacy_by_currency
        }

    def estimate_parameters(self) -> dict:
        """Computes parameters for the current 3-factor model: target, legacy local, legacy FX."""

        # --- 1. Target Asset in CHF ---
        target_simple = self.data[TARGET_ASSET].pct_change(fill_method=None).dropna()

        if target_simple.empty:
            raise ValueError(f"[!] SYSTEM HALT: Not enough return data for target asset {TARGET_ASSET}.")

        target_ret = np.log1p(target_simple)

        sigma_target = float(target_ret.std() * np.sqrt(252))
        mu_target_raw = float((target_ret.mean() * 252) + (sigma_target ** 2 / 2))
        mu_target = self._cap_drift(mu_target_raw, TARGET_DRIFT_CAP)

        # --- 2. Legacy Basket ---
        legacy_tickers_all = [t for t in self.tickers if t != TARGET_ASSET]

        legacy_values_all = np.array([
            HOLDINGS[t] * self.current_prices_chf[t]
            for t in legacy_tickers_all
        ], dtype=float)

        active_legacy = [
            (ticker, value)
            for ticker, value in zip(legacy_tickers_all, legacy_values_all)
            if value > 0
        ]

        # Robust path: no legacy assets.
        if not active_legacy:
            zero = pd.Series(0.0, index=target_ret.index)

            aligned_df = pd.concat(
                [
                    target_ret.rename("target"),
                    zero.rename("legacy_local"),
                    zero.rename("legacy_fx")
                ],
                axis=1
            ).dropna()

            corr_matrix = self._sanitize_corr_matrix(aligned_df.corr().values)

            return {
                "mu_target_raw": mu_target_raw,
                "mu_target": mu_target,
                "sigma_target": sigma_target,

                "mu_legacy_loc_raw": 0.0,
                "mu_legacy_loc": 0.0,
                "sigma_legacy_loc": 0.0,

                "mu_fx_raw": 0.0,
                "mu_fx": 0.0,
                "sigma_fx": 0.0,

                "corr_matrix": corr_matrix
            }

        legacy_tickers = [x[0] for x in active_legacy]
        legacy_values = np.array([x[1] for x in active_legacy], dtype=float)
        legacy_weights = legacy_values / legacy_values.sum()

        legacy_query_targets = [
            OTC_REGISTRY[t]["proxy_ticker"] if t in OTC_REGISTRY else t
            for t in legacy_tickers
        ]

        local_prices = self.raw_local_data[legacy_query_targets].copy()

        if isinstance(local_prices, pd.Series):
            local_prices = local_prices.to_frame()

        local_simple = local_prices.pct_change(fill_method=None).dropna(how="any")
        local_simple.columns = legacy_tickers

        if local_simple.empty:
            raise ValueError("[!] SYSTEM HALT: Not enough local return data for legacy assets.")

        # --- 3. Legacy Basket FX Returns ---
        fx_series_list = []

        for ticker in legacy_tickers:
            ccy = ASSET_CURRENCIES[ticker]

            if ccy == BASE_CURRENCY:
                fx_series_list.append(pd.Series(0.0, index=local_simple.index, name=ticker))
            else:
                fx_col = f"GBP{BASE_CURRENCY}=X" if ccy == "GBX" else f"{ccy}{BASE_CURRENCY}=X"

                if fx_col not in self.raw_fx_data.columns:
                    raise ValueError(f"[!] SYSTEM HALT: Missing FX column during parameter estimation: {fx_col}")

                fx_series = self.raw_fx_data[fx_col].pct_change(fill_method=None).rename(ticker)
                fx_series_list.append(fx_series)

        fx_simple_df = pd.concat(fx_series_list, axis=1)

        common_idx = local_simple.index.intersection(fx_simple_df.dropna(how="any").index)
        local_simple = local_simple.loc[common_idx]
        fx_simple_df = fx_simple_df.loc[common_idx]

        if local_simple.empty or fx_simple_df.empty:
            raise ValueError("[!] SYSTEM HALT: No overlapping local/FX return history for legacy basket.")

        legacy_local_simple = (local_simple * legacy_weights).sum(axis=1)
        legacy_local_ret = pd.Series(np.log1p(legacy_local_simple), index=local_simple.index)

        sigma_legacy_loc = float(legacy_local_ret.std() * np.sqrt(252))
        mu_legacy_loc_raw = float((legacy_local_ret.mean() * 252) + (sigma_legacy_loc ** 2 / 2))
        mu_legacy_loc = self._cap_drift(mu_legacy_loc_raw, LEGACY_DRIFT_CAP)

        legacy_fx_simple = (fx_simple_df * legacy_weights).sum(axis=1)
        legacy_fx_ret = pd.Series(np.log1p(legacy_fx_simple), index=fx_simple_df.index)

        sigma_fx = float(legacy_fx_ret.std() * np.sqrt(252))
        mu_fx_raw = float((legacy_fx_ret.mean() * 252) + (sigma_fx ** 2 / 2))

        # Deliberate modeling assumption: FX has zero expected drift.
        mu_fx = 0.0

        # --- 4. Sanitized 3x3 Correlation Matrix ---
        aligned_df = pd.concat(
            [
                target_ret.rename("target"),
                legacy_local_ret.rename("legacy_local"),
                legacy_fx_ret.rename("legacy_fx")
            ],
            axis=1
        ).dropna()

        corr_matrix = self._sanitize_corr_matrix(aligned_df.corr().values)

        return {
            "mu_target_raw": mu_target_raw,
            "mu_target": mu_target,
            "sigma_target": sigma_target,

            "mu_legacy_loc_raw": mu_legacy_loc_raw,
            "mu_legacy_loc": mu_legacy_loc,
            "sigma_legacy_loc": sigma_legacy_loc,

            "mu_fx_raw": mu_fx_raw,
            "mu_fx": mu_fx,
            "sigma_fx": sigma_fx,

            "corr_matrix": corr_matrix
        }