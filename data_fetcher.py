import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import track

from config import (
    ASSET_CURRENCIES, BASE_CURRENCY, CURRENT_DATE, HISTORICAL_LOOKBACK_YEARS,
    HOLDINGS, DRIFT_CAP, IM_REQUIREMENTS, MM_REQUIREMENTS, OTC_REGISTRY,
    ACTIVE_ASSET
)

console = Console()

class DataEngine:
    """Handles data ingestion, FX conversion, and dynamic factor estimation."""

    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)

        self.data = pd.DataFrame()
        self.raw_local_data = pd.DataFrame()
        self.raw_fx_data = pd.DataFrame()

        self.current_prices_local = {}
        self.current_prices_base = {}
        self.fx_tickers = {}

    @staticmethod
    def _extract_price_matrix(raw_data: pd.DataFrame, yf_query_list: list) -> pd.DataFrame:
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
                    raise ValueError("[!] SYSTEM HALT: Single-series yfinance result received for a multi-ticker query.")
                df = price_obj.rename(yf_query_list[0]).to_frame()
            else:
                df = price_obj.copy()

        df.columns = [str(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df

    @staticmethod
    def _cap_drift(raw_mu: float, cap: float) -> float:
        if not np.isfinite(raw_mu): return 0.0
        return min(float(raw_mu), cap)

    @staticmethod
    def _sanitize_corr_matrix(corr_matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        corr = np.asarray(corr_matrix, dtype=float)
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

    @staticmethod
    def _log_returns_from_prices(prices):
        prices = prices.sort_index().ffill()
        simple = prices.pct_change(fill_method=None)
        simple = simple.where(simple > -1.0)
        return np.log1p(simple)

    @staticmethod
    def _fx_column_for_currency(ccy: str) -> str:
        if ccy == "GBX": return f"GBP{BASE_CURRENCY}=X"
        return f"{ccy}{BASE_CURRENCY}=X"

    def _query_ticker_for_asset(self, ticker: str) -> str:
        return OTC_REGISTRY[ticker]["proxy_ticker"] if ticker in OTC_REGISTRY else ticker

    def fetch_data(self) -> None:
        if ACTIVE_ASSET not in HOLDINGS: 
            raise ValueError(f"[!] ACTIVE_ASSET {ACTIVE_ASSET!r} is not in HOLDINGS.")

        foreign_ccys = {ASSET_CURRENCIES[t] for t in self.tickers if ASSET_CURRENCIES[t] != BASE_CURRENCY}
        self.fx_tickers = {ccy: self._fx_column_for_currency(ccy) for ccy in sorted(foreign_ccys)}

        yf_query_list = list(self.fx_tickers.values())
        for ticker in self.tickers: yf_query_list.append(self._query_ticker_for_asset(ticker))
        yf_query_list = list(dict.fromkeys(yf_query_list))

        with console.status(f"Downloading history for {len(yf_query_list)} assets & FX rates...", spinner="dots"):
            raw_data = yf.download(yf_query_list[0] if len(yf_query_list) == 1 else yf_query_list,
                                   start=self.start_date, end=self.end_date, auto_adjust=False, progress=False)
            df = self._extract_price_matrix(raw_data, yf_query_list)

        fx_columns = list(dict.fromkeys(self.fx_tickers.values()))
        for fx in fx_columns: df[fx] = df[fx].ffill()

        self.raw_local_data = df.copy()
        self.raw_fx_data = df[fx_columns].copy() if fx_columns else pd.DataFrame(index=df.index)

        for ticker in track(self.tickers, description="Normalizing Asset Matrix...", console=console):
            ccy = ASSET_CURRENCIES[ticker]
            fx_ticker = self.fx_tickers.get(ccy)
            query_target = self._query_ticker_for_asset(ticker)

            clean_series = df[query_target].dropna()
            local_spot = float(OTC_REGISTRY[ticker]["live_price_local"]) if ticker in OTC_REGISTRY else float(clean_series.iloc[-1])

            self.current_prices_local[ticker] = local_spot
            self.data[ticker] = df[query_target]

            if ccy == BASE_CURRENCY:
                self.current_prices_base[ticker] = local_spot
            else:
                live_fx_rate = float(df[fx_ticker].dropna().iloc[-1])
                if ccy == "GBX":
                    self.current_prices_base[ticker] = local_spot * (live_fx_rate / 100.0)
                    self.data[ticker] = self.data[ticker] * (df[fx_ticker] / 100.0)
                else:
                    self.current_prices_base[ticker] = local_spot * live_fx_rate
                    self.data[ticker] = self.data[ticker] * df[fx_ticker]

        self.data.dropna(how="all", inplace=True)
        console.print("✓ [green]Unified market data ingested successfully.[/green]")

    def build_current_state(self) -> dict:
        # Force the Active Asset to be exactly at Index 0.
        portfolio_order = [ACTIVE_ASSET]
        for ticker in self.tickers:
            if ticker != ACTIVE_ASSET and (HOLDINGS[ticker] * self.current_prices_base[ticker]) > 0:
                portfolio_order.append(ticker)

        assets_dict = {}
        for ticker in portfolio_order:
            quantity = float(HOLDINGS[ticker])
            value_base = float(quantity * self.current_prices_base[ticker])
            
            assets_dict[ticker] = {
                "ticker": ticker, 
                "currency": ASSET_CURRENCIES[ticker], 
                "quantity": quantity,
                "v0": value_base, 
                "mmr": float(MM_REQUIREMENTS[ticker]), 
                "imr": float(IM_REQUIREMENTS[ticker]),
            }

        return {
            "portfolio_order": portfolio_order,
            "assets_dict": assets_dict,
            "active_asset": ACTIVE_ASSET
        }

    def estimate_parameters(self) -> dict:
        with console.status("Estimating unified covariance matrix...", spinner="dots"):
            state = self.build_current_state()
            portfolio_order = state["portfolio_order"]
            
            factor_names = []
            factor_types = {}
            factor_currencies = {}
            factor_returns = []
            assets_params = {}

            # Build factors for every asset in the portfolio uniformly
            for ticker in portfolio_order:
                query_ticker = self._query_ticker_for_asset(ticker)
                factor_name = f"asset:{ticker}"
                ret = self._log_returns_from_prices(self.raw_local_data[query_ticker].rename(factor_name)).dropna()

                sigma = float(ret.std() * np.sqrt(252))
                mu_raw = float((ret.mean() * 252) + (sigma**2 / 2))
                mu = self._cap_drift(mu_raw, DRIFT_CAP)

                factor_names.append(factor_name)
                factor_types[factor_name] = "asset"
                factor_currencies[factor_name] = ASSET_CURRENCIES[ticker]
                factor_returns.append(ret.rename(factor_name))

                assets_params[ticker] = {
                    "factor_name": factor_name, "currency": ASSET_CURRENCIES[ticker], 
                    "mu": mu, "sigma": sigma
                }

            # Build factors for required FX
            required_fx_ccys = {ASSET_CURRENCIES[t] for t in portfolio_order if ASSET_CURRENCIES[t] != BASE_CURRENCY}
            fx_factors = {}
            
            for ccy in sorted(required_fx_ccys):
                fx_col = self._fx_column_for_currency(ccy)
                factor_name = f"fx:{ccy}"
                fx_ret = self._log_returns_from_prices(self.raw_fx_data[fx_col].rename(factor_name)).dropna()

                sigma = float(fx_ret.std() * np.sqrt(252))
                factor_names.append(factor_name)
                factor_types[factor_name] = "fx"
                factor_returns.append(fx_ret.rename(factor_name))

                fx_factors[ccy] = {"factor_name": factor_name, "mu": 0.0, "sigma": sigma}

            aligned_df = pd.concat(factor_returns, axis=1).dropna(how="any")
            corr_matrix = self._sanitize_corr_matrix(aligned_df.corr().values)

            mu_by_factor, sigma_by_factor = {}, {}
            for d in list(assets_params.values()) + list(fx_factors.values()):
                mu_by_factor[d["factor_name"]] = d["mu"]
                sigma_by_factor[d["factor_name"]] = d["sigma"]

        console.print("✓ [green]Dynamic multifactor model assembled.[/green]")
        return {
            "assets_params": assets_params, "fx_factors": fx_factors, 
            "factor_names": factor_names, "factor_types": factor_types,
            "mu_by_factor": mu_by_factor, "sigma_by_factor": sigma_by_factor,
            "corr_matrix": corr_matrix, "base_currency": BASE_CURRENCY
        }