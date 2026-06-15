import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import (
    ASSET_CURRENCIES,
    BASE_CURRENCY,
    CURRENT_DATE,
    HISTORICAL_LOOKBACK_YEARS,
    HOLDINGS,
    LEGACY_DRIFT_CAP,
    MARGIN_REQUIREMENTS,
    OTC_REGISTRY,
    TARGET_ASSET,
    TARGET_DRIFT_CAP,
)


class DataEngine:
    """Handles data ingestion, FX conversion, and dynamic factor estimation."""

    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)

        # CHF-converted historical prices, kept mainly for diagnostics/backward compatibility.
        self.data = pd.DataFrame()

        # Native local prices and raw FX levels used for parameter estimation.
        self.raw_local_data = pd.DataFrame()
        self.raw_fx_data = pd.DataFrame()

        self.current_prices_local = {}
        self.current_prices_chf = {}
        self.fx_tickers = {}

    @staticmethod
    def _extract_price_matrix(raw_data: pd.DataFrame, yf_query_list: list) -> pd.DataFrame:
        """Normalizes yfinance output into a DataFrame with one column per query ticker."""
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
        Converts an empirical correlation matrix into a symmetric, finite,
        positive-definite correlation matrix suitable for Cholesky.
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

    @staticmethod
    def _log_returns_from_prices(prices):
        """Computes robust log returns from forward-filled price levels."""
        prices = prices.sort_index().ffill()
        simple = prices.pct_change(fill_method=None)
        simple = simple.where(simple > -1.0)
        return np.log1p(simple)

    @staticmethod
    def _fx_column_for_currency(ccy: str) -> str:
        """Maps a trading currency to the Yahoo FX ticker against the base currency."""
        if ccy == "GBX":
            # Yahoo quotes GBPCHF, while London stocks are usually priced in pence.
            # Returns are identical because the /100 GBX conversion is a constant scale.
            return f"GBP{BASE_CURRENCY}=X"
        return f"{ccy}{BASE_CURRENCY}=X"

    def _query_ticker_for_asset(self, ticker: str) -> str:
        """Returns the Yahoo ticker used for historical local prices."""
        return OTC_REGISTRY[ticker]["proxy_ticker"] if ticker in OTC_REGISTRY else ticker

    def fetch_data(self) -> None:
        """Downloads historical data, spot prices, and CHF-converted diagnostics."""
        missing_currency = [t for t in self.tickers if t not in ASSET_CURRENCIES]
        if missing_currency:
            raise ValueError(f"[!] Missing ASSET_CURRENCIES entries: {missing_currency}")

        missing_margin = [t for t in self.tickers if t not in MARGIN_REQUIREMENTS]
        if missing_margin:
            raise ValueError(f"[!] Missing MARGIN_REQUIREMENTS entries: {missing_margin}")

        if TARGET_ASSET not in HOLDINGS:
            raise ValueError(f"[!] TARGET_ASSET {TARGET_ASSET!r} is not present in HOLDINGS.")

        foreign_ccys = {
            ASSET_CURRENCIES[t]
            for t in self.tickers
            if ASSET_CURRENCIES[t] != BASE_CURRENCY
        }

        self.fx_tickers = {
            ccy: self._fx_column_for_currency(ccy)
            for ccy in sorted(foreign_ccys)
        }

        yf_query_list = list(self.fx_tickers.values())

        for ticker in self.tickers:
            yf_query_list.append(self._query_ticker_for_asset(ticker))

        yf_query_list = list(dict.fromkeys(yf_query_list))

        if not yf_query_list:
            raise ValueError("[!] SYSTEM HALT: No tickers available for yfinance download.")

        print(f"[*] Fetching market data and {list(self.fx_tickers.values())} FX rates...")

        download_query = yf_query_list[0] if len(yf_query_list) == 1 else yf_query_list

        raw_data = yf.download(
            download_query,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
        )

        df = self._extract_price_matrix(raw_data, yf_query_list)

        fx_columns = list(dict.fromkeys(self.fx_tickers.values()))
        missing_fx = [fx for fx in fx_columns if fx not in df.columns]

        if missing_fx:
            raise ValueError(
                "\n[!] SYSTEM HALT: Required FX series missing from API response.\n"
                f"    Missing FX tickers: {missing_fx}\n"
                f"    Received columns: {list(df.columns)}"
            )

        for fx in fx_columns:
            df[fx] = df[fx].ffill()

        self.raw_local_data = df.copy()
        self.raw_fx_data = df[fx_columns].copy() if fx_columns else pd.DataFrame(index=df.index)

        for ticker in tqdm(
            self.tickers,
            desc="Processing Asset Data",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="blue",
        ):
            ccy = ASSET_CURRENCIES[ticker]
            fx_ticker = self.fx_tickers.get(ccy)
            query_target = self._query_ticker_for_asset(ticker)

            if query_target not in df.columns:
                raise ValueError(
                    f"\n[!] SYSTEM HALT: '{query_target}' is completely missing from the API response.\n"
                    "    Check Yahoo Finance to see if the ticker is delisted or mistyped."
                )

            clean_series = df[query_target].dropna()
            if clean_series.empty:
                raise ValueError(
                    f"\n[!] SYSTEM HALT: '{query_target}' returned empty price data.\n"
                    "    It may be temporarily unavailable or delisted."
                )

            if ticker in OTC_REGISTRY:
                local_spot = float(OTC_REGISTRY[ticker]["live_price_local"])
            else:
                local_spot = float(clean_series.iloc[-1])

            self.current_prices_local[ticker] = local_spot

            # CHF-converted history is useful for diagnostics, but estimation uses local prices.
            self.data[ticker] = df[query_target]

            if ccy == BASE_CURRENCY:
                self.current_prices_chf[ticker] = local_spot
            else:
                if fx_ticker not in df.columns or df[fx_ticker].dropna().empty:
                    raise ValueError(f"\n[!] SYSTEM HALT: FX Rate '{fx_ticker}' failed to download.")

                live_fx_rate = float(df[fx_ticker].dropna().iloc[-1])
                if ccy == "GBX":
                    self.current_prices_chf[ticker] = local_spot * (live_fx_rate / 100.0)
                    self.data[ticker] = self.data[ticker] * (df[fx_ticker] / 100.0)
                else:
                    self.current_prices_chf[ticker] = local_spot * live_fx_rate
                    self.data[ticker] = self.data[ticker] * df[fx_ticker]

        self.data.dropna(how="all", inplace=True)

    def build_current_state(self) -> dict:
        """Calculates today's balance sheet and exact per-asset legacy state."""
        v_target_0 = float(HOLDINGS[TARGET_ASSET] * self.current_prices_chf[TARGET_ASSET])

        legacy_assets = {}
        legacy_by_currency = {}
        legacy_asset_order = []
        v_legacy_0 = 0.0
        leg_mm_total = 0.0

        for ticker in self.tickers:
            if ticker == TARGET_ASSET:
                continue

            quantity = float(HOLDINGS[ticker])
            value_chf = float(quantity * self.current_prices_chf[ticker])
            margin_rate = float(MARGIN_REQUIREMENTS[ticker])
            margin_chf = value_chf * margin_rate
            ccy = ASSET_CURRENCIES[ticker]

            v_legacy_0 += value_chf
            leg_mm_total += margin_chf

            if value_chf > 0:
                legacy_asset_order.append(ticker)

            legacy_assets[ticker] = {
                "ticker": ticker,
                "currency": ccy,
                "quantity": quantity,
                "v0": value_chf,
                "m": margin_rate,
                "maintenance_margin_chf": margin_chf,
            }

            if ccy not in legacy_by_currency:
                legacy_by_currency[ccy] = {
                    "v0": 0.0,
                    "maintenance_margin_chf": 0.0,
                    "tickers": [],
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
            "m_target": float(MARGIN_REQUIREMENTS[TARGET_ASSET]),
            "m_legacy": m_legacy,
            "legacy_assets": legacy_assets,
            "legacy_asset_order": legacy_asset_order,
            "legacy_by_currency": legacy_by_currency,
        }

    def estimate_parameters(self) -> dict:
        """
        Builds a dynamic factor model:
          - one local target factor,
          - one local factor per active legacy asset,
          - one zero-drift FX factor per required non-base currency.

        The empirical correlation matrix is estimated jointly across all factors,
        so covariances between target, legacy assets, and currencies are retained.
        """
        if self.raw_local_data.empty:
            raise ValueError("[!] fetch_data() must be called before estimate_parameters().")

        target_ccy = ASSET_CURRENCIES[TARGET_ASSET]
        target_query = self._query_ticker_for_asset(TARGET_ASSET)

        if target_query not in self.raw_local_data.columns:
            raise ValueError(f"[!] Missing target local history column: {target_query}")

        target_ret = self._log_returns_from_prices(
            self.raw_local_data[target_query].rename("target")
        ).dropna()

        if target_ret.empty:
            raise ValueError(f"[!] SYSTEM HALT: Not enough return data for target asset {TARGET_ASSET}.")

        sigma_target = float(target_ret.std() * np.sqrt(252))
        mu_target_raw = float((target_ret.mean() * 252) + (sigma_target**2 / 2))
        mu_target = self._cap_drift(mu_target_raw, TARGET_DRIFT_CAP)

        legacy_tickers_all = [t for t in self.tickers if t != TARGET_ASSET]
        active_legacy = [
            t
            for t in legacy_tickers_all
            if float(HOLDINGS[t]) * float(self.current_prices_chf[t]) > 0.0
        ]

        required_fx_ccys = {
            ccy
            for ccy in [target_ccy] + [ASSET_CURRENCIES[t] for t in active_legacy]
            if ccy != BASE_CURRENCY
        }

        factor_names = ["target"]
        factor_types = {"target": "target"}
        factor_currencies = {"target": target_ccy}
        factor_tickers = {"target": TARGET_ASSET}

        factor_returns = [target_ret.rename("target")]

        target_factor = {
            "name": "target",
            "ticker": TARGET_ASSET,
            "currency": target_ccy,
            "mu_raw": mu_target_raw,
            "mu": mu_target,
            "sigma": sigma_target,
        }

        legacy_assets = {}

        for ticker in active_legacy:
            query_ticker = self._query_ticker_for_asset(ticker)
            if query_ticker not in self.raw_local_data.columns:
                raise ValueError(f"[!] Missing local history column for {ticker}: {query_ticker}")

            factor_name = f"asset:{ticker}"
            ret = self._log_returns_from_prices(
                self.raw_local_data[query_ticker].rename(factor_name)
            ).dropna()

            if ret.empty:
                raise ValueError(f"[!] SYSTEM HALT: Not enough local return data for legacy asset {ticker}.")

            sigma = float(ret.std() * np.sqrt(252))
            mu_raw = float((ret.mean() * 252) + (sigma**2 / 2))
            mu = self._cap_drift(mu_raw, LEGACY_DRIFT_CAP)
            value_chf = float(HOLDINGS[ticker] * self.current_prices_chf[ticker])
            margin_rate = float(MARGIN_REQUIREMENTS[ticker])
            ccy = ASSET_CURRENCIES[ticker]

            factor_names.append(factor_name)
            factor_types[factor_name] = "asset"
            factor_currencies[factor_name] = ccy
            factor_tickers[factor_name] = ticker
            factor_returns.append(ret.rename(factor_name))

            legacy_assets[ticker] = {
                "factor_name": factor_name,
                "ticker": ticker,
                "query_ticker": query_ticker,
                "currency": ccy,
                "v0": value_chf,
                "m": margin_rate,
                "maintenance_margin_chf": value_chf * margin_rate,
                "mu_raw": mu_raw,
                "mu": mu,
                "sigma": sigma,
            }

        fx_factors = {}

        for ccy in sorted(required_fx_ccys):
            fx_col = self._fx_column_for_currency(ccy)
            if fx_col not in self.raw_fx_data.columns:
                raise ValueError(f"[!] SYSTEM HALT: Missing FX column during parameter estimation: {fx_col}")

            factor_name = f"fx:{ccy}"
            fx_ret = self._log_returns_from_prices(
                self.raw_fx_data[fx_col].rename(factor_name)
            ).dropna()

            if fx_ret.empty:
                raise ValueError(f"[!] SYSTEM HALT: Not enough FX return data for {ccy}/{BASE_CURRENCY}.")

            sigma = float(fx_ret.std() * np.sqrt(252))
            mu_raw = float((fx_ret.mean() * 252) + (sigma**2 / 2))
            mu = 0.0

            factor_names.append(factor_name)
            factor_types[factor_name] = "fx"
            factor_currencies[factor_name] = ccy
            factor_tickers[factor_name] = fx_col
            factor_returns.append(fx_ret.rename(factor_name))

            fx_factors[ccy] = {
                "factor_name": factor_name,
                "currency": ccy,
                "fx_ticker": fx_col,
                "mu_raw": mu_raw,
                "mu": mu,
                "sigma": sigma,
            }

        aligned_df = pd.concat(factor_returns, axis=1).dropna(how="any")

        if aligned_df.empty:
            raise ValueError(
                "[!] SYSTEM HALT: No overlapping return history across target, legacy assets, and FX factors."
            )

        if len(aligned_df) < max(30, len(factor_names) * 3):
            raise ValueError(
                "[!] SYSTEM HALT: Too few aligned observations for a stable covariance estimate.\n"
                f"    Observations: {len(aligned_df)}\n"
                f"    Factors: {len(factor_names)}\n"
                f"    Factor order: {factor_names}"
            )

        corr_matrix = self._sanitize_corr_matrix(aligned_df.corr().values)
        cov_matrix_annual = (aligned_df.cov().values * 252.0).astype(float)

        mu_by_factor = {target_factor["name"]: target_factor["mu"]}
        mu_raw_by_factor = {target_factor["name"]: target_factor["mu_raw"]}
        sigma_by_factor = {target_factor["name"]: target_factor["sigma"]}

        for info in legacy_assets.values():
            mu_by_factor[info["factor_name"]] = info["mu"]
            mu_raw_by_factor[info["factor_name"]] = info["mu_raw"]
            sigma_by_factor[info["factor_name"]] = info["sigma"]

        for info in fx_factors.values():
            mu_by_factor[info["factor_name"]] = info["mu"]
            mu_raw_by_factor[info["factor_name"]] = info["mu_raw"]
            sigma_by_factor[info["factor_name"]] = info["sigma"]

        return {
            # Backward-friendly target aliases.
            "mu_target_raw": mu_target_raw,
            "mu_target": mu_target,
            "sigma_target": sigma_target,

            # Dynamic model metadata.
            "target_factor": target_factor,
            "legacy_assets": legacy_assets,
            "legacy_asset_order": active_legacy,
            "fx_factors": fx_factors,
            "factor_names": factor_names,
            "factor_types": factor_types,
            "factor_currencies": factor_currencies,
            "factor_tickers": factor_tickers,
            "mu_by_factor": mu_by_factor,
            "mu_raw_by_factor": mu_raw_by_factor,
            "sigma_by_factor": sigma_by_factor,
            "corr_matrix": corr_matrix,
            "cov_matrix_annual": cov_matrix_annual,
            "aligned_observations": int(len(aligned_df)),
            "base_currency": BASE_CURRENCY,
        }
