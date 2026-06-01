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
    """Handles data ingestion, FX conversion, and dynamic factor estimation."""

    MIN_FACTOR_OBSERVATIONS = 252

    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)
        self.data = pd.DataFrame()
        self.current_prices_chf = {}
        self.raw_local_data = pd.DataFrame()
        self.raw_fx_data = pd.DataFrame()
        self.fx_tickers_by_currency = {}

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

    @staticmethod
    def _market_data_ticker(ticker: str) -> str:
        """Returns the Yahoo ticker used for historical data for listed and OTC assets."""
        return OTC_REGISTRY[ticker]["proxy_ticker"] if ticker in OTC_REGISTRY else ticker

    @staticmethod
    def _fx_yahoo_ticker(ccy: str) -> str | None:
        """Returns the Yahoo FX ticker for one unit of ccy expressed in BASE_CURRENCY."""
        if ccy == BASE_CURRENCY:
            return None
        if ccy == "GBX":
            # Yahoo quotes GBPCHF; GBX values are converted with GBPCHF / 100 for levels.
            return f"GBP{BASE_CURRENCY}=X"
        return f"{ccy}{BASE_CURRENCY}=X"

    @staticmethod
    def _clean_simple_returns(price_series: pd.Series, name: str) -> pd.Series:
        """Computes finite simple returns, excluding invalid <= -100% observations."""
        simple = price_series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        simple = simple.dropna()
        simple = simple[simple > -1.0]
        simple.name = name
        return simple

    @staticmethod
    def _annualized_mu_sigma(log_returns: pd.Series) -> tuple[float, float]:
        sigma = float(log_returns.std() * np.sqrt(252))
        mu_raw = float((log_returns.mean() * 252) + (sigma ** 2 / 2))
        return mu_raw, sigma

    def fetch_data(self) -> None:
        """Downloads historical data, spot prices, and applies vectorized FX conversion to CHF."""

        # 1. Validate configuration and identify required FX pairs.
        missing_currency = [t for t in self.tickers if t not in ASSET_CURRENCIES]
        if missing_currency:
            raise ValueError(f"[!] Missing ASSET_CURRENCIES entries: {missing_currency}")

        missing_margin = [t for t in self.tickers if t not in MARGIN_REQUIREMENTS]
        if missing_margin:
            raise ValueError(f"[!] Missing MARGIN_REQUIREMENTS entries: {missing_margin}")

        if TARGET_ASSET not in HOLDINGS:
            raise ValueError(f"[!] TARGET_ASSET {TARGET_ASSET} is not present in HOLDINGS.")

        foreign_ccys = {
            ASSET_CURRENCIES[t]
            for t in self.tickers
            if ASSET_CURRENCIES[t] != BASE_CURRENCY
        }

        fx_tickers = {
            ccy: self._fx_yahoo_ticker(ccy)
            for ccy in sorted(foreign_ccys)
        }
        fx_tickers = {ccy: fx for ccy, fx in fx_tickers.items() if fx is not None}
        self.fx_tickers_by_currency = fx_tickers.copy()

        # 2. Build the unified yfinance query list.
        yf_query_list = list(fx_tickers.values())

        for ticker in self.tickers:
            yf_query_list.append(self._market_data_ticker(ticker))

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

        # 4. Process live prices and historical data.
        for ticker in tqdm(
            self.tickers,
            desc="Processing Asset Data",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="blue"
        ):
            ccy = ASSET_CURRENCIES[ticker]
            fx_ticker = fx_tickers.get(ccy)

            query_target = self._market_data_ticker(ticker)
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
                self.data[ticker] = clean_series
            else:
                local_spot = float(clean_series.iloc[-1])
                self.data[ticker] = clean_series

            if ccy == BASE_CURRENCY:
                self.current_prices_chf[ticker] = local_spot
            else:
                if fx_ticker not in df.columns or df[fx_ticker].dropna().empty:
                    raise ValueError(f"\n[!] SYSTEM HALT: FX Rate '{fx_ticker}' failed to download.")

                live_fx_rate = float(df[fx_ticker].dropna().iloc[-1])

                if ccy == "GBX":
                    self.current_prices_chf[ticker] = local_spot * (live_fx_rate / 100.0)
                else:
                    self.current_prices_chf[ticker] = local_spot * live_fx_rate

            # Keep self.data as CHF-denominated price history for diagnostics/backward compatibility.
            if ccy != BASE_CURRENCY:
                if ccy == "GBX":
                    self.data[ticker] = self.data[ticker] * (df[fx_ticker] / 100.0)
                else:
                    self.data[ticker] = self.data[ticker] * df[fx_ticker]

        self.data.dropna(how="all", inplace=True)

    def build_current_state(self) -> dict:
        """Calculates today's explicit initial values using CHF spot prices and legacy FX buckets."""
        target_ccy = ASSET_CURRENCIES[TARGET_ASSET]
        v_target_0 = float(HOLDINGS[TARGET_ASSET] * self.current_prices_chf[TARGET_ASSET])

        legacy_by_currency = {}
        v_legacy_0 = 0.0
        leg_mm_total = 0.0

        for ticker in self.tickers:
            if ticker == TARGET_ASSET:
                continue

            units = float(HOLDINGS[ticker])
            value_chf = units * float(self.current_prices_chf[ticker])

            # Skip zero positions in the simulation state, but still allow their data to be configured.
            if value_chf <= 0.0:
                continue

            margin_chf = value_chf * float(MARGIN_REQUIREMENTS[ticker])
            ccy = ASSET_CURRENCIES[ticker]

            v_legacy_0 += value_chf
            leg_mm_total += margin_chf

            if ccy not in legacy_by_currency:
                legacy_by_currency[ccy] = {
                    "v0": 0.0,
                    "maintenance_margin_chf": 0.0,
                    "tickers": [],
                    "weights_chf": {}
                }

            legacy_by_currency[ccy]["v0"] += value_chf
            legacy_by_currency[ccy]["maintenance_margin_chf"] += margin_chf
            legacy_by_currency[ccy]["tickers"].append(ticker)
            legacy_by_currency[ccy]["weights_chf"][ticker] = value_chf

        for bucket in legacy_by_currency.values():
            bucket["m"] = (
                bucket["maintenance_margin_chf"] / bucket["v0"]
                if bucket["v0"] > 0
                else 0.0
            )
            bucket["weights_chf"] = {
                ticker: value / bucket["v0"]
                for ticker, value in bucket["weights_chf"].items()
            }

        m_legacy = leg_mm_total / v_legacy_0 if v_legacy_0 > 0 else 0.0

        return {
            "v_target_0": v_target_0,
            "v_legacy_0": float(v_legacy_0),
            "m_target": float(MARGIN_REQUIREMENTS[TARGET_ASSET]),
            "m_legacy": float(m_legacy),
            "target_currency": target_ccy,
            "legacy_by_currency": dict(sorted(legacy_by_currency.items()))
        }

    def _bucket_local_log_returns(self, ccy: str, tickers: list[str], values_chf: np.ndarray) -> pd.Series:
        """Builds one local-currency legacy-bucket return series from its constituent assets."""
        if len(tickers) == 0:
            raise ValueError(f"[!] Empty legacy bucket for currency {ccy}.")

        weights = values_chf / values_chf.sum()
        simple_returns = []

        for ticker in tickers:
            query_ticker = self._market_data_ticker(ticker)
            if query_ticker not in self.raw_local_data.columns:
                raise ValueError(
                    f"[!] Missing local history for {ticker} via Yahoo ticker {query_ticker}."
                )

            simple_returns.append(
                self._clean_simple_returns(self.raw_local_data[query_ticker], ticker)
            )

        local_simple = pd.concat(simple_returns, axis=1).dropna(how="any")

        if local_simple.empty:
            raise ValueError(
                f"[!] SYSTEM HALT: No overlapping local return history for legacy {ccy} bucket."
            )

        bucket_simple = (local_simple * weights).sum(axis=1)
        bucket_simple = bucket_simple.replace([np.inf, -np.inf], np.nan).dropna()
        bucket_simple = bucket_simple[bucket_simple > -1.0]

        if len(bucket_simple) < self.MIN_FACTOR_OBSERVATIONS:
            raise ValueError(
                f"[!] SYSTEM HALT: Legacy {ccy} bucket has only {len(bucket_simple)} observations; "
                f"need at least {self.MIN_FACTOR_OBSERVATIONS}."
            )

        return pd.Series(np.log1p(bucket_simple), index=bucket_simple.index, name=f"legacy:{ccy}")

    def _fx_log_returns(self, ccy: str) -> pd.Series:
        """Builds a zero-drift FX factor return history for ccy/BASE_CURRENCY."""
        fx_col = self._fx_yahoo_ticker(ccy)
        if fx_col is None:
            raise ValueError("Base currency has no stochastic FX factor.")

        if fx_col not in self.raw_fx_data.columns:
            raise ValueError(f"[!] SYSTEM HALT: Missing FX column during parameter estimation: {fx_col}")

        fx_simple = self._clean_simple_returns(self.raw_fx_data[fx_col], f"fx:{ccy}")

        if len(fx_simple) < self.MIN_FACTOR_OBSERVATIONS:
            raise ValueError(
                f"[!] SYSTEM HALT: FX {ccy}/{BASE_CURRENCY} has only {len(fx_simple)} observations; "
                f"need at least {self.MIN_FACTOR_OBSERVATIONS}."
            )

        return pd.Series(np.log1p(fx_simple), index=fx_simple.index, name=f"fx:{ccy}")

    def estimate_parameters(self) -> dict:
        """
        Computes a dynamic factor model:
        target local factor + one local legacy factor per currency bucket + one FX factor per foreign currency.
        """

        # --- 1. Target local asset factor, deliberately not CHF-converted to avoid FX double counting. ---
        target_ccy = ASSET_CURRENCIES[TARGET_ASSET]
        target_query_ticker = self._market_data_ticker(TARGET_ASSET)

        if target_query_ticker not in self.raw_local_data.columns:
            raise ValueError(f"[!] Missing local history for target asset {TARGET_ASSET}.")

        target_simple = self._clean_simple_returns(self.raw_local_data[target_query_ticker], "target")
        if len(target_simple) < self.MIN_FACTOR_OBSERVATIONS:
            raise ValueError(
                f"[!] SYSTEM HALT: Target asset has only {len(target_simple)} observations; "
                f"need at least {self.MIN_FACTOR_OBSERVATIONS}."
            )

        target_ret = pd.Series(np.log1p(target_simple), index=target_simple.index, name="target")
        mu_target_raw, sigma_target = self._annualized_mu_sigma(target_ret)
        mu_target = self._cap_drift(mu_target_raw, TARGET_DRIFT_CAP)

        factor_series = [target_ret]
        factor_names = ["target"]
        factor_types = {"target": "target_local"}
        factor_currencies = {"target": target_ccy}
        mu_raw_by_factor = {"target": mu_target_raw}
        mu_by_factor = {"target": mu_target}
        sigma_by_factor = {"target": sigma_target}

        target_factor = {
            "factor_name": "target",
            "asset": TARGET_ASSET,
            "currency": target_ccy,
            "mu_raw": mu_target_raw,
            "mu": mu_target,
            "sigma": sigma_target
        }

        # --- 2. Legacy local asset factors, one bucket per currency. ---
        active_legacy_by_currency: dict[str, list[tuple[str, float]]] = {}

        for ticker in self.tickers:
            if ticker == TARGET_ASSET:
                continue

            value_chf = float(HOLDINGS[ticker]) * float(self.current_prices_chf[ticker])
            if value_chf <= 0.0:
                continue

            ccy = ASSET_CURRENCIES[ticker]
            active_legacy_by_currency.setdefault(ccy, []).append((ticker, value_chf))

        legacy_buckets = {}
        legacy_bucket_factors = {}

        for ccy in sorted(active_legacy_by_currency):
            members = active_legacy_by_currency[ccy]
            tickers = [ticker for ticker, _ in members]
            values_chf = np.array([value for _, value in members], dtype=float)

            bucket_ret = self._bucket_local_log_returns(ccy, tickers, values_chf)
            factor_name = f"legacy:{ccy}"
            bucket_ret.name = factor_name

            mu_raw, sigma = self._annualized_mu_sigma(bucket_ret)
            mu = self._cap_drift(mu_raw, LEGACY_DRIFT_CAP)
            weights = values_chf / values_chf.sum()

            factor_series.append(bucket_ret)
            factor_names.append(factor_name)
            factor_types[factor_name] = "legacy_local"
            factor_currencies[factor_name] = ccy
            mu_raw_by_factor[factor_name] = mu_raw
            mu_by_factor[factor_name] = mu
            sigma_by_factor[factor_name] = sigma
            legacy_bucket_factors[ccy] = factor_name
            legacy_buckets[ccy] = {
                "factor_name": factor_name,
                "currency": ccy,
                "tickers": tickers,
                "weights_chf": {ticker: float(weight) for ticker, weight in zip(tickers, weights)},
                "mu_raw": mu_raw,
                "mu": mu,
                "sigma": sigma
            }

        # --- 3. FX factors, one per foreign currency actually exposed by target or legacy buckets. ---
        exposed_currencies = {target_ccy} | set(active_legacy_by_currency.keys())
        fx_currencies = sorted(ccy for ccy in exposed_currencies if ccy != BASE_CURRENCY)

        fx_factors = {}
        fx_factor_names_by_currency = {}

        for ccy in fx_currencies:
            fx_ret = self._fx_log_returns(ccy)
            factor_name = f"fx:{ccy}"
            fx_ret.name = factor_name

            mu_raw, sigma = self._annualized_mu_sigma(fx_ret)
            mu = 0.0  # Intentional: no expected indefinite appreciation/depreciation of any currency.

            factor_series.append(fx_ret)
            factor_names.append(factor_name)
            factor_types[factor_name] = "fx"
            factor_currencies[factor_name] = ccy
            mu_raw_by_factor[factor_name] = mu_raw
            mu_by_factor[factor_name] = mu
            sigma_by_factor[factor_name] = sigma
            fx_factor_names_by_currency[ccy] = factor_name
            fx_factors[ccy] = {
                "factor_name": factor_name,
                "currency": ccy,
                "pair": f"{ccy}/{BASE_CURRENCY}",
                "yahoo_ticker": self._fx_yahoo_ticker(ccy),
                "mu_raw": mu_raw,
                "mu": mu,
                "sigma": sigma
            }

        # --- 4. Dynamic covariance/correlation matrix across every target, legacy and FX factor. ---
        aligned_df = pd.concat(factor_series, axis=1).dropna(how="any")

        if len(aligned_df) < self.MIN_FACTOR_OBSERVATIONS:
            raise ValueError(
                f"[!] SYSTEM HALT: Only {len(aligned_df)} fully aligned factor observations available; "
                f"need at least {self.MIN_FACTOR_OBSERVATIONS}.\n"
                f"    Factors: {factor_names}"
            )

        corr_matrix = self._sanitize_corr_matrix(aligned_df.corr().loc[factor_names, factor_names].values)
        cov_matrix_annual = aligned_df.loc[:, factor_names].cov().values * 252.0

        mu_vector = np.array([mu_by_factor[name] for name in factor_names], dtype=float)
        sigma_vector = np.array([sigma_by_factor[name] for name in factor_names], dtype=float)

        return {
            "target_factor": target_factor,
            "target_factor_name": "target",
            "target_currency": target_ccy,

            "legacy_buckets": legacy_buckets,
            "legacy_bucket_factors": legacy_bucket_factors,
            "legacy_bucket_currencies": sorted(legacy_buckets.keys()),

            "fx_factors": fx_factors,
            "fx_factor_names_by_currency": fx_factor_names_by_currency,
            "fx_currencies": fx_currencies,

            "factor_names": factor_names,
            "factor_types": factor_types,
            "factor_currencies": factor_currencies,
            "mu_raw_by_factor": mu_raw_by_factor,
            "mu_by_factor": mu_by_factor,
            "sigma_by_factor": sigma_by_factor,
            "mu_vector": mu_vector,
            "sigma_vector": sigma_vector,
            "corr_matrix": corr_matrix,
            "cov_matrix_annual": cov_matrix_annual,
            "factor_return_observations": int(len(aligned_df)),

            # Backward-compatible target summary fields used by reporting code.
            "mu_target_raw": mu_target_raw,
            "mu_target": mu_target,
            "sigma_target": sigma_target
        }
