import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import track

from config import (
    ASSET_CURRENCIES, BASE_CURRENCY, CURRENT_DATE, HISTORICAL_LOOKBACK_YEARS,
    HOLDINGS, IM_REQUIREMENTS, MM_REQUIREMENTS, BOND_REGISTRY, ACTIVE_ASSET, ASSET_PROXIES
)

console = Console(highlight=False)


class DataEngine:
    def __init__(self):
        self._validate_configuration()
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=int(HISTORICAL_LOOKBACK_YEARS * 365.25))
        self.data = pd.DataFrame()
        self.raw_local_data = pd.DataFrame()
        self.raw_fx_data = pd.DataFrame()
        self.current_prices_base = {}
        self.fx_tickers = {}
        self.asset_meta = {}

    @staticmethod
    def _validate_configuration() -> None:
        if ACTIVE_ASSET not in HOLDINGS:
            raise ValueError(f"[CONFIG ERROR] ACTIVE_ASSET '{ACTIVE_ASSET}' must be a key in HOLDINGS.")
        if ACTIVE_ASSET in BOND_REGISTRY:
            raise ValueError(f"[CONFIG ERROR] ACTIVE_ASSET '{ACTIVE_ASSET}' cannot be an individual bond.")
        for t in HOLDINGS:
            if t not in ASSET_CURRENCIES:
                raise ValueError(f"[CONFIG ERROR] Missing currency mapping in ASSET_CURRENCIES for '{t}'.")
            if t not in MM_REQUIREMENTS or t not in IM_REQUIREMENTS:
                raise ValueError(f"[CONFIG ERROR] Missing margin requirement in MM/IM_REQUIREMENTS for '{t}'.")
            if IM_REQUIREMENTS[t] < MM_REQUIREMENTS[t] or MM_REQUIREMENTS[t] <= 0.0:
                raise ValueError(f"[CONFIG ERROR] Invalid IMR/MMR hierarchy for '{t}'.")

    @staticmethod
    def _extract_price_matrix(raw_data: pd.DataFrame, yf_query_list: list) -> pd.DataFrame:
        if raw_data is None or raw_data.empty:
            raise RuntimeError("[!] SYSTEM HALT: yfinance returned an empty dataset.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            if "Close" not in raw_data.columns.get_level_values(0):
                raise RuntimeError("[!] SYSTEM HALT: 'Close' level missing from yfinance response.")
            df = raw_data.xs("Close", level=0, axis=1).copy()
        else:
            if "Close" not in raw_data.columns:
                raise RuntimeError("[!] SYSTEM HALT: 'Close' column missing from yfinance response.")
            price_obj = raw_data["Close"]
            df = price_obj.rename(yf_query_list[0]).to_frame() if isinstance(price_obj, pd.Series) else price_obj.copy()

        df.columns = [str(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        missing_cols = [q for q in yf_query_list if q not in df.columns or df[q].dropna().empty]
        if missing_cols:
            raise RuntimeError(f"[!] SYSTEM HALT: Failed to download valid price series for: {missing_cols}")

        return df.sort_index()

    def calibrate_dividend_profile(self, ticker: str) -> dict:
        default_profile = {
            "has_dividends": False,
            "ttm_yield": 0.0,
            "mean_yield": 0.0,
            "yield_vol": 0.0,
            "payout_months": []
        }
        if ticker in BOND_REGISTRY:
            return default_profile

        try:
            query_target = self._query_ticker_for_asset(ticker)
            if query_target not in self.raw_local_data.columns:
                return default_profile

            prices = self.raw_local_data[query_target].dropna()
            if prices.empty:
                return default_profile

            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs is None or divs.empty:
                return default_profile

            divs_idx = divs.index.tz_localize(None) if divs.index.tz is not None else divs.index
            divs = pd.Series(divs.values, index=divs_idx, dtype=float).sort_index()

            cutoff_10y = pd.Timestamp(CURRENT_DATE) - pd.DateOffset(years=10)
            divs_10y = divs[divs.index >= cutoff_10y]
            prices_10y = prices[prices.index >= cutoff_10y]

            if divs_10y.empty or prices_10y.empty:
                return default_profile

            annual_divs = divs_10y.resample("YE").sum()
            annual_prices = prices_10y.resample("YE").last()
            aligned = pd.DataFrame({"Div": annual_divs, "Price": annual_prices}).dropna()
            aligned = aligned[aligned["Price"] > 0.0]
            if aligned.empty:
                return default_profile

            yields = (aligned["Div"] / aligned["Price"]).clip(lower=0.0, upper=0.25)

            cutoff_1y = pd.Timestamp(CURRENT_DATE) - pd.Timedelta(days=365)
            cutoff_3y = pd.Timestamp(CURRENT_DATE) - pd.DateOffset(years=3)
            ttm_cash = float(divs[divs.index >= cutoff_1y].sum())
            curr_price = float(prices.iloc[-1])

            ttm_yield = float(np.clip(ttm_cash / curr_price, 0.0, 0.25)) if curr_price > 0.0 else float(yields.mean())
            payout_months = sorted(list(set(divs[divs.index >= cutoff_3y].index.month)))
            if not payout_months and ttm_cash > 0.0:
                payout_months = sorted(list(set(divs_10y.index.month)))

            return {
                "has_dividends": len(payout_months) > 0 and (ttm_yield > 0.0 or float(yields.mean()) > 0.0),
                "ttm_yield": ttm_yield,
                "mean_yield": float(yields.mean()),
                "yield_vol": float(yields.std(ddof=1)) if len(yields) > 1 else 0.0,
                "payout_months": payout_months
            }
        except Exception as exc:
            console.print(f"[yellow]Warning: Dividend calibration fallback for {ticker} ({exc})[/yellow]")
            return default_profile

    def _fx_column_for_currency(self, ccy: str) -> str:
        return f"GBP{BASE_CURRENCY}=X" if ccy == "GBX" else f"{ccy}{BASE_CURRENCY}=X"

    def _query_ticker_for_asset(self, ticker: str) -> str:
        return BOND_REGISTRY[ticker]["proxy_ticker"] if ticker in BOND_REGISTRY else ticker

    def fetch_data(self) -> None:
        portfolio_foreign_ccys = {ASSET_CURRENCIES[t] for t in self.tickers if ASSET_CURRENCIES[t] != BASE_CURRENCY}
        all_foreign_ccys = set(portfolio_foreign_ccys)
        for _, (_, proxy_ccy) in ASSET_PROXIES.items():
            if proxy_ccy != BASE_CURRENCY:
                all_foreign_ccys.add(proxy_ccy)

        all_fx_tickers = {ccy: self._fx_column_for_currency(ccy) for ccy in sorted(all_foreign_ccys)}
        # Keep self.fx_tickers scoped strictly to portfolio currencies for FX trackers
        self.fx_tickers = {ccy: all_fx_tickers[ccy] for ccy in sorted(portfolio_foreign_ccys)}

        yf_query_list = list(all_fx_tickers.values())
        for ticker in self.tickers:
            yf_query_list.append(self._query_ticker_for_asset(ticker))
        for _, (proxy_ticker, _) in ASSET_PROXIES.items():
            yf_query_list.append(proxy_ticker)
        yf_query_list = list(dict.fromkeys(yf_query_list))

        with console.status("Downloading unadjusted historical prices...", spinner="line"):
            raw_data = yf.download(
                yf_query_list[0] if len(yf_query_list) == 1 else yf_query_list,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )
            df = self._extract_price_matrix(raw_data, yf_query_list)

        # Remove weekend rows before forward-filling to prevent artificial zero-return dilution
        df = df[df.index.dayofweek < 5].copy()

        all_fx_cols = list(dict.fromkeys(all_fx_tickers.values()))
        for fx in all_fx_cols:
            df[fx] = df[fx].ffill().bfill()

        self.raw_local_data = df.copy()
        portfolio_fx_cols = list(dict.fromkeys(self.fx_tickers.values()))
        self.raw_fx_data = df[portfolio_fx_cols].copy() if portfolio_fx_cols else pd.DataFrame(index=df.index)

        def _fx_series_for(ccy: str):
            if ccy == BASE_CURRENCY:
                return 1.0
            col = all_fx_tickers[ccy]
            return df[col] / 100.0 if ccy == "GBX" else df[col]

        # 1. ADD FX SERIES AS INDEPENDENT TRACKERS FOR PCA CALIBRATION
        for ccy in sorted(self.fx_tickers.keys()):
            self.data[f"FX_{ccy}"] = _fx_series_for(ccy)

        # 2. ADD ASSET SERIES, WALK BACK USING PROXIES & EXTRACT DIVIDEND PROFILES
        for ticker in track(self.tickers, description="Extracting yield & dividend profiles...", console=console):
            ccy = ASSET_CURRENCIES[ticker]
            query_target = self._query_ticker_for_asset(ticker)

            clean_series = df[query_target].dropna()
            local_spot = float(clean_series.iloc[-1])
            live_fx_rate = 1.0 if ccy == BASE_CURRENCY else float(_fx_series_for(ccy).iloc[-1])

            self.current_prices_base[ticker] = local_spot * live_fx_rate

            # Forward-fill real asset data from its first valid trading day onward
            if ticker in BOND_REGISTRY:
                asset_series = df[query_target].ffill()
            else:
                asset_series = (df[query_target] * _fx_series_for(ccy)).ffill()

            # Walk backward day-by-day using forward-filled proxy returns prior to first_valid_dt
            first_valid_dt = asset_series.first_valid_index()
            if ticker in ASSET_PROXIES and first_valid_dt is not None and first_valid_dt > df.index[0]:
                proxy_ticker, proxy_ccy = ASSET_PROXIES[ticker]
                if ticker in BOND_REGISTRY:
                    proxy_series = df[proxy_ticker].ffill().bfill()
                else:
                    proxy_series = (df[proxy_ticker] * _fx_series_for(proxy_ccy)).ffill().bfill()

                anchor_ratio = float(asset_series.loc[first_valid_dt]) / float(proxy_series.loc[first_valid_dt])
                backfill_slice = proxy_series.loc[:first_valid_dt].iloc[:-1] * anchor_ratio
                asset_series.loc[backfill_slice.index] = backfill_slice

            self.data[ticker] = asset_series

            div_profile = self.calibrate_dividend_profile(ticker)
            div_profile["fx_rate"] = live_fx_rate
            div_profile["currency"] = ccy
            div_profile["is_fx_tracker"] = False
            self.asset_meta[ticker] = div_profile

        self.data.dropna(how="all", inplace=True)
        console.print("Market data ingestion complete.\n")

    def build_current_state(self) -> dict:
        portfolio_order = [ACTIVE_ASSET]
        for ticker in self.tickers:
            if ticker != ACTIVE_ASSET and (HOLDINGS.get(ticker, 0.0) * self.current_prices_base[ticker]) > 0.0:
                portfolio_order.append(ticker)

        # Append FX trackers without mutating global config dicts or inflating v0
        for ccy in sorted(self.fx_tickers.keys()):
            fx_name = f"FX_{ccy}"
            portfolio_order.append(fx_name)

            raw_fx = float(self.raw_fx_data[self.fx_tickers[ccy]].dropna().iloc[-1])
            live_fx_rate = raw_fx / 100.0 if ccy == "GBX" else raw_fx

            self.asset_meta[fx_name] = {
                "has_dividends": False,
                "ttm_yield": 0.0,
                "mean_yield": 0.0,
                "yield_vol": 0.0,
                "payout_months": [],
                "fx_rate": live_fx_rate,
                "currency": ccy,
                "is_fx_tracker": True
            }
            self.current_prices_base[fx_name] = live_fx_rate

        assets_dict = {}
        for ticker in portfolio_order:
            meta = self.asset_meta[ticker].copy()
            is_fx = meta.get("is_fx_tracker", False)
            quantity = 0.0 if is_fx else float(HOLDINGS.get(ticker, 0.0))
            value_base = float(quantity * self.current_prices_base[ticker])

            if ticker in BOND_REGISTRY:
                live_price = float(BOND_REGISTRY[ticker]["current_price_local"])
                value_base = quantity * live_price * meta["fx_rate"]
                meta["current_price_local"] = live_price

            assets_dict[ticker] = {
                "ticker": ticker,
                "quantity": quantity,
                "v0": value_base,
                "mmr": 0.0 if is_fx else float(MM_REQUIREMENTS.get(ticker, 0.25)),
                "imr": 0.0 if is_fx else float(IM_REQUIREMENTS.get(ticker, 0.2875)),
                **meta
            }

        return {
            "portfolio_order": portfolio_order,
            "assets_dict": assets_dict,
            "active_asset": ACTIVE_ASSET
        }