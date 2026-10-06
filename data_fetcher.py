import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import track

from config import (
    ASSET_CURRENCIES, BASE_CURRENCY, CURRENT_DATE, HISTORICAL_LOOKBACK_YEARS,
    HOLDINGS, IM_REQUIREMENTS, MM_REQUIREMENTS, BOND_REGISTRY, ACTIVE_ASSET
)

# highlight=False disables arbitrary numeric coloration for professional consistency
console = Console(highlight=False)

class DataEngine:
    def __init__(self):
        self.tickers = list(HOLDINGS.keys())
        self.end_date = CURRENT_DATE + datetime.timedelta(days=1)
        self.start_date = CURRENT_DATE - datetime.timedelta(days=HISTORICAL_LOOKBACK_YEARS * 365)
        self.data = pd.DataFrame()
        self.raw_local_data = pd.DataFrame()
        self.raw_fx_data = pd.DataFrame()
        self.current_prices_base = {}
        self.fx_tickers = {}
        self.asset_meta = {}

    @staticmethod
    def _extract_price_matrix(raw_data: pd.DataFrame, yf_query_list: list) -> pd.DataFrame:
        if raw_data is None or raw_data.empty:
            raise ValueError("[!] SYSTEM HALT: yfinance returned no data.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data.xs("Close", level=0, axis=1).copy()
        else:
            price_obj = raw_data["Close"]
            df = price_obj.rename(yf_query_list[0]).to_frame() if isinstance(price_obj, pd.Series) else price_obj.copy()

        df.columns = [str(c) for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()].copy()

    def calibrate_dividend_profile(self, ticker: str) -> dict:
        if ticker in BOND_REGISTRY: 
            return {"has_dividends": False, "ttm_yield": 0.0, "mean_yield": 0.0, "yield_vol": 0.0, "payout_months": []}
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            prices = t.history(period="10y")["Close"]
            if divs.empty or prices.empty: raise ValueError("No divs")
            
            annual_divs = divs.resample("YE").sum()
            annual_prices = prices.resample("YE").last()
            aligned = pd.DataFrame({"Div": annual_divs, "Price": annual_prices}).dropna()
            if aligned.empty: raise ValueError("No aligned data")
            
            yields = (aligned["Div"] / aligned["Price"]).clip(lower=0.0)
            ttm_cash = float(divs.last("365D").sum())
            curr_price = float(prices.iloc[-1])
            
            return {
                "has_dividends": True,
                "ttm_yield": ttm_cash / curr_price if curr_price > 0 else float(yields.mean()),
                "mean_yield": float(yields.mean()),
                "yield_vol": float(yields.std()),
                "payout_months": sorted(list(set(divs.last("3Y").index.month)))
            }
        except Exception:
            return {"has_dividends": False, "ttm_yield": 0.0, "mean_yield": 0.0, "yield_vol": 0.0, "payout_months": []}

    def _fx_column_for_currency(self, ccy: str) -> str:
        return f"GBP{BASE_CURRENCY}=X" if ccy == "GBX" else f"{ccy}{BASE_CURRENCY}=X"

    def _query_ticker_for_asset(self, ticker: str) -> str:
        return BOND_REGISTRY[ticker]["proxy_ticker"] if ticker in BOND_REGISTRY else ticker

    def fetch_data(self) -> None:
        foreign_ccys = {ASSET_CURRENCIES[t] for t in self.tickers if ASSET_CURRENCIES[t] != BASE_CURRENCY}
        self.fx_tickers = {ccy: self._fx_column_for_currency(ccy) for ccy in sorted(foreign_ccys)}

        yf_query_list = list(self.fx_tickers.values())
        for ticker in self.tickers: yf_query_list.append(self._query_ticker_for_asset(ticker))
        yf_query_list = list(dict.fromkeys(yf_query_list))

        with console.status("Downloading unadjusted historical prices...", spinner="line"):
            raw_data = yf.download(yf_query_list[0] if len(yf_query_list) == 1 else yf_query_list,
                                   start=self.start_date, end=self.end_date, auto_adjust=False, progress=False)
            df = self._extract_price_matrix(raw_data, yf_query_list)

        fx_columns = list(dict.fromkeys(self.fx_tickers.values()))
        for fx in fx_columns: df[fx] = df[fx].ffill()

        self.raw_local_data = df.copy()
        self.raw_fx_data = df[fx_columns].copy() if fx_columns else pd.DataFrame(index=df.index)

        for ticker in track(self.tickers, description="Extracting yield & dividend profiles...", console=console):
            ccy = ASSET_CURRENCIES[ticker]
            fx_ticker = self.fx_tickers.get(ccy)
            query_target = self._query_ticker_for_asset(ticker)

            clean_series = df[query_target].dropna()
            local_spot = float(clean_series.iloc[-1])
            live_fx_rate = 1.0

            if ccy != BASE_CURRENCY:
                raw_fx = float(df[fx_ticker].dropna().iloc[-1])
                live_fx_rate = raw_fx / 100.0 if ccy == "GBX" else raw_fx
            
            self.current_prices_base[ticker] = local_spot * live_fx_rate
            self.data[ticker] = df[query_target] * (df[fx_ticker] / 100.0 if ccy == "GBX" else df[fx_ticker]) if ccy != BASE_CURRENCY else df[query_target]
            
            div_profile = self.calibrate_dividend_profile(ticker)
            div_profile["fx_rate"] = live_fx_rate
            self.asset_meta[ticker] = div_profile

        self.data.dropna(how="all", inplace=True)
        console.print("Market data ingestion complete.\n")

    def build_current_state(self) -> dict:
        portfolio_order = [ACTIVE_ASSET]
        for ticker in self.tickers:
            if ticker != ACTIVE_ASSET and (HOLDINGS[ticker] * self.current_prices_base[ticker]) > 0:
                portfolio_order.append(ticker)

        assets_dict = {}
        for ticker in portfolio_order:
            quantity = float(HOLDINGS[ticker])
            value_base = float(quantity * self.current_prices_base[ticker])
            meta = self.asset_meta[ticker]
            
            if ticker in BOND_REGISTRY:
                live_price = BOND_REGISTRY[ticker]["current_price_local"]
                value_base = quantity * live_price * meta["fx_rate"]
                meta["current_price_local"] = live_price  # Pass to engine
                
            assets_dict[ticker] = {
                "ticker": ticker, 
                "quantity": quantity,
                "v0": value_base, 
                "mmr": float(MM_REQUIREMENTS[ticker]), 
                "imr": float(IM_REQUIREMENTS[ticker]),
                **meta
            }
        return {"portfolio_order": portfolio_order, "assets_dict": assets_dict, "active_asset": ACTIVE_ASSET}