import datetime

import numpy as np
import pandas as pd

from config import (
    CURRENT_DATE,
    CURRENT_DEBT,
    DEFAULT_MONTHLY_DEPOSIT_2026,
    DEFAULT_MONTHLY_DEPOSIT_FUTURE,
    HESTON_KAPPA,
    HESTON_RHO,
    HESTON_XI,
    JUMP_FREQUENCY_PER_YEAR,
    JUMP_MEAN_SIZE,
    JUMP_VOLATILITY,
    MARGIN_INTEREST_RATE,
    NUM_PATHS,
    RNG_SEED,
    TODAY_DEPOSIT,
    WITHDRAWAL_SCHEDULE,
)


class MarketSimulator:
    """
    Vectorized stochastic engine using:
      - one local target factor,
      - one local factor per legacy asset,
      - one zero-drift FX factor per non-base currency,
      - a full dynamic correlation matrix across all factors.

    To keep optimization fast, the engine precomputes target CHF step multipliers
    and aggregate legacy CHF value/maintenance paths once. It does not store a
    full NUM_ASSETS x NUM_PATHS x NUM_DAYS tensor.
    """

    def __init__(self, state: dict, params: dict, end_date: datetime.date):
        self.state = state
        self.params = params

        self.days = (end_date - CURRENT_DATE).days + 1

        if self.days < 2:
            raise ValueError(
                f"Simulation end date {end_date} must be after CURRENT_DATE {CURRENT_DATE}."
            )

        self.dt = 1 / 365.0

        self.factor_names = list(self.params["factor_names"])
        self.factor_index = {name: i for i, name in enumerate(self.factor_names)}
        self.factor_types = dict(self.params["factor_types"])
        self.factor_currencies = dict(self.params["factor_currencies"])

        self.target_factor_name = self.params["target_factor"]["name"]
        self.target_currency = self.params["target_factor"]["currency"]
        self.base_currency = self.params.get("base_currency", "CHF")

        self.legacy_asset_order = list(self.params.get("legacy_asset_order", []))
        self.legacy_assets = dict(self.params.get("legacy_assets", {}))
        self.fx_factors = dict(self.params.get("fx_factors", {}))

        # Precompute market evolution before optimization tests. The optimizer can
        # then test many leverage values without regenerating stochastic paths.
        self._precompute_market_paths()

        # --- Liability Schedule ---
        self.initial_withdrawal_amount = 0.0
        self.withdrawals_by_day = {}

        for w in WITHDRAWAL_SCHEDULE:
            day = (w["date"] - CURRENT_DATE).days
            amount = float(w["amount"])

            if day == 0:
                self.initial_withdrawal_amount += amount
            elif 0 < day < self.days:
                self.withdrawals_by_day[day] = self.withdrawals_by_day.get(day, 0.0) + amount

        self.withdrawal_days = sorted(self.withdrawals_by_day.keys())
        self.withdrawal_amounts = [self.withdrawals_by_day[d] for d in self.withdrawal_days]

        # --- Institutional Calendar: Last Business Day of the Month ---
        business_month_ends = pd.date_range(start=CURRENT_DATE, end=end_date, freq="BME").date

        self.deposits_by_day = {}

        for bme in business_month_ends:
            days_from_start = (bme - CURRENT_DATE).days

            if 0 < days_from_start < self.days:
                if bme.year == 2026:
                    amount = DEFAULT_MONTHLY_DEPOSIT_2026
                else:
                    amount = DEFAULT_MONTHLY_DEPOSIT_FUTURE

                self.deposits_by_day[days_from_start] = (
                    self.deposits_by_day.get(days_from_start, 0.0) + amount
                )

        self.deposit_days = sorted(self.deposits_by_day.keys())
        self.deposit_amounts = [self.deposits_by_day[d] for d in self.deposit_days]

    def _asset_factor_names(self):
        return [self.legacy_assets[t]["factor_name"] for t in self.legacy_asset_order]

    def _fx_factor_name_for_currency(self, currency: str):
        if currency == self.base_currency:
            return None
        info = self.fx_factors.get(currency)
        return None if info is None else info["factor_name"]

    @staticmethod
    def _shared_jump_multiplier(rng: np.random.Generator, jump_lambda: float) -> np.ndarray:
        """Compound-Poisson macro jump multiplier shared by all asset-local factors."""
        jump_counts = rng.poisson(jump_lambda, size=NUM_PATHS)
        jump_log_sizes = np.zeros(NUM_PATHS, dtype=np.float64)

        has_jump = jump_counts > 0
        if np.any(has_jump):
            counts = jump_counts[has_jump].astype(np.float64)
            jump_log_sizes[has_jump] = rng.normal(
                loc=counts * JUMP_MEAN_SIZE,
                scale=np.sqrt(counts) * JUMP_VOLATILITY,
            )

        return np.exp(jump_log_sizes)

    def _precompute_market_paths(self) -> None:
        """
        Precomputes target CHF step multipliers plus aggregate legacy CHF value
        and exact maintenance-margin paths. This preserves per-asset dynamics
        without material path-storage blowup.
        """
        rng = np.random.default_rng(RNG_SEED)

        corr_matrix = np.asarray(self.params["corr_matrix"], dtype=np.float64)
        if corr_matrix.shape != (len(self.factor_names), len(self.factor_names)):
            raise ValueError(
                "[!] Correlation matrix shape does not match factor list.\n"
                f"    Matrix: {corr_matrix.shape}\n"
                f"    Factors: {len(self.factor_names)}"
            )

        try:
            cholesky = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "[!] Correlation matrix is not positive definite after sanitisation."
            ) from exc

        factor_sigmas = np.array(
            [float(self.params["sigma_by_factor"][name]) for name in self.factor_names],
            dtype=np.float64,
        )
        factor_mus = np.array(
            [float(self.params["mu_by_factor"][name]) for name in self.factor_names],
            dtype=np.float64,
        )

        asset_factor_names = [self.target_factor_name] + self._asset_factor_names()
        asset_factor_indices = np.array(
            [self.factor_index[name] for name in asset_factor_names],
            dtype=np.int64,
        )
        asset_mus = factor_mus[asset_factor_indices]
        asset_sigmas = factor_sigmas[asset_factor_indices]
        theta = asset_sigmas**2

        var_assets = np.repeat(theta[:, None], NUM_PATHS, axis=1).astype(np.float64)

        fx_factor_names = [
            name for name in self.factor_names if self.factor_types.get(name) == "fx"
        ]
        fx_factor_indices = np.array(
            [self.factor_index[name] for name in fx_factor_names],
            dtype=np.int64,
        )
        fx_factor_to_row = {name: i for i, name in enumerate(fx_factor_names)}
        fx_sigmas = factor_sigmas[fx_factor_indices] if len(fx_factor_indices) else np.array([])
        fx_mus = factor_mus[fx_factor_indices] if len(fx_factor_indices) else np.array([])

        n_legacy_assets = len(self.legacy_asset_order)
        legacy_values = np.array(
            [self.legacy_assets[t]["v0"] for t in self.legacy_asset_order],
            dtype=np.float64,
        )[:, None]
        legacy_values = np.repeat(legacy_values, NUM_PATHS, axis=1) if n_legacy_assets else np.empty((0, NUM_PATHS))

        legacy_margin_rates = np.array(
            [self.legacy_assets[t]["m"] for t in self.legacy_asset_order],
            dtype=np.float64,
        )[:, None]

        legacy_asset_fx_rows = []
        for ticker in self.legacy_asset_order:
            ccy = self.legacy_assets[ticker]["currency"]
            fx_factor_name = self._fx_factor_name_for_currency(ccy)
            legacy_asset_fx_rows.append(
                None if fx_factor_name is None else fx_factor_to_row[fx_factor_name]
            )

        target_fx_factor_name = self._fx_factor_name_for_currency(self.target_currency)
        target_fx_row = None if target_fx_factor_name is None else fx_factor_to_row[target_fx_factor_name]

        self.target_step_multiplier = np.ones((NUM_PATHS, self.days), dtype=np.float32)
        self.legacy_value_path = np.zeros((NUM_PATHS, self.days), dtype=np.float32)
        self.legacy_maintenance_path = np.zeros((NUM_PATHS, self.days), dtype=np.float32)

        if n_legacy_assets:
            self.legacy_value_path[:, 0] = legacy_values.sum(axis=0).astype(np.float32)
            self.legacy_maintenance_path[:, 0] = (
                legacy_margin_rates * legacy_values
            ).sum(axis=0).astype(np.float32)

        expected_jump = JUMP_FREQUENCY_PER_YEAR * (
            np.exp(JUMP_MEAN_SIZE + 0.5 * JUMP_VOLATILITY**2) - 1
        )
        jump_lambda = JUMP_FREQUENCY_PER_YEAR * self.dt

        kappa = HESTON_KAPPA
        xi = HESTON_XI
        rho_sv = HESTON_RHO
        rho_sv_comp = np.sqrt(max(0.0, 1 - rho_sv**2))
        shock_scale = np.sqrt(self.dt)

        for t in range(1, self.days):
            independent_shocks = rng.normal(
                0.0,
                shock_scale,
                size=(len(self.factor_names), NUM_PATHS),
            )
            correlated_shocks = cholesky @ independent_shocks

            dW_assets = correlated_shocks[asset_factor_indices, :]
            z_vol_assets = rng.normal(
                0.0,
                shock_scale,
                size=(len(asset_factor_names), NUM_PATHS),
            )

            jump_multiplier = self._shared_jump_multiplier(rng, jump_lambda)

            v_prev = np.maximum(var_assets, 0.0)
            dW_v_assets = rho_sv * dW_assets + rho_sv_comp * z_vol_assets
            var_assets = (
                var_assets
                + kappa * (theta[:, None] - v_prev) * self.dt
                + xi * np.sqrt(v_prev) * dW_v_assets
            )
            var_assets = np.maximum(var_assets, 0.0)

            inst_vol_assets = np.sqrt(var_assets)
            asset_drift = (
                asset_mus[:, None]
                - expected_jump
                - 0.5 * inst_vol_assets**2
            ) * self.dt

            asset_local_multipliers = (
                np.exp(asset_drift + inst_vol_assets * dW_assets)
                * jump_multiplier[None, :]
            )

            if len(fx_factor_indices):
                dW_fx = correlated_shocks[fx_factor_indices, :]
                fx_step_multipliers = np.exp(
                    (fx_mus[:, None] - 0.5 * fx_sigmas[:, None] ** 2) * self.dt
                    + fx_sigmas[:, None] * dW_fx
                )
            else:
                fx_step_multipliers = np.empty((0, NUM_PATHS), dtype=np.float64)

            target_multiplier = asset_local_multipliers[0]
            if target_fx_row is not None:
                target_multiplier = target_multiplier * fx_step_multipliers[target_fx_row]

            self.target_step_multiplier[:, t] = target_multiplier.astype(np.float32)

            if n_legacy_assets:
                legacy_local_multipliers = asset_local_multipliers[1:, :]
                legacy_values *= legacy_local_multipliers

                for asset_row, fx_row in enumerate(legacy_asset_fx_rows):
                    if fx_row is not None:
                        legacy_values[asset_row, :] *= fx_step_multipliers[fx_row]

                self.legacy_value_path[:, t] = legacy_values.sum(axis=0).astype(np.float32)
                self.legacy_maintenance_path[:, t] = (
                    legacy_margin_rates * legacy_values
                ).sum(axis=0).astype(np.float32)

    def simulate(self, target_leverage: float, store_paths: bool = True) -> dict:
        path_dtype = np.float32

        # --- Day 0 Mechanics ---
        V_0 = self.state["v_target_0"] + self.state["v_legacy_0"]
        E_0 = V_0 - CURRENT_DEBT + TODAY_DEPOSIT

        target_assets = E_0 * target_leverage
        purchase_amount = max(0.0, target_assets - V_0)

        initial_debt = (
            CURRENT_DEBT
            + purchase_amount
            - TODAY_DEPOSIT
            + self.initial_withdrawal_amount
        )

        if initial_debt < -1e-8:
            raise ValueError(
                "[!] TODAY_DEPOSIT would create positive CHF cash / negative CHF debt.\n"
                f"    CURRENT_DEBT: {CURRENT_DEBT:,.2f}\n"
                f"    TODAY_DEPOSIT: {TODAY_DEPOSIT:,.2f}\n"
                f"    Initial purchase: {purchase_amount:,.2f}\n"
                f"    Resulting debt: {initial_debt:,.2f}\n"
                "    This model assumes you intentionally maintain CHF margin debt."
            )

        initial_debt = max(0.0, initial_debt)

        v_target = np.full(
            NUM_PATHS,
            self.state["v_target_0"] + purchase_amount,
            dtype=np.float64,
        )
        debt = np.full(NUM_PATHS, initial_debt, dtype=np.float64)
        ruined = np.zeros(NUM_PATHS, dtype=bool)
        max_median_leverage = float("nan")

        if store_paths:
            V_target = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            V_legacy = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            Leverage = np.empty((NUM_PATHS, self.days), dtype=path_dtype)

        daily_rate = MARGIN_INTEREST_RATE / 365.0

        def record(day: int) -> None:
            nonlocal max_median_leverage

            v_legacy = self.legacy_value_path[:, day].astype(np.float64, copy=False)
            legacy_maintenance = self.legacy_maintenance_path[:, day].astype(np.float64, copy=False)

            gross_assets = v_target + v_legacy
            equity = gross_assets - debt

            maintenance_margin = self.state["m_target"] * v_target + legacy_maintenance
            excess_liquidity = equity - maintenance_margin
            ruined[:] |= excess_liquidity < 0

            if store_paths:
                leverage = np.full(NUM_PATHS, np.nan, dtype=path_dtype)
                positive_equity = equity > 0

                leverage[positive_equity] = (
                    gross_assets[positive_equity] / equity[positive_equity]
                ).astype(path_dtype)

                if np.any(positive_equity):
                    median_leverage = float(np.nanmedian(leverage))
                    if np.isfinite(median_leverage):
                        if not np.isfinite(max_median_leverage):
                            max_median_leverage = median_leverage
                        else:
                            max_median_leverage = max(max_median_leverage, median_leverage)

                V_target[:, day] = v_target.astype(path_dtype)
                V_legacy[:, day] = v_legacy.astype(path_dtype)
                Leverage[:, day] = leverage

        record(0)

        for t in range(1, self.days):
            v_target *= self.target_step_multiplier[:, t]
            debt *= (1 + daily_rate)

            if t in self.deposits_by_day:
                base_deposit = self.deposits_by_day[t]
                leveraged_purchase = base_deposit * target_leverage
                new_debt = leveraged_purchase - base_deposit

                v_target += leveraged_purchase
                debt += new_debt

            if t in self.withdrawals_by_day:
                debt += self.withdrawals_by_day[t]

            record(t)

        result = {
            "prob_ruin": float(np.mean(ruined)),
            "max_median_leverage": max_median_leverage,
            "time_axis": np.arange(self.days),
            "optimal_purchase_chf": purchase_amount,
        }

        if store_paths:
            result.update(
                {
                    "V_target": V_target,
                    "V_legacy": V_legacy,
                    "Leverage": Leverage,
                }
            )

        return result
