import numpy as np
import pandas as pd
import datetime
from config import (
    CURRENT_DATE, CURRENT_DEBT,
    DEFAULT_MONTHLY_DEPOSIT_2026, DEFAULT_MONTHLY_DEPOSIT_FUTURE,
    MARGIN_INTEREST_RATE, WITHDRAWAL_SCHEDULE, NUM_PATHS,
    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY,
    TODAY_DEPOSIT, HESTON_KAPPA, HESTON_XI, HESTON_RHO, RNG_SEED
)


class MarketSimulator:
    """
    Vectorized stochastic asset engine.

    Target and local legacy bucket factors use the existing Heston + Merton-jump model.
    FX factors are separate correlated GBMs with zero drift and no Heston/jump component.
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

        self.legacy_currencies = list(params.get("legacy_bucket_currencies", []))
        self.target_currency = params["target_currency"]

        # Precompute market multipliers once. The optimizer can then test many leverage
        # values without regenerating or reprocessing stochastic paths.
        self._precompute_market_multipliers()

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

    @staticmethod
    def _jump_multiplier(rng: np.random.Generator, jump_lambda: float) -> np.ndarray:
        """
        Simulates compound-Poisson log jumps.

        This uses the mathematically correct N-jump aggregation: conditional on N jumps,
        the total log jump is Normal(N * mean, sqrt(N) * volatility).
        """
        jump_counts = rng.poisson(jump_lambda, size=NUM_PATHS)
        jump_log_sizes = rng.normal(
            loc=jump_counts * JUMP_MEAN_SIZE,
            scale=np.sqrt(jump_counts) * JUMP_VOLATILITY
        )
        return np.exp(jump_log_sizes)

    def _precompute_market_multipliers(self) -> None:
        """
        Precomputes one-day CHF return multipliers once.

        The full dynamic factor covariance is used while generating shocks. Only the final
        CHF multipliers needed by the balance-sheet simulator are retained:
        target CHF multiplier and one CHF multiplier per legacy currency bucket.
        """
        rng = np.random.default_rng(RNG_SEED)

        factor_names = list(self.params["factor_names"])
        factor_index = {name: i for i, name in enumerate(factor_names)}
        n_factors = len(factor_names)

        if n_factors == 0:
            raise ValueError("[!] No stochastic factors were supplied to the simulator.")

        corr_matrix = np.asarray(self.params["corr_matrix"], dtype=float)
        if corr_matrix.shape != (n_factors, n_factors):
            raise ValueError(
                f"[!] Correlation matrix shape {corr_matrix.shape} does not match "
                f"{n_factors} factors."
            )

        try:
            cholesky = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "[!] Correlation matrix is not positive definite after sanitisation."
            ) from exc

        self.target_step_multiplier = np.ones((NUM_PATHS, self.days), dtype=np.float32)
        self.legacy_bucket_step_multipliers = np.ones(
            (len(self.legacy_currencies), NUM_PATHS, self.days),
            dtype=np.float32
        )

        # Local asset factors: target + one local legacy factor per currency bucket.
        target_factor_name = self.params["target_factor_name"]
        legacy_factor_names = [
            self.params["legacy_bucket_factors"][ccy]
            for ccy in self.legacy_currencies
        ]
        asset_factor_names = [target_factor_name] + legacy_factor_names
        asset_factor_indices = np.array([factor_index[name] for name in asset_factor_names], dtype=int)
        n_asset_factors = len(asset_factor_names)

        asset_mu = np.array(
            [self.params["mu_by_factor"][name] for name in asset_factor_names],
            dtype=float
        )
        asset_sigma = np.array(
            [self.params["sigma_by_factor"][name] for name in asset_factor_names],
            dtype=float
        )
        theta_asset = asset_sigma ** 2
        var_asset = np.repeat(theta_asset[:, None], NUM_PATHS, axis=1).astype(np.float64)

        # FX factors: one per exposed non-base currency, always zero-drift GBMs.
        fx_factor_names_by_currency = dict(self.params.get("fx_factor_names_by_currency", {}))
        fx_factor_indices_by_currency = {
            ccy: factor_index[factor_name]
            for ccy, factor_name in fx_factor_names_by_currency.items()
        }
        fx_sigma_by_currency = {
            ccy: float(self.params["sigma_by_factor"][factor_name])
            for ccy, factor_name in fx_factor_names_by_currency.items()
        }

        expected_jump = JUMP_FREQUENCY_PER_YEAR * (
            np.exp(JUMP_MEAN_SIZE + 0.5 * JUMP_VOLATILITY ** 2) - 1
        )
        jump_lambda = JUMP_FREQUENCY_PER_YEAR * self.dt

        kappa = float(HESTON_KAPPA)
        xi = float(HESTON_XI)
        rho_sv = float(HESTON_RHO)

        if not -1.0 <= rho_sv <= 1.0:
            raise ValueError(f"[!] HESTON_RHO must be in [-1, 1], got {rho_sv}.")

        rho_sv_comp = np.sqrt(max(0.0, 1.0 - rho_sv ** 2))
        shock_scale = np.sqrt(self.dt)

        for t in range(1, self.days):
            independent_shocks = rng.normal(0.0, shock_scale, size=(n_factors, NUM_PATHS))
            dW_all = cholesky @ independent_shocks

            # --- Target and local legacy buckets: Heston + Merton jumps. ---
            dW_asset = dW_all[asset_factor_indices, :]
            z_vol_asset = rng.normal(0.0, shock_scale, size=(n_asset_factors, NUM_PATHS))

            jump_multiplier = self._jump_multiplier(rng, jump_lambda)

            var_prev = np.maximum(var_asset, 0.0)
            dW_var = rho_sv * dW_asset + rho_sv_comp * z_vol_asset

            var_asset = (
                var_asset
                + kappa * (theta_asset[:, None] - var_prev) * self.dt
                + xi * np.sqrt(var_prev) * dW_var
            )

            inst_vol_asset = np.sqrt(np.maximum(var_asset, 0.0))
            drift_asset = (
                asset_mu[:, None]
                - expected_jump
                - 0.5 * inst_vol_asset ** 2
            ) * self.dt

            local_asset_multiplier = (
                np.exp(drift_asset + inst_vol_asset * dW_asset)
                * jump_multiplier[None, :]
            )

            # --- FX: correlated zero-drift GBM, no jumps, no Heston. ---
            fx_multiplier_by_currency = {}
            for ccy, idx in fx_factor_indices_by_currency.items():
                sigma_fx = fx_sigma_by_currency[ccy]
                dW_fx = dW_all[idx, :]
                fx_multiplier_by_currency[ccy] = np.exp(
                    (-0.5 * sigma_fx ** 2) * self.dt + sigma_fx * dW_fx
                )

            # --- Retain only the CHF multipliers the balance sheet needs. ---
            target_fx_multiplier = fx_multiplier_by_currency.get(self.target_currency, 1.0)
            self.target_step_multiplier[:, t] = (
                local_asset_multiplier[0, :] * target_fx_multiplier
            ).astype(np.float32)

            for bucket_row, ccy in enumerate(self.legacy_currencies):
                local_row = 1 + bucket_row
                fx_multiplier = fx_multiplier_by_currency.get(ccy, 1.0)
                self.legacy_bucket_step_multipliers[bucket_row, :, t] = (
                    local_asset_multiplier[local_row, :] * fx_multiplier
                ).astype(np.float32)

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
            dtype=np.float64
        )

        legacy_initial_values = np.array(
            [self.state["legacy_by_currency"][ccy]["v0"] for ccy in self.legacy_currencies],
            dtype=np.float64
        )
        legacy_margin_rates = np.array(
            [self.state["legacy_by_currency"][ccy]["m"] for ccy in self.legacy_currencies],
            dtype=np.float64
        )

        if len(self.legacy_currencies) > 0:
            v_legacy_by_bucket = np.repeat(legacy_initial_values[:, None], NUM_PATHS, axis=1)
            legacy_margin_rates = legacy_margin_rates[:, None]
        else:
            v_legacy_by_bucket = np.zeros((0, NUM_PATHS), dtype=np.float64)
            legacy_margin_rates = np.zeros((0, 1), dtype=np.float64)

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

            v_legacy_total = v_legacy_by_bucket.sum(axis=0)
            gross_assets = v_target + v_legacy_total
            equity = gross_assets - debt

            maintenance_margin = (
                self.state["m_target"] * v_target
                + (legacy_margin_rates * v_legacy_by_bucket).sum(axis=0)
            )

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
                V_legacy[:, day] = v_legacy_total.astype(path_dtype)
                Leverage[:, day] = leverage

        record(0)

        for t in range(1, self.days):
            v_target *= self.target_step_multiplier[:, t]

            if len(self.legacy_currencies) > 0:
                v_legacy_by_bucket *= self.legacy_bucket_step_multipliers[:, :, t]

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
            "legacy_currencies": self.legacy_currencies,
            "target_currency": self.target_currency
        }

        if store_paths:
            result.update({
                "V_target": V_target,
                "V_legacy": V_legacy,
                "Leverage": Leverage
            })

        return result
