import numpy as np
import pandas as pd
import datetime
from config import (CURRENT_DATE, CURRENT_DEBT, DEFAULT_MONTHLY_DEPOSIT_2026, DEFAULT_MONTHLY_DEPOSIT_FUTURE,
                    MARGIN_INTEREST_RATE, WITHDRAWAL_SCHEDULE, NUM_PATHS,
                    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY, TODAY_DEPOSIT, 
                    HESTON_KAPPA, HESTON_XI, HESTON_RHO, RNG_SEED)

class MarketSimulator:
    """Vectorized Stochastic Asset Engine with Merton Jump-Diffusion."""
    
    def __init__(self, state: dict, params: dict, end_date: datetime.date):
        self.state = state
        self.params = params
        
        self.days = (end_date - CURRENT_DATE).days + 1

        if self.days < 2:
            raise ValueError(
                f"Simulation end date {end_date} must be after CURRENT_DATE {CURRENT_DATE}."
            )

        self.dt = 1 / 365.0

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

    def _precompute_market_multipliers(self) -> None:
        """
        Precomputes stochastic one-day return multipliers once.

        This is much more memory- and time-efficient than regenerating full
        asset, volatility, jump, and correlated-shock matrices inside every
        leverage test during optimization.
        """
        rng = np.random.default_rng(RNG_SEED)

        try:
            cholesky = np.linalg.cholesky(self.params["corr_matrix"])
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "[!] Correlation matrix is not positive definite after sanitisation."
            ) from exc

        self.target_step_multiplier = np.ones((NUM_PATHS, self.days), dtype=np.float32)
        self.legacy_local_step_multiplier = np.ones((NUM_PATHS, self.days), dtype=np.float32)
        self.fx_step_multiplier = np.ones((NUM_PATHS, self.days), dtype=np.float32)

        theta_target = self.params["sigma_target"] ** 2
        theta_leg = self.params["sigma_legacy_loc"] ** 2

        var_target = np.full(NUM_PATHS, theta_target, dtype=np.float64)
        var_leg = np.full(NUM_PATHS, theta_leg, dtype=np.float64)

        expected_jump = JUMP_FREQUENCY_PER_YEAR * (
            np.exp(JUMP_MEAN_SIZE + 0.5 * JUMP_VOLATILITY ** 2) - 1
        )

        jump_lambda = JUMP_FREQUENCY_PER_YEAR * self.dt

        kappa = HESTON_KAPPA
        xi = HESTON_XI
        rho_sv = HESTON_RHO
        rho_sv_comp = np.sqrt(1 - rho_sv ** 2)

        shock_scale = np.sqrt(self.dt)

        for t in range(1, self.days):
            z = rng.normal(0.0, shock_scale, size=(3, NUM_PATHS))
            dW_target, dW_leg_loc, dW_fx = cholesky @ z

            z_vol_target = rng.normal(0.0, shock_scale, size=NUM_PATHS)
            z_vol_leg = rng.normal(0.0, shock_scale, size=NUM_PATHS)

            jump_counts = rng.poisson(jump_lambda, size=NUM_PATHS)
            jump_sizes = rng.normal(JUMP_MEAN_SIZE, JUMP_VOLATILITY, size=NUM_PATHS)
            jump_multiplier = np.exp(jump_counts * jump_sizes)

            v_prev_target = np.maximum(var_target, 0.0)
            v_prev_leg = np.maximum(var_leg, 0.0)

            dW_v_target = rho_sv * dW_target + rho_sv_comp * z_vol_target
            dW_v_leg = rho_sv * dW_leg_loc + rho_sv_comp * z_vol_leg

            var_target = (
                var_target
                + kappa * (theta_target - v_prev_target) * self.dt
                + xi * np.sqrt(v_prev_target) * dW_v_target
            )

            var_leg = (
                var_leg
                + kappa * (theta_leg - v_prev_leg) * self.dt
                + xi * np.sqrt(v_prev_leg) * dW_v_leg
            )

            inst_vol_target = np.sqrt(np.maximum(var_target, 0.0))
            inst_vol_leg = np.sqrt(np.maximum(var_leg, 0.0))

            drift_target = (
                self.params["mu_target"]
                - expected_jump
                - 0.5 * inst_vol_target ** 2
            ) * self.dt

            drift_leg = (
                self.params["mu_legacy_loc"]
                - expected_jump
                - 0.5 * inst_vol_leg ** 2
            ) * self.dt

            drift_fx = (
                self.params["mu_fx"]
                - 0.5 * self.params["sigma_fx"] ** 2
            ) * self.dt

            self.target_step_multiplier[:, t] = (
                np.exp(drift_target + inst_vol_target * dW_target) * jump_multiplier
            ).astype(np.float32)

            self.legacy_local_step_multiplier[:, t] = (
                np.exp(drift_leg + inst_vol_leg * dW_leg_loc) * jump_multiplier
            ).astype(np.float32)

            self.fx_step_multiplier[:, t] = (
                np.exp(drift_fx + self.params["sigma_fx"] * dW_fx)
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

        v_target = np.full(NUM_PATHS, self.state["v_target_0"] + purchase_amount, dtype=np.float64)
        v_legacy = np.full(NUM_PATHS, self.state["v_legacy_0"], dtype=np.float64)
        debt = np.full(NUM_PATHS, initial_debt, dtype=np.float64)

        local_index = np.ones(NUM_PATHS, dtype=np.float64)
        fx_index = np.ones(NUM_PATHS, dtype=np.float64)

        ruined = np.zeros(NUM_PATHS, dtype=bool)
        max_median_leverage = float("nan")

        if store_paths:
            V_target = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            V_legacy = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            Leverage = np.empty((NUM_PATHS, self.days), dtype=path_dtype)

        daily_rate = MARGIN_INTEREST_RATE / 365.0

        def record(day: int) -> None:
            nonlocal max_median_leverage

            gross_assets = v_target + v_legacy
            equity = gross_assets - debt

            maintenance_margin = (
                self.state["m_target"] * v_target
                + self.state["m_legacy"] * v_legacy
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
                V_legacy[:, day] = v_legacy.astype(path_dtype)
                Leverage[:, day] = leverage

        record(0)

        for t in range(1, self.days):
            v_target *= self.target_step_multiplier[:, t]

            local_index *= self.legacy_local_step_multiplier[:, t]
            fx_index *= self.fx_step_multiplier[:, t]
            v_legacy = self.state["v_legacy_0"] * local_index * fx_index

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
            "optimal_purchase_chf": purchase_amount
        }

        if store_paths:
            result.update({
                "V_target": V_target,
                "V_legacy": V_legacy,
                "Leverage": Leverage
            })

        return result