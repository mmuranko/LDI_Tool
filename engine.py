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
    CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX,
    CONTRIBUTION_POLICY_NO_INVEST_MIN
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
    


    @staticmethod
    def _portfolio_leverage(
        v_target: np.ndarray,
        v_legacy: np.ndarray,
        cash: np.ndarray,
        debt: np.ndarray,
    ) -> np.ndarray:
        """
        Total portfolio leverage used for contribution-policy guardrails.

        Leverage = gross_assets / equity
        gross_assets = target assets + legacy assets + cash
        equity = gross_assets - debt

        If equity <= 0, leverage is treated as infinity.
        """
        gross_assets = v_target + v_legacy + cash
        equity = gross_assets - debt

        leverage = np.full_like(gross_assets, np.inf, dtype=np.float64)
        positive_equity = equity > 0.0

        leverage[positive_equity] = (
            gross_assets[positive_equity] / equity[positive_equity]
        )

        return leverage

    @staticmethod
    def _validate_contribution_policy_config() -> None:
        """Validates X/Y contribution guardrails from config.py."""
        if CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX < 1.0:
            raise ValueError(
                "[!] CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX should be >= 1.0."
            )

        if CONTRIBUTION_POLICY_NO_INVEST_MIN <= CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX:
            raise ValueError(
                "[!] Invalid contribution policy guardrails.\n"
                f"    X = CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX = "
                f"{CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX:.2f}\n"
                f"    Y = CONTRIBUTION_POLICY_NO_INVEST_MIN = "
                f"{CONTRIBUTION_POLICY_NO_INVEST_MIN:.2f}\n"
                "    Required: Y > X."
            )

    def _contribution_multiplier(
        self,
        portfolio_leverage: np.ndarray,
        contribution_leverage: float,
        contribution_policy_mode: str = "guardrailed",
    ) -> np.ndarray:
        """
        Maps current portfolio leverage to the target-asset purchase multiplier
        applied to a cash contribution.

        Modes:
          "guardrailed":
              leverage <= X      -> contribution_leverage
              X < leverage < Y   -> 1.0
              leverage >= Y      -> 0.0

          "always_unlevered":
              every contribution is invested at 1.0x, regardless of portfolio leverage.
              This is the correct benchmark for "no future leverage".
        """
        contribution_leverage = float(contribution_leverage)

        if contribution_leverage < 1.0:
            raise ValueError(
                "[!] contribution_leverage must be >= 1.0.\n"
                f"    Received: {contribution_leverage:.4f}"
            )

        if contribution_policy_mode == "always_unlevered":
            return np.ones_like(portfolio_leverage, dtype=np.float64)

        if contribution_policy_mode != "guardrailed":
            raise ValueError(
                "[!] Unknown contribution_policy_mode.\n"
                f"    Received: {contribution_policy_mode!r}\n"
                "    Valid modes: 'guardrailed', 'always_unlevered'."
            )

        self._validate_contribution_policy_config()

        multiplier = np.zeros_like(portfolio_leverage, dtype=np.float64)

        full_policy_mask = portfolio_leverage <= CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX
        unlevered_mask = (
            (portfolio_leverage > CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX)
            & (portfolio_leverage < CONTRIBUTION_POLICY_NO_INVEST_MIN)
        )

        multiplier[full_policy_mask] = contribution_leverage
        multiplier[unlevered_mask] = 1.0

        return multiplier

    @staticmethod
    def _withdraw_from_balance_sheet(
        cash: np.ndarray,
        debt: np.ndarray,
        withdrawal_amount: float,
    ) -> None:
        """
        Applies a liability withdrawal.

        Cash is used first. Any unfunded remainder increases margin debt.
        """
        withdrawal_amount = float(withdrawal_amount)

        if withdrawal_amount <= 0.0:
            return

        cash_used = np.minimum(cash, withdrawal_amount)
        cash -= cash_used

        unfunded_withdrawal = withdrawal_amount - cash_used
        debt += unfunded_withdrawal

    def _apply_contribution_policy(
        self,
        v_target: np.ndarray,
        cash: np.ndarray,
        debt: np.ndarray,
        day: int,
        deposit_amount: float,
        contribution_leverage: float,
        contribution_policy_mode: str = "guardrailed",
    ) -> dict:
        """
        Applies the contribution-leverage policy to a deposit.

        Interpretation:
          - deposit enters the account as cash;
          - cash first offsets existing margin debt;
          - then the policy determines the target-asset purchase amount;
          - purchases use available cash first, then margin debt.

        This avoids the accounting bug where a 'deposit but do not invest'
        contribution disappears from the simulation.
        """
        deposit_amount = float(deposit_amount)

        if deposit_amount < 0.0:
            raise ValueError(
                f"[!] deposit_amount must be non-negative. Received {deposit_amount:,.2f}."
            )

        v_legacy = self.legacy_value_path[:, day].astype(np.float64, copy=False)

        pre_contribution_leverage = self._portfolio_leverage(
            v_target=v_target,
            v_legacy=v_legacy,
            cash=cash,
            debt=debt,
        )

        multiplier = self._contribution_multiplier(
            portfolio_leverage=pre_contribution_leverage,
            contribution_leverage=contribution_leverage,
            contribution_policy_mode=contribution_policy_mode,
        )

        purchase_amount = deposit_amount * multiplier

        if contribution_policy_mode == "always_unlevered":
            full_policy_mask = np.zeros_like(pre_contribution_leverage, dtype=bool)
            unlevered_mask = np.ones_like(pre_contribution_leverage, dtype=bool)
            no_invest_mask = np.zeros_like(pre_contribution_leverage, dtype=bool)
        else:
            full_policy_mask = (
                pre_contribution_leverage <= CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX
            )
            unlevered_mask = (
                (pre_contribution_leverage > CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX)
                & (pre_contribution_leverage < CONTRIBUTION_POLICY_NO_INVEST_MIN)
            )
            no_invest_mask = (
                pre_contribution_leverage >= CONTRIBUTION_POLICY_NO_INVEST_MIN
            )

        if deposit_amount > 0.0:
            # 1. Contribution arrives as cash.
            cash += deposit_amount

            # 2. In a margin account, positive cash offsets margin debt first.
            debt_repayment = np.minimum(cash, debt)
            cash -= debt_repayment
            debt -= debt_repayment

            # 3. Execute the target-asset purchase chosen by the policy.
            cash_used_for_purchase = np.minimum(cash, purchase_amount)
            cash -= cash_used_for_purchase

            borrowed_for_purchase = purchase_amount - cash_used_for_purchase
            debt += borrowed_for_purchase

            v_target += purchase_amount

        return {
            "pre_contribution_leverage": pre_contribution_leverage,
            "multiplier": multiplier,
            "purchase_amount": purchase_amount,
            "full_policy_share": float(np.mean(full_policy_mask)),
            "unlevered_share": float(np.mean(unlevered_mask)),
            "no_invest_share": float(np.mean(no_invest_mask)),
        }

    @staticmethod
    def _contribution_action_label(
        pre_contribution_leverage: float,
        contribution_policy_mode: str = "guardrailed",
    ) -> str:
        """Human-readable action label for a single current-state leverage value."""
        if contribution_policy_mode == "always_unlevered":
            return "benchmark_always_invest_unlevered"

        if pre_contribution_leverage <= CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX:
            return "invest_at_contribution_leverage"

        if pre_contribution_leverage < CONTRIBUTION_POLICY_NO_INVEST_MIN:
            return "invest_unlevered"

        return "deposit_only_do_not_invest"

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

            dW_v_assets = rho_sv * dW_assets + rho_sv_comp * z_vol_assets

            var_assets = (
                var_assets
                + kappa * (theta[:, None] - v_prev) * self.dt
                + xi * np.sqrt(v_prev) * dW_v_assets
            )

            var_assets = np.maximum(var_assets, 0.0)

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

    def simulate(
        self,
        contribution_leverage: float,
        store_paths: bool = True,
        store_final_nav: bool = False,
        contribution_policy_mode: str = "guardrailed",
    ) -> dict:
        
        path_dtype = np.float32
        contribution_leverage = float(contribution_leverage)

        if contribution_leverage < 1.0:
            raise ValueError(
                "[!] contribution_leverage must be >= 1.0.\n"
                f"    Received: {contribution_leverage:.4f}"
            )

        self._validate_contribution_policy_config()

        # --- Day 0 Balance Sheet ---
        v_target = np.full(
            NUM_PATHS,
            self.state["v_target_0"],
            dtype=np.float64,
        )

        cash = np.zeros(NUM_PATHS, dtype=np.float64)
        debt = np.full(NUM_PATHS, CURRENT_DEBT, dtype=np.float64)

        ruined = np.zeros(NUM_PATHS, dtype=bool)
        max_median_leverage = float("nan")

        contribution_policy_log = []

        # Apply any day-0 liability before today's contribution decision.
        # This is conservative: the contribution guardrail sees the post-liability balance sheet.
        if self.initial_withdrawal_amount > 0.0:
            self._withdraw_from_balance_sheet(
                cash=cash,
                debt=debt,
                withdrawal_amount=self.initial_withdrawal_amount,
            )

            # Conservative: a day-0 liability can cause a breach before today's deposit.
            v_legacy_0 = self.legacy_value_path[:, 0].astype(np.float64, copy=False)
            legacy_maintenance_0 = self.legacy_maintenance_path[:, 0].astype(
                np.float64,
                copy=False,
            )
            gross_assets_0 = v_target + v_legacy_0 + cash
            equity_0 = gross_assets_0 - debt
            maintenance_margin_0 = (
                self.state["m_target"] * v_target + legacy_maintenance_0
            )
            ruined[:] |= (equity_0 - maintenance_margin_0) < 0.0

        # Apply TODAY_DEPOSIT using the same contribution policy as future deposits.
        day0_contribution_info = self._apply_contribution_policy(
            v_target=v_target,
            cash=cash,
            debt=debt,
            day=0,
            deposit_amount=TODAY_DEPOSIT,
            contribution_leverage=contribution_leverage,
            contribution_policy_mode=contribution_policy_mode,
        )

        today_pre_contribution_leverage = float(
            day0_contribution_info["pre_contribution_leverage"][0]
        )
        today_contribution_multiplier = float(day0_contribution_info["multiplier"][0])
        today_purchase_chf = float(day0_contribution_info["purchase_amount"][0])
        today_policy_action = self._contribution_action_label(
            today_pre_contribution_leverage,
            contribution_policy_mode=contribution_policy_mode,
        )

        if store_paths:
            contribution_policy_log.append(
                {
                    "day": 0,
                    "deposit_amount": float(TODAY_DEPOSIT),
                    "mean_purchase_amount": float(
                        np.mean(day0_contribution_info["purchase_amount"])
                    ),
                    "full_policy_share": day0_contribution_info["full_policy_share"],
                    "unlevered_share": day0_contribution_info["unlevered_share"],
                    "no_invest_share": day0_contribution_info["no_invest_share"],
                }
            )

            V_target = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            V_legacy = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            Cash = np.empty((NUM_PATHS, self.days), dtype=path_dtype)
            Leverage = np.empty((NUM_PATHS, self.days), dtype=path_dtype)

        daily_rate = MARGIN_INTEREST_RATE / 365.0

        def check_margin(day: int) -> None:
            v_legacy = self.legacy_value_path[:, day].astype(np.float64, copy=False)
            legacy_maintenance = self.legacy_maintenance_path[:, day].astype(
                np.float64,
                copy=False,
            )

            gross_assets = v_target + v_legacy + cash
            equity = gross_assets - debt

            maintenance_margin = self.state["m_target"] * v_target + legacy_maintenance
            excess_liquidity = equity - maintenance_margin

            ruined[:] |= excess_liquidity < 0.0

        def record(day: int) -> None:
            nonlocal max_median_leverage

            v_legacy = self.legacy_value_path[:, day].astype(np.float64, copy=False)
            legacy_maintenance = self.legacy_maintenance_path[:, day].astype(
                np.float64,
                copy=False,
            )

            gross_assets = v_target + v_legacy + cash
            equity = gross_assets - debt

            maintenance_margin = self.state["m_target"] * v_target + legacy_maintenance
            excess_liquidity = equity - maintenance_margin

            ruined[:] |= excess_liquidity < 0.0

            if store_paths:
                leverage = np.full(NUM_PATHS, np.nan, dtype=path_dtype)
                positive_equity = equity > 0.0

                leverage[positive_equity] = (
                    gross_assets[positive_equity] / equity[positive_equity]
                ).astype(path_dtype)

                if np.any(positive_equity):
                    median_leverage = float(np.nanmedian(leverage))

                    if np.isfinite(median_leverage):
                        if not np.isfinite(max_median_leverage):
                            max_median_leverage = median_leverage
                        else:
                            max_median_leverage = max(
                                max_median_leverage,
                                median_leverage,
                            )

                V_target[:, day] = v_target.astype(path_dtype)
                V_legacy[:, day] = v_legacy.astype(path_dtype)
                Cash[:, day] = cash.astype(path_dtype)
                Leverage[:, day] = leverage

        record(0)

        for t in range(1, self.days):
            # 1. Market evolution.
            v_target *= self.target_step_multiplier[:, t]

            # 2. Financing cost.
            debt *= (1.0 + daily_rate)

            # 3. Liabilities first.
            #    We check margin immediately after withdrawals so a same-day
            #    deposit cannot hide a temporary margin breach.
            if t in self.withdrawals_by_day:
                self._withdraw_from_balance_sheet(
                    cash=cash,
                    debt=debt,
                    withdrawal_amount=self.withdrawals_by_day[t],
                )
                check_margin(t)

            # 4. Contribution policy.
            if t in self.deposits_by_day:
                deposit_info = self._apply_contribution_policy(
                    v_target=v_target,
                    cash=cash,
                    debt=debt,
                    day=t,
                    deposit_amount=self.deposits_by_day[t],
                    contribution_leverage=contribution_leverage,
                    contribution_policy_mode=contribution_policy_mode,
                )

                if store_paths:
                    contribution_policy_log.append(
                        {
                            "day": int(t),
                            "deposit_amount": float(self.deposits_by_day[t]),
                            "mean_purchase_amount": float(
                                np.mean(deposit_info["purchase_amount"])
                            ),
                            "full_policy_share": deposit_info["full_policy_share"],
                            "unlevered_share": deposit_info["unlevered_share"],
                            "no_invest_share": deposit_info["no_invest_share"],
                        }
                    )

            record(t)

        # --- Terminal NAV / Equity ---
        # NAV is the economically relevant terminal wealth metric:
        #   NAV = target assets + legacy assets + cash - margin debt
        final_legacy = self.legacy_value_path[:, -1].astype(np.float64, copy=False)
        final_gross_assets = v_target + final_legacy + cash
        final_nav = final_gross_assets - debt

        final_leverage = np.full(NUM_PATHS, np.nan, dtype=np.float64)
        positive_final_nav = final_nav > 0.0
        final_leverage[positive_final_nav] = (
            final_gross_assets[positive_final_nav] / final_nav[positive_final_nav]
        )

        result = {
            "prob_ruin": float(np.mean(ruined)),
            "max_median_leverage": max_median_leverage,
            "time_axis": np.arange(self.days),

            # Today's actionable order.
            "optimal_purchase_chf": today_purchase_chf,
            "today_pre_contribution_leverage": today_pre_contribution_leverage,
            "today_contribution_multiplier": today_contribution_multiplier,
            "today_policy_action": today_policy_action,

            # Backward/transitional metadata.
            "contribution_leverage_tested": contribution_leverage,
        }

        if store_paths:
            result.update(
                {
                    "V_target": V_target,
                    "V_legacy": V_legacy,
                    "Cash": Cash,
                    "Leverage": Leverage,
                    "contribution_policy_log": contribution_policy_log,
                }
            )
        if store_paths or store_final_nav:
            result.update(
                {
                    "Final_NAV": final_nav.copy(),
                    "Final_Gross_Assets": final_gross_assets.copy(),
                    "Final_Debt": debt.copy(),
                    "Final_Cash": cash.copy(),
                    "Final_Leverage": final_leverage.copy(),
                    "Final_Ruined": ruined.copy(),
                    "contribution_policy_mode": contribution_policy_mode,
                }
            )
            
        return result