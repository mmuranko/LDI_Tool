import datetime
import numpy as np
import pandas as pd
import os
from numba import njit, prange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, DEFAULT_MONTHLY_DEPOSIT_2026,
    DEFAULT_MONTHLY_DEPOSIT_FUTURE, HESTON_KAPPA, HESTON_RHO, HESTON_XI,
    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY,
    MARGIN_INTEREST_RATE, NUM_PATHS, RNG_SEED, TODAY_DEPOSIT,
    WITHDRAWAL_SCHEDULE, HISTORY_INTERVAL_DAYS
)

console = Console()

# =============================================================================
# NUMBA KERNEL 1: VECTORIZED CRN GRID SEARCH
# =============================================================================
@njit(parallel=True, fastmath=True)
def _fused_crn_grid_chunk(
    start_idx: int, end_idx: int, days: int, dt: np.float32, 
    current_debt: np.float32, current_sma: np.float32,
    init_withdrawal: np.float32, today_deposit: np.float32, daily_rate: np.float32, 
    leverage_grid: np.ndarray,
    factor_mus: np.ndarray, factor_sigmas: np.ndarray, theta: np.ndarray, 
    kappa: np.float32, xi: np.float32, rho: np.float32, cholesky: np.ndarray, 
    expected_jump: np.float32, jump_lambda: np.float32, jump_mean: np.float32, jump_vol: np.float32,
    asset_v0: np.ndarray, asset_mmr: np.ndarray, asset_imr: np.ndarray, 
    asset_indices: np.ndarray, fx_indices: np.ndarray, fx_rows: np.ndarray,
    deposits: np.ndarray, withdrawals: np.ndarray, out_ruined_grid: np.ndarray
):
    num_factors = len(factor_mus)
    num_assets = len(asset_v0)
    num_fx = len(fx_indices)
    num_levs = len(leverage_grid)
    
    shock_scale = np.float32(np.sqrt(dt))
    rho_comp = np.float32(np.sqrt(max(0.0, 1.0 - rho**2)))
    
    for i in prange(start_idx, end_idx):
        var_assets = theta.copy()
        base_asset_values = asset_v0.copy()
        purchased_active = np.zeros(num_levs, dtype=np.float32)
        
        cashes = np.zeros(num_levs, dtype=np.float32)
        debts = np.full(num_levs, current_debt, dtype=np.float32)
        smas = np.full(num_levs, current_sma, dtype=np.float32)
        local_ruined = np.zeros(num_levs, dtype=np.bool_)
        
        indep_shocks = np.zeros(num_factors, dtype=np.float32)
        corr_shocks = np.zeros(num_factors, dtype=np.float32)
        z_vol_assets = np.zeros(num_assets, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        fx_mults = np.zeros(num_fx, dtype=np.float32)

        base_gross = np.float32(0.0)
        for a in range(num_assets): 
            base_gross += base_asset_values[a]

        for l_idx in range(num_levs):
            if init_withdrawal > np.float32(0.0):
                cash_used = min(cashes[l_idx], init_withdrawal)
                cashes[l_idx] -= cash_used
                debts[l_idx] += (init_withdrawal - cash_used)
                smas[l_idx] -= init_withdrawal 

            if today_deposit > np.float32(0.0):
                cashes[l_idx] += today_deposit
                smas[l_idx] += today_deposit 
                
            debt_repay = min(cashes[l_idx], debts[l_idx])
            cashes[l_idx] -= debt_repay
            debts[l_idx] -= debt_repay
                
            gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
            nlv = gross - debts[l_idx]
            
            target_gross = gross
            if nlv > np.float32(0.0): 
                target_gross = nlv * leverage_grid[l_idx]
                
            purchase = max(np.float32(0.0), target_gross - gross)
            if purchase > np.float32(0.0):
                cash_used = min(cashes[l_idx], purchase)
                cashes[l_idx] -= cash_used
                debts[l_idx] += (purchase - cash_used)
                purchased_active[l_idx] += purchase
                smas[l_idx] -= (purchase * asset_imr[0]) 

            # FIX: Sync gross and nlv AFTER Day 0 trades for accurate margin calculations
            gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
            nlv = gross - debts[l_idx]

            total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
            for a in range(num_assets):
                val = base_asset_values[a]
                if a == 0: val += purchased_active[l_idx]
                total_mmr += val * asset_mmr[a]
                total_imr += val * asset_imr[a]

            smas[l_idx] = max(smas[l_idx], nlv - total_imr) 
            el = nlv - total_mmr 

            if el < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                local_ruined[l_idx] = True

        for t in range(1, days):
            if np.all(local_ruined): break

            for f in range(num_factors): indep_shocks[f] = np.float32(np.random.normal(0.0, shock_scale))
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors): dot += cholesky[r, c] * indep_shocks[c]
                corr_shocks[r] = dot
            
            for a in range(num_assets): z_vol_assets[a] = np.float32(np.random.normal(0.0, shock_scale))
                
            jump_count = np.random.poisson(jump_lambda)
            jump_mult = np.float32(1.0)
            if jump_count > 0:
                jump_log_size = np.random.normal(jump_count * jump_mean, np.sqrt(float(jump_count)) * jump_vol)
                jump_mult = np.float32(np.exp(jump_log_size))
                
            for f in range(num_fx):
                f_idx = fx_indices[f]
                fx_mults[f] = np.float32(np.exp((factor_mus[f_idx] - np.float32(0.5) * factor_sigmas[f_idx]**2) * dt + factor_sigmas[f_idx] * corr_shocks[f_idx]))

            base_gross = np.float32(0.0)
            for a in range(num_assets):
                f_idx = asset_indices[a]
                dW_a = corr_shocks[f_idx]
                v_prev = max(var_assets[a], np.float32(0.0))
                inst_vol = np.float32(np.sqrt(v_prev))
                
                drift = (factor_mus[f_idx] - expected_jump - np.float32(0.5) * inst_vol**2) * dt
                mult = np.float32(np.exp(drift + inst_vol * dW_a)) * jump_mult
                
                dW_v = rho * dW_a + rho_comp * z_vol_assets[a]
                var_assets[a] = max(v_prev + kappa * (theta[a] - v_prev) * dt + xi * inst_vol * dW_v, np.float32(0.0))
                
                fx_row = fx_rows[a]
                if fx_row != -1: mult *= fx_mults[fx_row]
                    
                asset_mults[a] = mult
                base_asset_values[a] *= mult
                base_gross += base_asset_values[a]

            for l_idx in range(num_levs):
                if local_ruined[l_idx]: continue 

                purchased_active[l_idx] *= asset_mults[0]
                debts[l_idx] *= (np.float32(1.0) + daily_rate)

                w_amt = withdrawals[t]
                if w_amt > np.float32(0.0):
                    cash_used = min(cashes[l_idx], w_amt)
                    cashes[l_idx] -= cash_used
                    debts[l_idx] += (w_amt - cash_used)
                    smas[l_idx] -= w_amt 

                d_amt = deposits[t]
                if d_amt > np.float32(0.0):
                    cashes[l_idx] += d_amt
                    smas[l_idx] += d_amt 
                    debt_repay = min(cashes[l_idx], debts[l_idx])
                    cashes[l_idx] -= debt_repay
                    debts[l_idx] -= debt_repay

                    gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                    nlv = gross - debts[l_idx]
                    
                    target_gross = gross
                    if nlv > np.float32(0.0): target_gross = nlv * leverage_grid[l_idx]
                        
                    purchase = max(np.float32(0.0), target_gross - gross)
                    if purchase > np.float32(0.0):
                        cash_used = min(cashes[l_idx], purchase)
                        cashes[l_idx] -= cash_used
                        debts[l_idx] += (purchase - cash_used)
                        purchased_active[l_idx] += purchase
                        smas[l_idx] -= (purchase * asset_imr[0]) 

                # FIX: Force universal sync of gross and nlv every single day after all mechanics
                gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                nlv = gross - debts[l_idx]
                
                total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
                for a in range(num_assets):
                    val = base_asset_values[a]
                    if a == 0: val += purchased_active[l_idx]
                    total_mmr += val * asset_mmr[a]
                    total_imr += val * asset_imr[a]
                
                smas[l_idx] = max(smas[l_idx], nlv - total_imr)
                el = nlv - total_mmr
                
                if el < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                    local_ruined[l_idx] = True

        out_ruined_grid[i, :] = local_ruined


# =============================================================================
# NUMBA KERNEL 2: SINGLE SCENARIO DETAIL EXTRACTOR (Used for plotting)
# =============================================================================
@njit(parallel=True, fastmath=True)
def _fused_simulation_chunk(
    start_idx: int, end_idx: int, days: int, dt: np.float32, 
    current_debt: np.float32, current_sma: np.float32,
    init_withdrawal: np.float32, today_deposit: np.float32, daily_rate: np.float32, target_leverage: np.float32,
    factor_mus: np.ndarray, factor_sigmas: np.ndarray, theta: np.ndarray, 
    kappa: np.float32, xi: np.float32, rho: np.float32, cholesky: np.ndarray, 
    expected_jump: np.float32, jump_lambda: np.float32, jump_mean: np.float32, jump_vol: np.float32,
    asset_v0: np.ndarray, asset_mmr: np.ndarray, asset_imr: np.ndarray, 
    asset_indices: np.ndarray, fx_indices: np.ndarray, fx_rows: np.ndarray,
    deposits: np.ndarray, withdrawals: np.ndarray,
    out_ruined: np.ndarray, out_final_nlv: np.ndarray, out_final_gross: np.ndarray,
    out_final_debt: np.ndarray, out_final_cash: np.ndarray,
    store_history: bool, history_interval: int, out_nlv_hist: np.ndarray, out_lev_hist: np.ndarray
):
    num_factors = len(factor_mus)
    num_assets = len(asset_v0)
    num_fx = len(fx_indices)
    
    shock_scale = np.float32(np.sqrt(dt))
    rho_comp = np.float32(np.sqrt(max(0.0, 1.0 - rho**2)))
    
    for i in prange(start_idx, end_idx):
        cash = np.float32(0.0)
        debt = np.float32(current_debt)
        sma = np.float32(current_sma)
        
        var_assets = theta.copy()
        base_asset_values = asset_v0.copy()
        purchased_active = np.float32(0.0)
        
        indep_shocks = np.zeros(num_factors, dtype=np.float32)
        corr_shocks = np.zeros(num_factors, dtype=np.float32)
        z_vol_assets = np.zeros(num_assets, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        fx_mults = np.zeros(num_fx, dtype=np.float32)

        base_gross = np.float32(0.0)
        for a in range(num_assets): 
            base_gross += base_asset_values[a]

        if init_withdrawal > np.float32(0.0):
            cash_used = min(cash, init_withdrawal)
            cash -= cash_used
            debt += (init_withdrawal - cash_used)
            sma -= init_withdrawal

        if today_deposit > np.float32(0.0):
            cash += today_deposit
            sma += today_deposit
            
        debt_repay = min(cash, debt)
        cash -= debt_repay
        debt -= debt_repay
            
        gross = base_gross + purchased_active + cash
        nlv = gross - debt
        
        target_gross = gross
        if nlv > np.float32(0.0): target_gross = nlv * target_leverage
            
        purchase = max(np.float32(0.0), target_gross - gross)
        if purchase > np.float32(0.0):
            cash_used = min(cash, purchase)
            cash -= cash_used
            debt += (purchase - cash_used)
            purchased_active += purchase
            sma -= (purchase * asset_imr[0])

        # FIX: Sync gross and nlv AFTER Day 0 trades for accurate logging
        gross = base_gross + purchased_active + cash
        nlv = gross - debt

        total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
        for a in range(num_assets):
            val = base_asset_values[a]
            if a == 0: val += purchased_active
            total_mmr += val * asset_mmr[a]
            total_imr += val * asset_imr[a]

        sma = max(sma, nlv - total_imr)
        
        hist_idx = 0
        if store_history:
            out_nlv_hist[i, hist_idx] = nlv
            out_lev_hist[i, hist_idx] = gross / nlv if nlv > np.float32(0.0) else np.nan
            hist_idx += 1

        if (nlv - total_mmr) < np.float32(0.0) or sma < np.float32(0.0): 
            out_ruined[i] = True
            continue 

        for t in range(1, days):
            for f in range(num_factors): indep_shocks[f] = np.float32(np.random.normal(0.0, shock_scale))
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors): dot += cholesky[r, c] * indep_shocks[c]
                corr_shocks[r] = dot
            
            for a in range(num_assets): z_vol_assets[a] = np.float32(np.random.normal(0.0, shock_scale))
                
            jump_count = np.random.poisson(jump_lambda)
            jump_mult = np.float32(1.0)
            if jump_count > 0:
                jump_log_size = np.random.normal(jump_count * jump_mean, np.sqrt(float(jump_count)) * jump_vol)
                jump_mult = np.float32(np.exp(jump_log_size))
                
            for f in range(num_fx):
                f_idx = fx_indices[f]
                fx_mults[f] = np.float32(np.exp((factor_mus[f_idx] - np.float32(0.5) * factor_sigmas[f_idx]**2) * dt + factor_sigmas[f_idx] * corr_shocks[f_idx]))

            base_gross = np.float32(0.0)
            for a in range(num_assets):
                f_idx = asset_indices[a]
                dW_a = corr_shocks[f_idx]
                v_prev = max(var_assets[a], np.float32(0.0))
                inst_vol = np.float32(np.sqrt(v_prev))
                
                drift = (factor_mus[f_idx] - expected_jump - np.float32(0.5) * inst_vol**2) * dt
                mult = np.float32(np.exp(drift + inst_vol * dW_a)) * jump_mult
                
                dW_v = rho * dW_a + rho_comp * z_vol_assets[a]
                var_assets[a] = max(v_prev + kappa * (theta[a] - v_prev) * dt + xi * inst_vol * dW_v, np.float32(0.0))
                
                fx_row = fx_rows[a]
                if fx_row != -1: mult *= fx_mults[fx_row]
                    
                asset_mults[a] = mult
                base_asset_values[a] *= mult
                base_gross += base_asset_values[a]

            purchased_active *= asset_mults[0]
            debt *= (np.float32(1.0) + daily_rate)

            w_amt = withdrawals[t]
            if w_amt > np.float32(0.0):
                cash_used = min(cash, w_amt)
                cash -= cash_used
                debt += (w_amt - cash_used)
                sma -= w_amt

            d_amt = deposits[t]
            if d_amt > np.float32(0.0):
                cash += d_amt
                sma += d_amt
                debt_repay = min(cash, debt)
                cash -= debt_repay
                debt -= debt_repay

                gross = base_gross + purchased_active + cash
                nlv = gross - debt
                
                target_gross = gross
                if nlv > np.float32(0.0): target_gross = nlv * target_leverage
                    
                purchase = max(np.float32(0.0), target_gross - gross)
                if purchase > np.float32(0.0):
                    cash_used = min(cash, purchase)
                    cash -= cash_used
                    debt += (purchase - cash_used)
                    purchased_active += purchase
                    sma -= (purchase * asset_imr[0])

            # FIX: Force universal sync of gross and nlv every single day after all mechanics
            gross = base_gross + purchased_active + cash
            nlv = gross - debt
            
            total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
            for a in range(num_assets):
                val = base_asset_values[a]
                if a == 0: val += purchased_active
                total_mmr += val * asset_mmr[a]
                total_imr += val * asset_imr[a]
                
            sma = max(sma, nlv - total_imr)
            
            if (nlv - total_mmr) < np.float32(0.0) or sma < np.float32(0.0): 
                out_ruined[i] = True
                break 

            if store_history and t % history_interval == 0:
                out_nlv_hist[i, hist_idx] = nlv
                out_lev_hist[i, hist_idx] = gross / nlv if nlv > np.float32(0.0) else np.nan
                hist_idx += 1

        if not out_ruined[i]:
            out_final_gross[i] = base_gross + purchased_active + cash
            out_final_nlv[i] = nlv
            out_final_debt[i] = debt
            out_final_cash[i] = cash

# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================
class MarketSimulator:
    def __init__(self, state: dict, params: dict, end_date: datetime.date, skip_next_deposit: bool = False):
        self.state = state
        self.params = params
        self.days = (end_date - CURRENT_DATE).days + 1

        if self.days < 2:
            raise ValueError(f"Simulation end date {end_date} must be after CURRENT_DATE.")

        self.dt = 1 / 365.0
        self.portfolio_order = self.state["portfolio_order"]
        self.assets_dict = self.state["assets_dict"]

        self.factor_names = list(self.params["factor_names"])
        self.factor_index = {name: i for i, name in enumerate(self.factor_names)}
        self.factor_types = dict(self.params["factor_types"])
        self.base_currency = self.params.get("base_currency", "CHF")
        self.fx_factors = dict(self.params.get("fx_factors", {}))

        self.initial_withdrawal_amount = 0.0
        self.withdrawals_arr = np.zeros(self.days, dtype=np.float32)
        for w in WITHDRAWAL_SCHEDULE:
            day = (w["date"] - CURRENT_DATE).days
            amount = float(w["amount"])
            if day == 0: self.initial_withdrawal_amount += amount
            elif 0 < day < self.days: self.withdrawals_arr[day] += amount

        self.deposits_arr = np.zeros(self.days, dtype=np.float32)
        business_month_ends = pd.date_range(start=CURRENT_DATE, end=end_date, freq="BME").date
        for bme in business_month_ends:
            days_from_start = (bme - CURRENT_DATE).days
            if 0 < days_from_start < self.days:
                amount = DEFAULT_MONTHLY_DEPOSIT_2026 if bme.year == 2026 else DEFAULT_MONTHLY_DEPOSIT_FUTURE
                self.deposits_arr[days_from_start] += amount

        if skip_next_deposit:
            first_deposit_idx = np.nonzero(self.deposits_arr)[0]
            if len(first_deposit_idx) > 0:
                self.deposits_arr[first_deposit_idx[0]] = 0.0

        self._prepare_vectors()

    def _fx_factor_name_for_currency(self, currency: str):
        if currency == self.base_currency: return None
        info = self.fx_factors.get(currency)
        return None if info is None else info["factor_name"]

    def _prepare_vectors(self) -> None:
        self.corr_matrix = np.asarray(self.params["corr_matrix"], dtype=np.float32)
        self.cholesky = np.linalg.cholesky(self.corr_matrix)

        self.factor_sigmas = np.array([self.params["sigma_by_factor"][n] for n in self.factor_names], dtype=np.float32)
        self.factor_mus = np.array([self.params["mu_by_factor"][n] for n in self.factor_names], dtype=np.float32)

        asset_factor_names = [self.params["assets_params"][t]["factor_name"] for t in self.portfolio_order]
        self.asset_indices = np.array([self.factor_index[n] for n in asset_factor_names], dtype=np.int64)
        
        asset_sigmas = self.factor_sigmas[self.asset_indices]
        self.theta = np.array(asset_sigmas**2, dtype=np.float32)

        fx_names = [n for n in self.factor_names if self.factor_types.get(n) == "fx"]
        self.fx_indices = np.array([self.factor_index[n] for n in fx_names], dtype=np.int64)
        fx_to_row = {n: i for i, n in enumerate(fx_names)}

        self.asset_v0 = np.array([self.assets_dict[t]["v0"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_mmr = np.array([self.assets_dict[t]["mmr"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_imr = np.array([self.assets_dict[t]["imr"] for t in self.portfolio_order], dtype=np.float32)

        fx_rows_list = [None if self._fx_factor_name_for_currency(self.assets_dict[t]["currency"]) is None else fx_to_row[self._fx_factor_name_for_currency(self.assets_dict[t]["currency"])] for t in self.portfolio_order]
        self.fx_rows_arr = np.array([-1 if r is None else r for r in fx_rows_list], dtype=np.int64)

        self.expected_jump = np.float32(JUMP_FREQUENCY_PER_YEAR * (np.exp(JUMP_MEAN_SIZE + 0.5 * JUMP_VOLATILITY**2) - 1.0))
        self.jump_lambda = np.float32(JUMP_FREQUENCY_PER_YEAR * self.dt)

    def simulate_grid(self, leverage_grid: np.ndarray) -> np.ndarray:
        np.random.seed(RNG_SEED)
        leverage_grid = np.asarray(leverage_grid, dtype=np.float32)
        num_levs = len(leverage_grid)
        
        ruined_grid = np.zeros((NUM_PATHS, num_levs), dtype=np.bool_)
        chunk_size = 10000 
        
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=40, style="blue"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Running CRN Optimizer ({num_levs} Leverage Tiers)...", total=NUM_PATHS)
            
            for start_idx in range(0, NUM_PATHS, chunk_size):
                end_idx = min(start_idx + chunk_size, NUM_PATHS)
                
                _fused_crn_grid_chunk(
                    start_idx=start_idx, end_idx=end_idx, days=self.days, dt=np.float32(self.dt),
                    current_debt=np.float32(CURRENT_DEBT), current_sma=np.float32(CURRENT_SMA),
                    init_withdrawal=np.float32(self.initial_withdrawal_amount), today_deposit=np.float32(TODAY_DEPOSIT),
                    daily_rate=np.float32(MARGIN_INTEREST_RATE / 365.0), leverage_grid=leverage_grid,
                    factor_mus=self.factor_mus, factor_sigmas=self.factor_sigmas, theta=self.theta,
                    kappa=np.float32(HESTON_KAPPA), xi=np.float32(HESTON_XI), rho=np.float32(HESTON_RHO), cholesky=self.cholesky,
                    expected_jump=self.expected_jump, jump_lambda=self.jump_lambda, jump_mean=np.float32(JUMP_MEAN_SIZE), jump_vol=np.float32(JUMP_VOLATILITY),
                    asset_v0=self.asset_v0, asset_mmr=self.asset_mmr, asset_imr=self.asset_imr,
                    asset_indices=self.asset_indices, fx_indices=self.fx_indices, fx_rows=self.fx_rows_arr,
                    deposits=self.deposits_arr, withdrawals=self.withdrawals_arr, out_ruined_grid=ruined_grid
                )
                progress.advance(task, advance=(end_idx - start_idx))

        return np.mean(ruined_grid, axis=0)

    def simulate(self, target_leverage: float, store_paths: bool = True, store_history: bool = False) -> dict:
        np.random.seed(RNG_SEED)

        ruined = np.zeros(NUM_PATHS, dtype=np.bool_)
        final_nlv = np.zeros(NUM_PATHS, dtype=np.float32)
        final_gross = np.zeros(NUM_PATHS, dtype=np.float32)
        final_debt = np.zeros(NUM_PATHS, dtype=np.float32)
        final_cash = np.zeros(NUM_PATHS, dtype=np.float32)

        if store_history:
            nlv_file = f'temp_nlv_hist_{target_leverage:.2f}.dat'
            lev_file = f'temp_lev_hist_{target_leverage:.2f}.dat'
            nlv_hist = np.memmap(nlv_file, dtype=np.float32, mode='w+', shape=(NUM_PATHS, self.days))
            lev_hist = np.memmap(lev_file, dtype=np.float32, mode='w+', shape=(NUM_PATHS, self.days))
            history_interval = 1
        else:
            nlv_hist = np.empty((0,0), dtype=np.float32)
            lev_hist = np.empty((0,0), dtype=np.float32)
            history_interval = HISTORY_INTERVAL_DAYS

        chunk_size = 10000 
        
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=40, style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Extracting Vectors to Disk ({target_leverage:.2f}x)...", total=NUM_PATHS)
            
            for start_idx in range(0, NUM_PATHS, chunk_size):
                end_idx = min(start_idx + chunk_size, NUM_PATHS)
                
                _fused_simulation_chunk(
                    start_idx=start_idx, end_idx=end_idx, days=self.days, dt=np.float32(self.dt),
                    current_debt=np.float32(CURRENT_DEBT), current_sma=np.float32(CURRENT_SMA),
                    init_withdrawal=np.float32(self.initial_withdrawal_amount), today_deposit=np.float32(TODAY_DEPOSIT),
                    daily_rate=np.float32(MARGIN_INTEREST_RATE / 365.0), target_leverage=np.float32(target_leverage),
                    factor_mus=self.factor_mus, factor_sigmas=self.factor_sigmas, theta=self.theta,
                    kappa=np.float32(HESTON_KAPPA), xi=np.float32(HESTON_XI), rho=np.float32(HESTON_RHO), cholesky=self.cholesky,
                    expected_jump=self.expected_jump, jump_lambda=self.jump_lambda, jump_mean=np.float32(JUMP_MEAN_SIZE), jump_vol=np.float32(JUMP_VOLATILITY),
                    asset_v0=self.asset_v0, asset_mmr=self.asset_mmr, asset_imr=self.asset_imr,
                    asset_indices=self.asset_indices, fx_indices=self.fx_indices, fx_rows=self.fx_rows_arr,
                    deposits=self.deposits_arr, withdrawals=self.withdrawals_arr,
                    out_ruined=ruined, out_final_nlv=final_nlv, out_final_gross=final_gross,
                    out_final_debt=final_debt, out_final_cash=final_cash,
                    store_history=store_history, history_interval=history_interval,
                    out_nlv_hist=nlv_hist, out_lev_hist=lev_hist
                )
                progress.advance(task, advance=(end_idx - start_idx))

        result = {
            "prob_ruin": float(np.mean(ruined)),
            "mean_terminal_nlv": float(np.mean(final_nlv[~ruined])) if np.any(~ruined) else 0.0,
            "target_leverage_tested": target_leverage
        }
            
        if store_paths:
            result.update({
                "Final_NLV": final_nlv,
                "Final_Gross_Assets": final_gross,
                "Final_Debt": final_debt,
                "Final_Cash": final_cash,
                "Final_Ruined": ruined,
            })
            
        if store_history:
            nlv_hist.flush()
            lev_hist.flush()
            
            history_days = [CURRENT_DATE + datetime.timedelta(days=t) for t in range(self.days)]
        result["history_paths"] = {
                "nlv_file": nlv_file,
                "lev_file": lev_file,
                "shape": (NUM_PATHS, self.days),
                "dates": history_days
            }
            
        return result