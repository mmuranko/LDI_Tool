import datetime
import numpy as np
import pandas as pd
from numba import njit, prange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, DEPOSIT_SCHEDULE,
    DEFAULT_MONTHLY_DEPOSIT_FUTURE, MARGIN_INTEREST_RATE, NUM_PATHS, 
    RNG_SEED, WITHDRAWAL_SCHEDULE, HISTORY_INTERVAL_DAYS,
    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY
)

console = Console()

# =============================================================================
# NUMBA KERNEL 1: VECTORIZED CRN GRID SEARCH WITH UNIFIED JUMP CRASHES
# =============================================================================
@njit(parallel=True, fastmath=True)
def _fused_hmm_crn_grid_chunk(
    start_idx: int, end_idx: int, days: int, 
    current_debt: np.float32, current_sma: np.float32,
    init_withdrawal: np.float32, daily_rate: np.float32, leverage_grid: np.ndarray,
    num_factors: int, num_assets: int,
    pca_loadings: np.ndarray, residual_scales: np.ndarray, asset_means: np.ndarray, asset_scales: np.ndarray,
    P_matrix: np.ndarray, chol_0: np.ndarray, chol_1: np.ndarray,
    daily_jump_prob: np.float32, mu_jump: np.float32, sigma_jump: np.float32,
    asset_v0: np.ndarray, asset_mmr: np.ndarray, asset_imr: np.ndarray, 
    deposits: np.ndarray, withdrawals: np.ndarray, out_ruined_grid: np.ndarray
):
    num_levs = len(leverage_grid)
    
    for i in prange(start_idx, end_idx):
        base_asset_values = asset_v0.copy()
        purchased_active = np.zeros(num_levs, dtype=np.float32)
        cashes = np.zeros(num_levs, dtype=np.float32)
        debts = np.full(num_levs, current_debt, dtype=np.float32)
        smas = np.full(num_levs, current_sma, dtype=np.float32)
        local_ruined = np.zeros(num_levs, dtype=np.bool_)
        
        Z = np.zeros(num_factors, dtype=np.float32)
        F = np.zeros(num_factors, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        
        current_regime = 0
        base_gross = np.float32(0.0)
        for a in range(num_assets): 
            base_gross += base_asset_values[a]

        # --- Day 0 Immediate Rebalancing ---
        for l_idx in range(num_levs):
            if init_withdrawal > np.float32(0.0):
                cash_used = min(cashes[l_idx], init_withdrawal)
                cashes[l_idx] -= cash_used
                debts[l_idx] += (init_withdrawal - cash_used)
                smas[l_idx] -= init_withdrawal 

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

            gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
            nlv = gross - debts[l_idx]

            total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
            for a in range(num_assets):
                val = base_asset_values[a]
                if a == 0: val += purchased_active[l_idx]
                total_mmr += val * asset_mmr[a]
                total_imr += val * asset_imr[a]

            if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                local_ruined[l_idx] = True

        # --- Time Horizon Simulation Loop ---
        for t in range(1, days):
            if np.all(local_ruined): break
            
            # 1. Markov Regime Transition
            u = np.random.rand()
            if current_regime == 0:
                if u > P_matrix[0, 0]: current_regime = 1
            else:
                if u > P_matrix[1, 1]: current_regime = 0
                
            # 2. Correlated Factor Shocks
            for f in range(num_factors): Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors):
                    dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot
                
            # 3. Macroeconomic Tail Risk Shock (Shared Merton Jump Process)
            jump_shock = np.float32(0.0)
            if np.random.rand() < daily_jump_prob:
                jump_shock = np.float32(mu_jump + sigma_jump * np.random.randn())

            # 4. Asset Price Evolution
            base_gross = np.float32(0.0)
            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors):
                    r_factor += pca_loadings[a, f] * F[f]
                
                # Add idiosyncratic residual variance
                idio_shock = np.float32(residual_scales[a] * np.random.randn())
                r_std = r_factor + idio_shock
                
                log_ret = r_std * asset_scales[a] + asset_means[a] + jump_shock
                asset_mults[a] = np.float32(np.exp(log_ret))
                base_asset_values[a] *= asset_mults[a]
                base_gross += base_asset_values[a]

            # 5. Margin Accounting & Cash Flows
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

                gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                nlv = gross - debts[l_idx]
                
                total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
                for a in range(num_assets):
                    val = base_asset_values[a]
                    if a == 0: val += purchased_active[l_idx]
                    total_mmr += val * asset_mmr[a]
                    total_imr += val * asset_imr[a]
                
                smas[l_idx] = max(smas[l_idx], nlv - total_imr)
                if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                    local_ruined[l_idx] = True

        out_ruined_grid[i, :] = local_ruined

# =============================================================================
# NUMBA KERNEL 2: SINGLE SCENARIO DETAIL EXTRACTOR
# =============================================================================
@njit(parallel=True, fastmath=True)
def _fused_hmm_simulation_chunk(
    start_idx: int, end_idx: int, days: int, 
    current_debt: np.float32, current_sma: np.float32,
    init_withdrawal: np.float32, daily_rate: np.float32, target_leverage: np.float32,
    num_factors: int, num_assets: int,
    pca_loadings: np.ndarray, residual_scales: np.ndarray, asset_means: np.ndarray, asset_scales: np.ndarray,
    P_matrix: np.ndarray, chol_0: np.ndarray, chol_1: np.ndarray,
    daily_jump_prob: np.float32, mu_jump: np.float32, sigma_jump: np.float32,
    asset_v0: np.ndarray, asset_mmr: np.ndarray, asset_imr: np.ndarray, 
    deposits: np.ndarray, withdrawals: np.ndarray,
    out_ruined: np.ndarray, out_final_nlv: np.ndarray, out_final_gross: np.ndarray,
    out_final_debt: np.ndarray, out_final_cash: np.ndarray,
    store_history: bool, history_interval: int, out_nlv_hist: np.ndarray, out_lev_hist: np.ndarray
):
    for i in prange(start_idx, end_idx):
        cash, debt, sma = np.float32(0.0), np.float32(current_debt), np.float32(current_sma)
        base_asset_values = asset_v0.copy()
        purchased_active = np.float32(0.0)
        
        Z = np.zeros(num_factors, dtype=np.float32)
        F = np.zeros(num_factors, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        
        current_regime = 0
        base_gross = np.float32(0.0)
        for a in range(num_assets): 
            base_gross += base_asset_values[a]

        if init_withdrawal > np.float32(0.0):
            cash_used = min(cash, init_withdrawal)
            cash -= cash_used
            debt += (init_withdrawal - cash_used)
            sma -= init_withdrawal

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

        gross = base_gross + purchased_active + cash
        nlv = gross - debt

        total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
        for a in range(num_assets):
            val = base_asset_values[a]
            if a == 0: val += purchased_active
            total_mmr += val * asset_mmr[a]
            total_imr += val * asset_imr[a]

        hist_idx = 0
        if store_history:
            out_nlv_hist[i, hist_idx] = nlv
            out_lev_hist[i, hist_idx] = gross / nlv if nlv > np.float32(0.0) else np.nan
            hist_idx += 1

        if (nlv - total_mmr) < np.float32(0.0) or sma < np.float32(0.0): 
            out_ruined[i] = True
            continue 

        for t in range(1, days):
            u = np.random.rand()
            if current_regime == 0:
                if u > P_matrix[0, 0]: current_regime = 1
            else:
                if u > P_matrix[1, 1]: current_regime = 0
                
            for f in range(num_factors): Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors):
                    dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot
                
            jump_shock = np.float32(0.0)
            if np.random.rand() < daily_jump_prob:
                jump_shock = np.float32(mu_jump + sigma_jump * np.random.randn())

            base_gross = np.float32(0.0)
            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors):
                    r_factor += pca_loadings[a, f] * F[f]
                
                idio_shock = np.float32(residual_scales[a] * np.random.randn())
                r_std = r_factor + idio_shock
                
                log_ret = r_std * asset_scales[a] + asset_means[a] + jump_shock
                asset_mults[a] = np.float32(np.exp(log_ret))
                base_asset_values[a] *= asset_mults[a]
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
        
        self.portfolio_order = self.state["portfolio_order"]
        self.assets_dict = self.state["assets_dict"]

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
                amount = DEFAULT_MONTHLY_DEPOSIT_FUTURE
                for start_date, end_date_sched, scheduled_amount in DEPOSIT_SCHEDULE:
                    after_start = (start_date is None) or (bme >= start_date)
                    before_end = (end_date_sched is None) or (bme <= end_date_sched)
                    if after_start and before_end:
                        amount = scheduled_amount
                        break
                self.deposits_arr[days_from_start] += amount

        if skip_next_deposit:
            first_deposit_idx = np.nonzero(self.deposits_arr)[0]
            if len(first_deposit_idx) > 0:
                self.deposits_arr[first_deposit_idx[0]] = 0.0

        self._prepare_vectors()

    def _prepare_vectors(self) -> None:
        self.num_factors = len(self.params["chol_0"])
        self.num_assets = len(self.portfolio_order)
        
        bayesian_names = self.params["asset_names"]
        name_to_idx = {name: i for i, name in enumerate(bayesian_names)}
        idx_mapping = [name_to_idx[name] for name in self.portfolio_order]
        
        self.pca_loadings = self.params["pca_loadings"][idx_mapping]
        self.residual_scales = self.params["residual_scales"][idx_mapping]
        self.asset_means = self.params["asset_means"][idx_mapping]
        self.asset_scales = self.params["asset_scales"][idx_mapping]
        
        self.P_matrix = self.params["P"]
        self.chol_0 = self.params["chol_0"]
        self.chol_1 = self.params["chol_1"]

        self.daily_jump_prob = np.float32(JUMP_FREQUENCY_PER_YEAR / 252.0)
        self.mu_jump = np.float32(JUMP_MEAN_SIZE)
        self.sigma_jump = np.float32(JUMP_VOLATILITY)

        self.asset_v0 = np.array([self.assets_dict[t]["v0"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_mmr = np.array([self.assets_dict[t]["mmr"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_imr = np.array([self.assets_dict[t]["imr"] for t in self.portfolio_order], dtype=np.float32)

    def simulate_grid(self, leverage_grid: np.ndarray) -> np.ndarray:
        np.random.seed(RNG_SEED)
        leverage_grid = np.asarray(leverage_grid, dtype=np.float32)
        num_levs = len(leverage_grid)
        ruined_grid = np.zeros((NUM_PATHS, num_levs), dtype=np.bool_)
        
        with Progress(TextColumn("{task.description}"), BarColumn(bar_width=40, style="blue"), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task(f"Simulating {num_levs} Leverage Tiers with Unified Jump Crashes...", total=NUM_PATHS)
            chunk_size = 10000 
            for start_idx in range(0, NUM_PATHS, chunk_size):
                end_idx = min(start_idx + chunk_size, NUM_PATHS)
                _fused_hmm_crn_grid_chunk(
                    start_idx, end_idx, self.days, 
                    np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.0), leverage_grid,
                    self.num_factors, self.num_assets,
                    self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1,
                    self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr,
                    self.deposits_arr, self.withdrawals_arr, ruined_grid
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

        history_interval = 1 if store_history else HISTORY_INTERVAL_DAYS
        if store_history:
            nlv_file = f'temp_nlv_hist_{target_leverage:.2f}.dat'
            lev_file = f'temp_lev_hist_{target_leverage:.2f}.dat'
            nlv_hist = np.memmap(nlv_file, dtype=np.float32, mode='w+', shape=(NUM_PATHS, self.days))
            lev_hist = np.memmap(lev_file, dtype=np.float32, mode='w+', shape=(NUM_PATHS, self.days))
        else:
            nlv_hist = np.empty((0,0), dtype=np.float32)
            lev_hist = np.empty((0,0), dtype=np.float32)

        with Progress(TextColumn("{task.description}"), BarColumn(bar_width=40, style="green"), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task(f"Simulating Jump Trajectories ({target_leverage:.2f}x)...", total=NUM_PATHS)
            chunk_size = 10000 
            for start_idx in range(0, NUM_PATHS, chunk_size):
                end_idx = min(start_idx + chunk_size, NUM_PATHS)
                _fused_hmm_simulation_chunk(
                    start_idx, end_idx, self.days, 
                    np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.0), np.float32(target_leverage),
                    self.num_factors, self.num_assets,
                    self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1,
                    self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr,
                    self.deposits_arr, self.withdrawals_arr,
                    ruined, final_nlv, final_gross, final_debt, final_cash,
                    store_history, history_interval, nlv_hist, lev_hist
                )
                progress.advance(task, advance=(end_idx - start_idx))

        result = {
            "prob_ruin": float(np.mean(ruined)), 
            "mean_terminal_nlv": float(np.mean(final_nlv[~ruined])) if np.any(~ruined) else 0.0, 
            "target_leverage_tested": target_leverage
        }
        if store_paths:
            result.update({
                "Final_NLV": final_nlv, "Final_Gross_Assets": final_gross, 
                "Final_Debt": final_debt, "Final_Cash": final_cash, "Final_Ruined": ruined
            })
        if store_history:
            nlv_hist.flush()
            lev_hist.flush()
            result["history_paths"] = {
                "nlv_file": nlv_file, "lev_file": lev_file, "shape": (NUM_PATHS, self.days), 
                "dates": [CURRENT_DATE + datetime.timedelta(days=t) for t in range(self.days)]
            }
        return result