import datetime
import numpy as np
import pandas as pd
from numba import njit, prange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, DEPOSIT_SCHEDULE, DEFAULT_MONTHLY_DEPOSIT_FUTURE, 
    MARGIN_INTEREST_RATE, NUM_PATHS, RNG_SEED, WITHDRAWAL_SCHEDULE, HISTORY_INTERVAL_DAYS,
    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY, BOND_REGISTRY
)

console = Console(highlight=False)

@njit(parallel=True, fastmath=True)
def _fused_hmm_crn_grid_chunk(
    start_idx, end_idx, days, current_debt, current_sma, init_withdrawal, daily_rate, leverage_grid,
    num_factors, num_assets, pca_loadings, residual_scales, asset_means, asset_scales,
    P_matrix, chol_0, chol_1, daily_jump_prob, mu_jump, sigma_jump, asset_v0, asset_mmr, asset_imr, 
    deposits, withdrawals, out_ruined_grid, asset_type, div_mean, div_vol, div_yield_init,
    is_div_day, div_weight, bond_duration, bond_convexity, bond_initial_ytm, bond_cf_days, bond_cf_amounts
):
    num_levs = len(leverage_grid)
    dt = np.float32(1.0 / 252.0)
    
    for i in prange(start_idx, end_idx):
        base_asset_values = asset_v0.copy()
        div_yields = div_yield_init.copy()
        bond_yields = bond_initial_ytm.copy()
        
        purchased_active = np.zeros(num_levs, dtype=np.float32)
        cashes = np.zeros(num_levs, dtype=np.float32)
        debts = np.full(num_levs, current_debt, dtype=np.float32)
        smas = np.full(num_levs, current_sma, dtype=np.float32)
        local_ruined = np.zeros(num_levs, dtype=np.bool_)
        
        Z = np.zeros(num_factors, dtype=np.float32)
        F = np.zeros(num_factors, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        div_cash_base = np.zeros(num_assets, dtype=np.float32)
        
        current_regime = 0
        base_gross = np.float32(0.0)
        for a in range(num_assets): base_gross += base_asset_values[a]

        # --- Day 0 Margin Initiation ---
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
            
            target_gross = nlv * leverage_grid[l_idx] if nlv > np.float32(0.0) else gross
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
                val = base_asset_values[a] + (purchased_active[l_idx] if a == 0 else np.float32(0.0))
                total_mmr += val * asset_mmr[a]
                total_imr += val * asset_imr[a]

            if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                local_ruined[l_idx] = True

        # --- Temporal Loop ---
        for t in range(1, days):
            if np.all(local_ruined): break
            
            if current_regime == 0:
                if np.random.rand() > P_matrix[0, 0]: current_regime = 1
            else:
                if np.random.rand() > P_matrix[1, 1]: current_regime = 0
                
            for f in range(num_factors): Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors): dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot
                
            jump_shock = np.float32(mu_jump + sigma_jump * np.random.randn()) if np.random.rand() < daily_jump_prob else np.float32(0.0)

            base_gross = np.float32(0.0)
            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors): r_factor += pca_loadings[a, f] * F[f]
                
                idio_shock = np.float32(residual_scales[a] * np.random.randn())
                log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a] + jump_shock
                asset_mults[a] = np.float32(np.exp(log_ret))
                
                if asset_type[a] == 0:  
                    base_asset_values[a] *= asset_mults[a]
                    dy = np.float32(2.0 * (div_mean[a] - div_yields[a]) * dt) + np.float32(div_vol[a] * np.sqrt(max(np.float32(0.0), div_yields[a]) * dt) * np.random.randn())
                    div_yields[a] = max(np.float32(0.0), div_yields[a] + dy)
                    
                    if is_div_day[t, a]:
                        div_amt = base_asset_values[a] * (div_yields[a] * div_weight[a])
                        base_asset_values[a] -= div_amt  
                        div_cash_base[a] = div_amt
                    else: div_cash_base[a] = np.float32(0.0)
                        
                elif asset_type[a] == 1:  
                    D, C = bond_duration[a], bond_convexity[a]
                    discriminant = np.float32(D**2 + 2.0 * C * log_ret)
                    yield_shock = (D - np.sqrt(discriminant)) / C if discriminant >= 0.0 else -log_ret / D
                    
                    dy = np.float32(1.0 * (bond_initial_ytm[a] - bond_yields[a]) * dt) + np.float32(yield_shock)
                    bond_yields[a] = max(np.float32(0.001), bond_yields[a] + dy)
                    
                    bp, cf_cash = np.float32(0.0), np.float32(0.0)
                    for c in range(20): 
                        cf_day = bond_cf_days[a, c]
                        if cf_day == -1: break
                        if cf_day == t: cf_cash += bond_cf_amounts[a, c]
                        elif cf_day > t: bp += bond_cf_amounts[a, c] * np.exp(-bond_yields[a] * np.float32((cf_day - t) / 365.25))
                            
                    base_asset_values[a] = bp  
                    div_cash_base[a] = cf_cash
                
                base_gross += base_asset_values[a]

            # --- Accounting & Reinvestment Logic ---
            for l_idx in range(num_levs):
                if local_ruined[l_idx]: continue 
                
                # Active asset price change and debt accrual
                purchased_active[l_idx] *= asset_mults[0]
                debts[l_idx] *= (np.float32(1.0) + daily_rate)
                
                # --- EXPLICIT DIVIDEND ROUTING RULE ---
                # Pool all dividends from passive assets
                total_div_cash = np.sum(div_cash_base)
                
                # Extract dividend generated by the active leveraged asset itself
                if is_div_day[t, 0]:
                    div_active = purchased_active[l_idx] * (div_yields[0] * div_weight[0])
                    purchased_active[l_idx] -= div_active
                    total_div_cash += div_active
                
                if total_div_cash > np.float32(0.0):
                    # 1. Dividend hits account in simulated Base Currency (CHF)
                    cashes[l_idx] += total_div_cash
                    smas[l_idx] += total_div_cash
                    
                    # 2. Check exactly what the leverage ratio is at this specific moment
                    gross_eval = base_gross + purchased_active[l_idx] + cashes[l_idx]
                    nlv_eval = gross_eval - debts[l_idx]
                    current_lev = gross_eval / nlv_eval if nlv_eval > np.float32(0.0) else np.float32(999.0)
                    
                    # 3. Conditional Reinvestment
                    if current_lev <= leverage_grid[l_idx]:
                        # Under or at target leverage: Reinvest entirely into ACTIVE_ASSET
                        cashes[l_idx] -= total_div_cash
                        purchased_active[l_idx] += total_div_cash
                        smas[l_idx] -= total_div_cash * asset_imr[0]
                    else:
                        # Over target leverage: Automatically sweep cash to pay down CHF margin debt
                        debt_repay_div = min(cashes[l_idx], debts[l_idx])
                        cashes[l_idx] -= debt_repay_div
                        debts[l_idx] -= debt_repay_div

                # --- Standard External Cash Flows ---
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
                    
                # End-of-day natural cash sweep for new deposits
                debt_repay = min(cashes[l_idx], debts[l_idx])
                cashes[l_idx] -= debt_repay
                debts[l_idx] -= debt_repay

                # --- General Target Leverage Balancing ---
                gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                nlv = gross - debts[l_idx]
                
                target_gross = nlv * leverage_grid[l_idx] if nlv > np.float32(0.0) else gross
                purchase = max(np.float32(0.0), target_gross - gross)
                if purchase > np.float32(0.0):
                    cash_used = min(cashes[l_idx], purchase)
                    cashes[l_idx] -= cash_used
                    debts[l_idx] += (purchase - cash_used)
                    purchased_active[l_idx] += purchase
                    smas[l_idx] -= (purchase * asset_imr[0]) 

                # --- Ruin Verification ---
                gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                nlv = gross - debts[l_idx]
                
                total_mmr, total_imr = np.float32(0.0), np.float32(0.0)
                for a in range(num_assets):
                    val = base_asset_values[a] + (purchased_active[l_idx] if a == 0 else np.float32(0.0))
                    total_mmr += val * asset_mmr[a]
                    total_imr += val * asset_imr[a]
                
                smas[l_idx] = max(smas[l_idx], nlv - total_imr)
                if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0): 
                    local_ruined[l_idx] = True

        out_ruined_grid[i, :] = local_ruined

@njit(parallel=True, fastmath=True)
def _fused_hmm_simulation_chunk(
    start_idx, end_idx, days, current_debt, current_sma, init_withdrawal, daily_rate, target_leverage,
    num_factors, num_assets, pca_loadings, residual_scales, asset_means, asset_scales,
    P_matrix, chol_0, chol_1, daily_jump_prob, mu_jump, sigma_jump, asset_v0, asset_mmr, asset_imr, 
    deposits, withdrawals, out_ruined, out_final_nlv, out_final_gross, out_final_debt, out_final_cash,
    store_history, history_interval, out_nlv_hist, out_lev_hist,
    asset_type, div_mean, div_vol, div_yield_init, is_div_day, div_weight, 
    bond_duration, bond_convexity, bond_initial_ytm, bond_cf_days, bond_cf_amounts
):
    dt = np.float32(1.0 / 252.0)
    for i in prange(start_idx, end_idx):
        cash, debt, sma = np.float32(0.0), np.float32(current_debt), np.float32(current_sma)
        base_asset_values = asset_v0.copy()
        div_yields = div_yield_init.copy()
        bond_yields = bond_initial_ytm.copy()
        purchased_active = np.float32(0.0)
        
        Z = np.zeros(num_factors, dtype=np.float32)
        F = np.zeros(num_factors, dtype=np.float32)
        asset_mults = np.zeros(num_assets, dtype=np.float32)
        div_cash_base = np.zeros(num_assets, dtype=np.float32)
        
        current_regime = 0
        base_gross = np.float32(0.0)
        for a in range(num_assets): base_gross += base_asset_values[a]

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
        
        target_gross = nlv * target_leverage if nlv > np.float32(0.0) else gross
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
            val = base_asset_values[a] + (purchased_active if a == 0 else np.float32(0.0))
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
            if current_regime == 0:
                if np.random.rand() > P_matrix[0, 0]: current_regime = 1
            else:
                if np.random.rand() > P_matrix[1, 1]: current_regime = 0
                
            for f in range(num_factors): Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors): dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot
                
            jump_shock = np.float32(mu_jump + sigma_jump * np.random.randn()) if np.random.rand() < daily_jump_prob else np.float32(0.0)

            base_gross = np.float32(0.0)
            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors): r_factor += pca_loadings[a, f] * F[f]
                idio_shock = np.float32(residual_scales[a] * np.random.randn())
                log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a] + jump_shock
                asset_mults[a] = np.float32(np.exp(log_ret))

                if asset_type[a] == 0:  
                    base_asset_values[a] *= asset_mults[a]
                    dy = np.float32(2.0 * (div_mean[a] - div_yields[a]) * dt) + np.float32(div_vol[a] * np.sqrt(max(np.float32(0.0), div_yields[a]) * dt) * np.random.randn())
                    div_yields[a] = max(np.float32(0.0), div_yields[a] + dy)
                    if is_div_day[t, a]:
                        div_amt = base_asset_values[a] * (div_yields[a] * div_weight[a])
                        base_asset_values[a] -= div_amt  
                        div_cash_base[a] = div_amt
                    else: div_cash_base[a] = np.float32(0.0)
                elif asset_type[a] == 1:  
                    D, C = bond_duration[a], bond_convexity[a]
                    discriminant = np.float32(D**2 + 2.0 * C * log_ret)
                    yield_shock = (D - np.sqrt(discriminant)) / C if discriminant >= 0.0 else -log_ret / D
                    dy = np.float32(1.0 * (bond_initial_ytm[a] - bond_yields[a]) * dt) + np.float32(yield_shock)
                    bond_yields[a] = max(np.float32(0.001), bond_yields[a] + dy)
                    bp, cf_cash = np.float32(0.0), np.float32(0.0)
                    for c in range(20): 
                        cf_day = bond_cf_days[a, c]
                        if cf_day == -1: break
                        if cf_day == t: cf_cash += bond_cf_amounts[a, c]
                        elif cf_day > t: bp += bond_cf_amounts[a, c] * np.exp(-bond_yields[a] * np.float32((cf_day - t) / 365.25))
                    base_asset_values[a] = bp  
                    div_cash_base[a] = cf_cash
                
                base_gross += base_asset_values[a]

            purchased_active *= asset_mults[0]
            debt *= (np.float32(1.0) + daily_rate)
            
            # --- EXPLICIT DIVIDEND ROUTING RULE ---
            total_div_cash = np.sum(div_cash_base)
            if is_div_day[t, 0]:
                div_active = purchased_active * (div_yields[0] * div_weight[0])
                purchased_active -= div_active
                total_div_cash += div_active

            if total_div_cash > np.float32(0.0):
                cash += total_div_cash
                sma += total_div_cash
                
                gross_eval = base_gross + purchased_active + cash
                nlv_eval = gross_eval - debt
                current_lev = gross_eval / nlv_eval if nlv_eval > np.float32(0.0) else np.float32(999.0)
                
                if current_lev <= target_leverage:
                    cash -= total_div_cash
                    purchased_active += total_div_cash
                    sma -= total_div_cash * asset_imr[0]
                else:
                    debt_repay_div = min(cash, debt)
                    cash -= debt_repay_div
                    debt -= debt_repay_div

            # --- Standard External Cash Flows ---
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
            
            target_gross = nlv * target_leverage if nlv > np.float32(0.0) else gross
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
                val = base_asset_values[a] + (purchased_active if a == 0 else np.float32(0.0))
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

class MarketSimulator:
    def __init__(self, state: dict, params: dict, end_date: datetime.date, skip_next_deposit: bool = False):
        self.state, self.params = state, params
        self.days = (end_date - CURRENT_DATE).days + 1
        self.portfolio_order, self.assets_dict = self.state["portfolio_order"], self.state["assets_dict"]
        
        self.initial_withdrawal_amount = 0.0
        self.withdrawals_arr = np.zeros(self.days, dtype=np.float32)
        for w in WITHDRAWAL_SCHEDULE:
            day = (w["date"] - CURRENT_DATE).days
            if day == 0: self.initial_withdrawal_amount += float(w["amount"])
            elif 0 < day < self.days: self.withdrawals_arr[day] += float(w["amount"])

        self.deposits_arr = np.zeros(self.days, dtype=np.float32)
        for bme in pd.date_range(start=CURRENT_DATE, end=end_date, freq="BME").date:
            days_from_start = (bme - CURRENT_DATE).days
            if 0 < days_from_start < self.days:
                amount = DEFAULT_MONTHLY_DEPOSIT_FUTURE
                for start_date, end_date_sched, scheduled_amount in DEPOSIT_SCHEDULE:
                    if (start_date is None or bme >= start_date) and (end_date_sched is None or bme <= end_date_sched):
                        amount = scheduled_amount
                        break
                self.deposits_arr[days_from_start] += amount

        if skip_next_deposit and len(np.nonzero(self.deposits_arr)[0]) > 0:
            self.deposits_arr[np.nonzero(self.deposits_arr)[0][0]] = 0.0

        self._prepare_vectors()

    def _prepare_vectors(self) -> None:
        self.num_factors, self.num_assets = len(self.params["chol_0"]), len(self.portfolio_order)
        idx_mapping = [{name: i for i, name in enumerate(self.params["asset_names"])}[name] for name in self.portfolio_order]
        
        self.pca_loadings = self.params["pca_loadings"][idx_mapping]
        self.residual_scales = self.params["residual_scales"][idx_mapping]
        self.asset_means = self.params["asset_means"][idx_mapping]
        self.asset_scales = self.params["asset_scales"][idx_mapping]
        self.P_matrix, self.chol_0, self.chol_1 = self.params["P"], self.params["chol_0"], self.params["chol_1"]
        
        self.daily_jump_prob, self.mu_jump, self.sigma_jump = np.float32(JUMP_FREQUENCY_PER_YEAR / 252.0), np.float32(JUMP_MEAN_SIZE), np.float32(JUMP_VOLATILITY)
        self.asset_v0 = np.array([self.assets_dict[t]["v0"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_mmr = np.array([self.assets_dict[t]["mmr"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_imr = np.array([self.assets_dict[t]["imr"] for t in self.portfolio_order], dtype=np.float32)
        
        self.asset_type = np.zeros(self.num_assets, dtype=np.int32)
        self.div_mean, self.div_vol, self.div_yield_init, self.div_weight = (np.zeros(self.num_assets, dtype=np.float32) for _ in range(4))
        self.is_div_day = np.zeros((self.days, self.num_assets), dtype=np.bool_)
        self.bond_duration, self.bond_convexity, self.bond_initial_ytm = (np.zeros(self.num_assets, dtype=np.float32) for _ in range(3))
        self.bond_cf_days = np.full((self.num_assets, 20), -1, dtype=np.int32)
        self.bond_cf_amounts = np.zeros((self.num_assets, 20), dtype=np.float32)

        for a, ticker in enumerate(self.portfolio_order):
            info = self.assets_dict[ticker]
            if ticker in BOND_REGISTRY:
                self.asset_type[a] = 1
                b_info = BOND_REGISTRY[ticker]
                
                self.bond_duration[a] = b_info["proxy_duration"]
                self.bond_convexity[a] = b_info.get("proxy_convexity", 0.0)
                self.bond_initial_ytm[a] = b_info["current_ytm"]
                
                total_face = info["quantity"] * b_info["face_value_local"] * info["fx_rate"]
                curr_date = b_info["maturity_date"]
                cfs = []
                while curr_date > CURRENT_DATE:
                    cfs.append(curr_date)
                    curr_date = datetime.date(curr_date.year - 1, curr_date.month, curr_date.day)
                cfs.sort()
                for idx, cf_d in enumerate(cfs[:20]):
                    days_out = (cf_d - CURRENT_DATE).days
                    if days_out < self.days:
                        self.bond_cf_days[a, idx] = days_out
                        self.bond_cf_amounts[a, idx] = total_face * (1 + b_info["coupon_rate"]) if cf_d == b_info["maturity_date"] else total_face * b_info["coupon_rate"]
            else:
                self.asset_type[a] = 0
                self.div_mean[a], self.div_vol[a], self.div_yield_init[a] = info.get("mean_yield", 0.0), info.get("yield_vol", 0.0), info.get("ttm_yield", 0.0)
                months = info.get("payout_months", [])
                if months:
                    self.div_weight[a] = 1.0 / len(months)
                    for t in range(self.days):
                        d = CURRENT_DATE + datetime.timedelta(days=t)
                        if d.month in months and d.day == 15: self.is_div_day[t, a] = True

    def simulate_grid(self, leverage_grid: np.ndarray) -> np.ndarray:
        np.random.seed(RNG_SEED)
        num_levs = len(leverage_grid)
        ruined_grid = np.zeros((NUM_PATHS, num_levs), dtype=np.bool_)
        
        with Progress(TextColumn("[dim]{task.description}[/dim]"), BarColumn(bar_width=40), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task(f"Simulating {num_levs} Strategy Tiers...", total=NUM_PATHS)
            for start_idx in range(0, NUM_PATHS, 10000):
                end_idx = min(start_idx + 10000, NUM_PATHS)
                _fused_hmm_crn_grid_chunk(
                    start_idx, end_idx, self.days, np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.0), np.asarray(leverage_grid, dtype=np.float32),
                    self.num_factors, self.num_assets, self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1, self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr, self.deposits_arr, self.withdrawals_arr, ruined_grid,
                    self.asset_type, self.div_mean, self.div_vol, self.div_yield_init, self.is_div_day, self.div_weight,
                    self.bond_duration, self.bond_convexity, self.bond_initial_ytm, self.bond_cf_days, self.bond_cf_amounts
                )
                progress.advance(task, advance=(end_idx - start_idx))
        return np.mean(ruined_grid, axis=0)

    def simulate(self, target_leverage: float, store_paths: bool = True, store_history: bool = False) -> dict:
        np.random.seed(RNG_SEED)
        ruined, final_nlv, final_gross, final_debt, final_cash = (np.zeros(NUM_PATHS, dtype=dt) for dt in [np.bool_, np.float32, np.float32, np.float32, np.float32])
        history_interval = 1 if store_history else HISTORY_INTERVAL_DAYS
        
        if store_history:
            nlv_file, lev_file = f'temp_nlv_hist_{target_leverage:.2f}.dat', f'temp_lev_hist_{target_leverage:.2f}.dat'
            nlv_hist, lev_hist = (np.memmap(f, dtype=np.float32, mode='w+', shape=(NUM_PATHS, self.days)) for f in (nlv_file, lev_file))
        else: nlv_hist = lev_hist = np.empty((0,0), dtype=np.float32)

        with Progress(TextColumn("[dim]{task.description}[/dim]"), BarColumn(bar_width=40), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
            task = progress.add_task(f"Extracting Path Details ({target_leverage:.2f}x)...", total=NUM_PATHS)
            for start_idx in range(0, NUM_PATHS, 10000):
                end_idx = min(start_idx + 10000, NUM_PATHS)
                _fused_hmm_simulation_chunk(
                    start_idx, end_idx, self.days, np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.0), np.float32(target_leverage),
                    self.num_factors, self.num_assets, self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1, self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr, self.deposits_arr, self.withdrawals_arr,
                    ruined, final_nlv, final_gross, final_debt, final_cash, store_history, history_interval, nlv_hist, lev_hist,
                    self.asset_type, self.div_mean, self.div_vol, self.div_yield_init, self.is_div_day, self.div_weight,
                    self.bond_duration, self.bond_convexity, self.bond_initial_ytm, self.bond_cf_days, self.bond_cf_amounts
                )
                progress.advance(task, advance=(end_idx - start_idx))

        result = {"prob_ruin": float(np.mean(ruined)), "mean_terminal_nlv": float(np.mean(final_nlv[~ruined])) if np.any(~ruined) else 0.0, "target_leverage_tested": target_leverage}
        if store_paths: result.update({"Final_NLV": final_nlv, "Final_Gross_Assets": final_gross, "Final_Debt": final_debt, "Final_Cash": final_cash, "Final_Ruined": ruined})
        if store_history:
            nlv_hist.flush(); lev_hist.flush()
            result["history_paths"] = {"nlv_file": nlv_file, "lev_file": lev_file, "shape": (NUM_PATHS, self.days), "dates": [CURRENT_DATE + datetime.timedelta(days=t) for t in range(self.days)]}
        return result