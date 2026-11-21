import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from numba import njit, prange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, DEPOSIT_SCHEDULE, DEFAULT_MONTHLY_DEPOSIT_FUTURE,
    MARGIN_INTEREST_RATE, NUM_PATHS, RNG_SEED, WITHDRAWAL_SCHEDULE, HISTORY_INTERVAL_DAYS,
    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY, BOND_REGISTRY,
    ASSET_CURRENCIES, BASE_CURRENCY, DIV_MEAN_REVERSION_SPEED, DIV_PAYOUT_DAY,
    BOND_YIELD_MR_SPEED, BOND_YIELD_FLOOR, REBALANCE_INTERVAL_DAYS
)

console = Console(highlight=False)


@njit(parallel=True, fastmath=True)
def _fused_hmm_crn_grid_chunk(
    start_idx, end_idx, base_seed, days, current_debt, current_sma, init_withdrawal, daily_rate,
    leverage_grid, num_factors, num_assets, pca_loadings, residual_scales, asset_means, asset_scales,
    P_matrix, chol_0, chol_1, init_prob_regime_1, daily_jump_prob, mu_jump, sigma_jump,
    asset_v0, asset_mmr, asset_imr, deposits, withdrawals, out_ruined_grid, asset_type,
    div_mean, div_vol, div_yield_init, is_div_day, div_weight, div_mr_speed, bond_proxy_dur,
    bond_initial_ytm, bond_cf_days, bond_cf_coupons, bond_cf_principals, bond_maturity_day,
    bond_fx_idx, bond_mr_speed, bond_yield_floor, rebalance_interval
):
    num_levs = len(leverage_grid)
    max_cfs = bond_cf_days.shape[1]
    dt = np.float32(1.0 / 365.25)

    for i in prange(start_idx, end_idx):
        np.random.seed(base_seed + i)
        current_regime = 1 if np.random.rand() < init_prob_regime_1 else 0

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
        asset_log_rets = np.zeros(num_assets, dtype=np.float32)
        asset_mults = np.ones(num_assets, dtype=np.float32)

        base_gross = np.float32(0.0)
        base_mmr = np.float32(0.0)
        base_imr = np.float32(0.0)
        for a in range(num_assets):
            if asset_type[a] != 2:
                base_gross += base_asset_values[a]
                base_mmr += base_asset_values[a] * asset_mmr[a]
                base_imr += base_asset_values[a] * asset_imr[a]

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
            total_mmr = base_mmr + purchased_active[l_idx] * asset_mmr[0]

            if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0):
                local_ruined[l_idx] = True

        # --- Temporal Loop ---
        for t in range(1, days):
            if np.all(local_ruined):
                break

            if current_regime == 0:
                if np.random.rand() > P_matrix[0, 0]:
                    current_regime = 1
            else:
                if np.random.rand() > P_matrix[1, 1]:
                    current_regime = 0

            for f in range(num_factors):
                Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors):
                    dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot

            jump_shock = (
                np.float32(mu_jump + sigma_jump * np.random.randn())
                if np.random.rand() < daily_jump_prob
                else np.float32(0.0)
            )

            base_gross = np.float32(0.0)
            base_div_cash = np.float32(0.0)
            base_div_sma_rel = np.float32(0.0)
            base_prin_cash = np.float32(0.0)
            base_prin_sma_rel = np.float32(0.0)

            # PASS 1: GENERATE MULTIPLIERS & UPDATE EQUITIES + FX TRACKERS
            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors):
                    r_factor += pca_loadings[a, f] * F[f]

                idio_shock = np.float32(residual_scales[a] * np.random.randn())

                if asset_type[a] == 0:
                    log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a] + jump_shock
                    mult = np.float32(np.exp(log_ret))
                    asset_log_rets[a] = log_ret
                    asset_mults[a] = mult
                    base_asset_values[a] *= mult

                    dy = (
                        np.float32(div_mr_speed * (div_mean[a] - div_yields[a]) * dt)
                        + np.float32(div_vol[a] * np.sqrt(max(np.float32(0.0), div_yields[a]) * dt) * np.random.randn())
                    )
                    div_yields[a] = max(np.float32(0.0), div_yields[a] + dy)

                    if is_div_day[t, a]:
                        div_amt = base_asset_values[a] * (div_yields[a] * div_weight[a])
                        base_div_cash += div_amt
                        base_div_sma_rel += div_amt

                    base_gross += base_asset_values[a]

                elif asset_type[a] == 2:
                    log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a]
                    mult = np.float32(np.exp(log_ret))
                    asset_log_rets[a] = log_ret
                    asset_mults[a] = mult
                    base_asset_values[a] *= mult

                else:
                    # Credit-sensitive bond proxies (IS3C.DE, EMB) receive demeaned factor + credit jump shock
                    asset_log_rets[a] = (r_factor + idio_shock) * asset_scales[a] + np.float32(0.5) * jump_shock

            # PASS 2: CALCULATE LOCAL BOND YIELD SHOCKS & APPLY DYNAMIC FX TRANSLATION
            for a in range(num_assets):
                if asset_type[a] == 1:
                    if t > bond_maturity_day[a]:
                        base_asset_values[a] = np.float32(0.0)
                        continue

                    yield_shock = -asset_log_rets[a] / bond_proxy_dur[a]
                    dy = np.float32(bond_mr_speed * (bond_initial_ytm[a] - bond_yields[a]) * dt) + yield_shock
                    bond_yields[a] = max(bond_yield_floor, bond_yields[a] + dy)

                    bp_local = np.float32(0.0)
                    cpn_cash_local = np.float32(0.0)
                    prin_cash_local = np.float32(0.0)

                    for c in range(max_cfs):
                        cf_day = bond_cf_days[a, c]
                        if cf_day == -1:
                            break
                        if cf_day == t:
                            cpn_cash_local += bond_cf_coupons[a, c]
                            prin_cash_local += bond_cf_principals[a, c]
                        elif cf_day > t:
                            total_cf = bond_cf_coupons[a, c] + bond_cf_principals[a, c]
                            bp_local += total_cf * np.exp(-bond_yields[a] * np.float32((cf_day - t) / 365.25))

                    fx_idx = bond_fx_idx[a]
                    live_fx = base_asset_values[fx_idx] if fx_idx != -1 else np.float32(1.0)

                    base_asset_values[a] = bp_local * live_fx
                    if cpn_cash_local > np.float32(0.0):
                        cpn_base = cpn_cash_local * live_fx
                        base_div_cash += cpn_base
                        base_div_sma_rel += cpn_base * asset_imr[a]
                    if prin_cash_local > np.float32(0.0):
                        prin_base = prin_cash_local * live_fx
                        base_prin_cash += prin_base
                        base_prin_sma_rel += prin_base * asset_imr[a]

                    base_gross += base_asset_values[a]

            base_mmr = np.float32(0.0)
            base_imr = np.float32(0.0)
            for a in range(num_assets):
                if asset_type[a] != 2:
                    base_mmr += base_asset_values[a] * asset_mmr[a]
                    base_imr += base_asset_values[a] * asset_imr[a]

            w_amt = withdrawals[t]
            d_amt = deposits[t]
            is_rebal_day = (t % rebalance_interval == 0) or (d_amt > np.float32(0.0)) or (w_amt > np.float32(0.0))

            # --- Accounting & Reinvestment Logic Across Leverage Grid ---
            for l_idx in range(num_levs):
                if local_ruined[l_idx]:
                    continue

                purchased_active[l_idx] *= asset_mults[0]

                interest_chg = debts[l_idx] * daily_rate
                debts[l_idx] += interest_chg
                smas[l_idx] -= interest_chg

                if base_prin_cash > np.float32(0.0):
                    cashes[l_idx] += base_prin_cash
                    smas[l_idx] += base_prin_sma_rel
                    prin_repay = min(cashes[l_idx], debts[l_idx])
                    cashes[l_idx] -= prin_repay
                    debts[l_idx] -= prin_repay

                total_div_cash = base_div_cash
                total_div_sma_rel = base_div_sma_rel
                if is_div_day[t, 0]:
                    div_active = purchased_active[l_idx] * (div_yields[0] * div_weight[0])
                    total_div_cash += div_active
                    total_div_sma_rel += div_active

                if total_div_cash > np.float32(0.0):
                    cashes[l_idx] += total_div_cash
                    smas[l_idx] += total_div_sma_rel

                    gross_eval = base_gross + purchased_active[l_idx] + cashes[l_idx]
                    nlv_eval = gross_eval - debts[l_idx]
                    current_lev = gross_eval / nlv_eval if nlv_eval > np.float32(0.0) else np.float32(999.0)

                    if current_lev <= leverage_grid[l_idx]:
                        cashes[l_idx] -= total_div_cash
                        purchased_active[l_idx] += total_div_cash
                        smas[l_idx] -= total_div_cash * asset_imr[0]
                    else:
                        debt_repay_div = min(cashes[l_idx], debts[l_idx])
                        cashes[l_idx] -= debt_repay_div
                        debts[l_idx] -= debt_repay_div

                if w_amt > np.float32(0.0):
                    cash_used = min(cashes[l_idx], w_amt)
                    cashes[l_idx] -= cash_used
                    debts[l_idx] += (w_amt - cash_used)
                    smas[l_idx] -= w_amt

                if d_amt > np.float32(0.0):
                    cashes[l_idx] += d_amt
                    smas[l_idx] += d_amt

                # If over-leveraged, deposit pays down margin debt and purchase below is 0.
                # If under-leveraged, is_rebal_day below invests the deposit + extra debt to hit target_leverage.
                debt_repay = min(cashes[l_idx], debts[l_idx])
                cashes[l_idx] -= debt_repay
                debts[l_idx] -= debt_repay

                if is_rebal_day:
                    invested_gross = base_gross + purchased_active[l_idx]
                    nlv = invested_gross + cashes[l_idx] - debts[l_idx]

                    target_gross = nlv * leverage_grid[l_idx] if nlv > np.float32(0.0) else invested_gross
                    purchase = max(np.float32(0.0), target_gross - invested_gross)
                    if purchase > np.float32(0.0):
                        cash_used = min(cashes[l_idx], purchase)
                        cashes[l_idx] -= cash_used
                        debts[l_idx] += (purchase - cash_used)
                        purchased_active[l_idx] += purchase
                        smas[l_idx] -= (purchase * asset_imr[0])

                gross = base_gross + purchased_active[l_idx] + cashes[l_idx]
                nlv = gross - debts[l_idx]
                total_mmr = base_mmr + purchased_active[l_idx] * asset_mmr[0]
                total_imr = base_imr + purchased_active[l_idx] * asset_imr[0]

                smas[l_idx] = max(smas[l_idx], nlv - total_imr)
                if (nlv - total_mmr) < np.float32(0.0) or smas[l_idx] < np.float32(0.0):
                    local_ruined[l_idx] = True

        out_ruined_grid[i, :] = local_ruined


@njit(parallel=True, fastmath=True)
def _fused_hmm_simulation_chunk(
    start_idx, end_idx, base_seed, days, current_debt, current_sma, init_withdrawal, daily_rate,
    target_leverage, num_factors, num_assets, pca_loadings, residual_scales, asset_means, asset_scales,
    P_matrix, chol_0, chol_1, init_prob_regime_1, daily_jump_prob, mu_jump, sigma_jump,
    asset_v0, asset_mmr, asset_imr, deposits, withdrawals, out_ruined, out_final_nlv,
    out_final_gross, out_final_debt, out_final_cash, store_history, history_interval,
    out_nlv_hist, out_lev_hist, asset_type, div_mean, div_vol, div_yield_init, is_div_day,
    div_weight, div_mr_speed, bond_proxy_dur, bond_initial_ytm, bond_cf_days, bond_cf_coupons,
    bond_cf_principals, bond_maturity_day, bond_fx_idx, bond_mr_speed, bond_yield_floor,
    rebalance_interval
):
    max_cfs = bond_cf_days.shape[1]
    num_hist_cols = out_nlv_hist.shape[1] if store_history else 0
    dt = np.float32(1.0 / 365.25)

    for i in prange(start_idx, end_idx):
        np.random.seed(base_seed + i)
        current_regime = 1 if np.random.rand() < init_prob_regime_1 else 0

        cash, debt, sma = np.float32(0.0), np.float32(current_debt), np.float32(current_sma)
        base_asset_values = asset_v0.copy()
        div_yields = div_yield_init.copy()
        bond_yields = bond_initial_ytm.copy()
        purchased_active = np.float32(0.0)

        Z = np.zeros(num_factors, dtype=np.float32)
        F = np.zeros(num_factors, dtype=np.float32)
        asset_log_rets = np.zeros(num_assets, dtype=np.float32)
        asset_mults = np.ones(num_assets, dtype=np.float32)

        base_gross = np.float32(0.0)
        base_mmr = np.float32(0.0)
        base_imr = np.float32(0.0)
        for a in range(num_assets):
            if asset_type[a] != 2:
                base_gross += base_asset_values[a]
                base_mmr += base_asset_values[a] * asset_mmr[a]
                base_imr += base_asset_values[a] * asset_imr[a]

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
        total_mmr = base_mmr + purchased_active * asset_mmr[0]

        hist_idx = 0
        if store_history:
            out_nlv_hist[i, hist_idx] = nlv
            out_lev_hist[i, hist_idx] = gross / nlv if nlv > np.float32(0.0) else np.nan
            hist_idx += 1

        if (nlv - total_mmr) < np.float32(0.0) or sma < np.float32(0.0):
            out_ruined[i] = True
            if store_history:
                for h in range(hist_idx, num_hist_cols):
                    out_nlv_hist[i, h] = np.nan
                    out_lev_hist[i, h] = np.nan
            continue

        for t in range(1, days):
            if current_regime == 0:
                if np.random.rand() > P_matrix[0, 0]:
                    current_regime = 1
            else:
                if np.random.rand() > P_matrix[1, 1]:
                    current_regime = 0

            for f in range(num_factors):
                Z[f] = np.random.randn()
            for r in range(num_factors):
                dot = np.float32(0.0)
                for c in range(num_factors):
                    dot += (chol_0[r, c] if current_regime == 0 else chol_1[r, c]) * Z[c]
                F[r] = dot

            jump_shock = (
                np.float32(mu_jump + sigma_jump * np.random.randn())
                if np.random.rand() < daily_jump_prob
                else np.float32(0.0)
            )

            base_gross = np.float32(0.0)
            base_div_cash = np.float32(0.0)
            base_div_sma_rel = np.float32(0.0)
            base_prin_cash = np.float32(0.0)
            base_prin_sma_rel = np.float32(0.0)

            for a in range(num_assets):
                r_factor = np.float32(0.0)
                for f in range(num_factors):
                    r_factor += pca_loadings[a, f] * F[f]
                idio_shock = np.float32(residual_scales[a] * np.random.randn())

                if asset_type[a] == 0:
                    log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a] + jump_shock
                    mult = np.float32(np.exp(log_ret))
                    asset_log_rets[a] = log_ret
                    asset_mults[a] = mult
                    base_asset_values[a] *= mult

                    dy = (
                        np.float32(div_mr_speed * (div_mean[a] - div_yields[a]) * dt)
                        + np.float32(div_vol[a] * np.sqrt(max(np.float32(0.0), div_yields[a]) * dt) * np.random.randn())
                    )
                    div_yields[a] = max(np.float32(0.0), div_yields[a] + dy)
                    if is_div_day[t, a]:
                        div_amt = base_asset_values[a] * (div_yields[a] * div_weight[a])
                        base_div_cash += div_amt
                        base_div_sma_rel += div_amt

                    base_gross += base_asset_values[a]

                elif asset_type[a] == 2:
                    log_ret = (r_factor + idio_shock) * asset_scales[a] + asset_means[a]
                    mult = np.float32(np.exp(log_ret))
                    asset_log_rets[a] = log_ret
                    asset_mults[a] = mult
                    base_asset_values[a] *= mult

                else:
                    asset_log_rets[a] = (r_factor + idio_shock) * asset_scales[a] + np.float32(0.5) * jump_shock

            for a in range(num_assets):
                if asset_type[a] == 1:
                    if t > bond_maturity_day[a]:
                        base_asset_values[a] = np.float32(0.0)
                        continue

                    yield_shock = -asset_log_rets[a] / bond_proxy_dur[a]
                    dy = np.float32(bond_mr_speed * (bond_initial_ytm[a] - bond_yields[a]) * dt) + yield_shock
                    bond_yields[a] = max(bond_yield_floor, bond_yields[a] + dy)

                    bp_local = np.float32(0.0)
                    cpn_cash_local = np.float32(0.0)
                    prin_cash_local = np.float32(0.0)

                    for c in range(max_cfs):
                        cf_day = bond_cf_days[a, c]
                        if cf_day == -1:
                            break
                        if cf_day == t:
                            cpn_cash_local += bond_cf_coupons[a, c]
                            prin_cash_local += bond_cf_principals[a, c]
                        elif cf_day > t:
                            total_cf = bond_cf_coupons[a, c] + bond_cf_principals[a, c]
                            bp_local += total_cf * np.exp(-bond_yields[a] * np.float32((cf_day - t) / 365.25))

                    fx_idx = bond_fx_idx[a]
                    live_fx = base_asset_values[fx_idx] if fx_idx != -1 else np.float32(1.0)

                    base_asset_values[a] = bp_local * live_fx
                    if cpn_cash_local > np.float32(0.0):
                        cpn_base = cpn_cash_local * live_fx
                        base_div_cash += cpn_base
                        base_div_sma_rel += cpn_base * asset_imr[a]
                    if prin_cash_local > np.float32(0.0):
                        prin_base = prin_cash_local * live_fx
                        base_prin_cash += prin_base
                        base_prin_sma_rel += prin_base * asset_imr[a]

                    base_gross += base_asset_values[a]

            base_mmr = np.float32(0.0)
            base_imr = np.float32(0.0)
            for a in range(num_assets):
                if asset_type[a] != 2:
                    base_mmr += base_asset_values[a] * asset_mmr[a]
                    base_imr += base_asset_values[a] * asset_imr[a]

            purchased_active *= asset_mults[0]

            interest_chg = debt * daily_rate
            debt += interest_chg
            sma -= interest_chg

            if base_prin_cash > np.float32(0.0):
                cash += base_prin_cash
                sma += base_prin_sma_rel
                prin_repay = min(cash, debt)
                cash -= prin_repay
                debt -= prin_repay

            total_div_cash = base_div_cash
            total_div_sma_rel = base_div_sma_rel
            if is_div_day[t, 0]:
                div_active = purchased_active * (div_yields[0] * div_weight[0])
                total_div_cash += div_active
                total_div_sma_rel += div_active

            if total_div_cash > np.float32(0.0):
                cash += total_div_cash
                sma += total_div_sma_rel

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

            is_rebal_day = (t % rebalance_interval == 0) or (d_amt > np.float32(0.0)) or (w_amt > np.float32(0.0))
            if is_rebal_day:
                invested_gross = base_gross + purchased_active
                nlv = invested_gross + cash - debt

                target_gross = nlv * target_leverage if nlv > np.float32(0.0) else invested_gross
                purchase = max(np.float32(0.0), target_gross - invested_gross)
                if purchase > np.float32(0.0):
                    cash_used = min(cash, purchase)
                    cash -= cash_used
                    debt += (purchase - cash_used)
                    purchased_active += purchase
                    sma -= (purchase * asset_imr[0])

            gross = base_gross + purchased_active + cash
            nlv = gross - debt
            total_mmr = base_mmr + purchased_active * asset_mmr[0]
            total_imr = base_imr + purchased_active * asset_imr[0]

            sma = max(sma, nlv - total_imr)
            if (nlv - total_mmr) < np.float32(0.0) or sma < np.float32(0.0):
                out_ruined[i] = True
                if store_history:
                    for h in range(hist_idx, num_hist_cols):
                        out_nlv_hist[i, h] = np.nan
                        out_lev_hist[i, h] = np.nan
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
        if self.days <= 1:
            raise ValueError(f"Simulation end_date ({end_date}) must be after CURRENT_DATE ({CURRENT_DATE}).")

        self.portfolio_order = self.state["portfolio_order"]
        self.assets_dict = self.state["assets_dict"]

        self.initial_withdrawal_amount = 0.0
        self.withdrawals_arr = np.zeros(self.days, dtype=np.float32)
        for w in WITHDRAWAL_SCHEDULE:
            day = (w["date"] - CURRENT_DATE).days
            if day == 0:
                self.initial_withdrawal_amount += float(w["amount"])
            elif 0 < day < self.days:
                self.withdrawals_arr[day] += float(w["amount"])

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
        self.num_factors = len(self.params["chol_0"])
        self.num_assets = len(self.portfolio_order)
        name_to_idx = {name: i for i, name in enumerate(self.params["asset_names"])}
        idx_mapping = [name_to_idx[name] for name in self.portfolio_order]

        self.pca_loadings = self.params["pca_loadings"][idx_mapping]
        self.residual_scales = self.params["residual_scales"][idx_mapping]
        self.asset_means = self.params["asset_means"][idx_mapping]
        self.asset_scales = self.params["asset_scales"][idx_mapping]
        self.P_matrix = self.params["P"]
        self.chol_0 = self.params["chol_0"]
        self.chol_1 = self.params["chol_1"]
        self.init_prob_regime_1 = np.float32(self.params.get("init_prob_regime_1", 0.0))

        self.daily_jump_prob = np.float32(JUMP_FREQUENCY_PER_YEAR / 365.25)
        self.mu_jump = np.float32(JUMP_MEAN_SIZE)
        self.sigma_jump = np.float32(JUMP_VOLATILITY)

        self.asset_v0 = np.array(
            [
                self.assets_dict[t]["fx_rate"] if self.assets_dict[t].get("is_fx_tracker", False) else self.assets_dict[t]["v0"]
                for t in self.portfolio_order
            ],
            dtype=np.float32
        )
        self.asset_mmr = np.array([self.assets_dict[t]["mmr"] for t in self.portfolio_order], dtype=np.float32)
        self.asset_imr = np.array([self.assets_dict[t]["imr"] for t in self.portfolio_order], dtype=np.float32)

        self.asset_type = np.zeros(self.num_assets, dtype=np.int32)
        self.div_mean = np.zeros(self.num_assets, dtype=np.float32)
        self.div_vol = np.zeros(self.num_assets, dtype=np.float32)
        self.div_yield_init = np.zeros(self.num_assets, dtype=np.float32)
        self.div_weight = np.zeros(self.num_assets, dtype=np.float32)
        self.is_div_day = np.zeros((self.days, self.num_assets), dtype=np.bool_)

        bond_schedules = {}
        max_cfs = 1
        for ticker in self.portfolio_order:
            if ticker in BOND_REGISTRY:
                b_info = BOND_REGISTRY[ticker]
                freq = int(b_info.get("coupon_frequency", 1))
                months_step = 12 // freq
                mat_date = b_info["maturity_date"]
                cfs = []
                step_idx = 0
                while True:
                    cf_date = mat_date - relativedelta(months=step_idx * months_step)
                    if cf_date <= CURRENT_DATE:
                        break
                    cfs.append(cf_date)
                    step_idx += 1
                cfs.sort()
                bond_schedules[ticker] = cfs
                max_cfs = max(max_cfs, len(cfs))

        self.bond_duration = np.zeros(self.num_assets, dtype=np.float32)
        self.bond_convexity = np.zeros(self.num_assets, dtype=np.float32)
        self.bond_proxy_dur = np.ones(self.num_assets, dtype=np.float32)
        self.bond_initial_ytm = np.zeros(self.num_assets, dtype=np.float32)
        self.bond_cf_days = np.full((self.num_assets, max_cfs), -1, dtype=np.int32)
        self.bond_cf_coupons = np.zeros((self.num_assets, max_cfs), dtype=np.float32)
        self.bond_cf_principals = np.zeros((self.num_assets, max_cfs), dtype=np.float32)
        self.bond_maturity_day = np.full(self.num_assets, -1, dtype=np.int32)
        self.bond_fx_idx = np.full(self.num_assets, -1, dtype=np.int32)

        for a, ticker in enumerate(self.portfolio_order):
            if ticker.startswith("FX_"):
                self.asset_type[a] = 2

        for a, ticker in enumerate(self.portfolio_order):
            info = self.assets_dict[ticker]
            if ticker in BOND_REGISTRY:
                self.asset_type[a] = 1
                b_info = BOND_REGISTRY[ticker]

                ccy = b_info.get("currency", ASSET_CURRENCIES[ticker])
                if ccy != BASE_CURRENCY:
                    self.bond_fx_idx[a] = self.portfolio_order.index(f"FX_{ccy}")

                freq = int(b_info.get("coupon_frequency", 1))
                coupon_payment = float(b_info["coupon_rate"]) / freq
                face_local = float(b_info["face_value_local"])
                total_face_local = float(info["quantity"]) * face_local
                target_price_local = float(info["current_price_local"])

                cfs = bond_schedules[ticker]
                days_arr = []
                amts_1b_arr = []

                for idx, cf_d in enumerate(cfs):
                    days_out = (cf_d - CURRENT_DATE).days
                    self.bond_cf_days[a, idx] = days_out
                    is_maturity = (cf_d == b_info["maturity_date"])

                    self.bond_cf_coupons[a, idx] = total_face_local * coupon_payment
                    self.bond_cf_principals[a, idx] = total_face_local if is_maturity else 0.0
                    if is_maturity:
                        self.bond_maturity_day[a] = days_out

                    days_arr.append(days_out)
                    cf_1b_local = face_local * (1.0 + coupon_payment) if is_maturity else face_local * coupon_payment
                    amts_1b_arr.append(cf_1b_local)

                days_np = np.array(days_arr, dtype=np.float64)
                amts_np = np.array(amts_1b_arr, dtype=np.float64)

                low, high = -0.50, 2.00
                for _ in range(100):
                    mid = 0.5 * (low + high)
                    pv = np.sum(amts_np * np.exp(-mid * (days_np / 365.25)))
                    if pv > target_price_local:
                        low = mid
                    else:
                        high = mid

                y = float(mid)
                self.bond_initial_ytm[a] = y
                t_years = days_np / 365.25
                pv_cfs = amts_np * np.exp(-y * t_years)
                total_pv = float(np.sum(pv_cfs))

                self.bond_duration[a] = float(np.sum(t_years * pv_cfs) / total_pv)
                self.bond_convexity[a] = float(np.sum((t_years ** 2) * pv_cfs) / total_pv)
                self.bond_proxy_dur[a] = float(b_info.get("proxy_duration", self.bond_duration[a]))

            elif self.asset_type[a] == 0:
                self.div_mean[a] = float(info.get("mean_yield", 0.0))
                self.div_vol[a] = float(info.get("yield_vol", 0.0))
                self.div_yield_init[a] = float(info.get("ttm_yield", 0.0))
                months = info.get("payout_months", [])
                if months:
                    self.div_weight[a] = 1.0 / len(months)
                    for t in range(self.days):
                        d = CURRENT_DATE + datetime.timedelta(days=t)
                        if d.month in months and d.day == DIV_PAYOUT_DAY:
                            self.is_div_day[t, a] = True

    def simulate_grid(self, leverage_grid: np.ndarray, desc: str = "Simulating Strategy Tiers...") -> np.ndarray:
        num_levs = len(leverage_grid)
        ruined_grid = np.zeros((NUM_PATHS, num_levs), dtype=np.bool_)

        with Progress(
            TextColumn("[dim]{task.description}[/dim]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(desc, total=NUM_PATHS)
            for start_idx in range(0, NUM_PATHS, 10000):
                end_idx = min(start_idx + 10000, NUM_PATHS)
                _fused_hmm_crn_grid_chunk(
                    start_idx, end_idx, int(RNG_SEED), self.days, np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.25),
                    np.asarray(leverage_grid, dtype=np.float32), self.num_factors, self.num_assets,
                    self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1, self.init_prob_regime_1,
                    self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr, self.deposits_arr, self.withdrawals_arr,
                    ruined_grid, self.asset_type, self.div_mean, self.div_vol, self.div_yield_init,
                    self.is_div_day, self.div_weight, np.float32(DIV_MEAN_REVERSION_SPEED),
                    self.bond_proxy_dur, self.bond_initial_ytm, self.bond_cf_days, self.bond_cf_coupons,
                    self.bond_cf_principals, self.bond_maturity_day, self.bond_fx_idx,
                    np.float32(BOND_YIELD_MR_SPEED), np.float32(BOND_YIELD_FLOOR), int(REBALANCE_INTERVAL_DAYS)
                )
                progress.advance(task, advance=(end_idx - start_idx))
        return np.mean(ruined_grid, axis=0)

    def simulate(self, target_leverage: float, store_paths: bool = True, store_history: bool = False) -> dict:
        ruined = np.zeros(NUM_PATHS, dtype=np.bool_)
        final_nlv = np.zeros(NUM_PATHS, dtype=np.float32)
        final_gross = np.zeros(NUM_PATHS, dtype=np.float32)
        final_debt = np.zeros(NUM_PATHS, dtype=np.float32)
        final_cash = np.zeros(NUM_PATHS, dtype=np.float32)

        history_interval = max(1, int(HISTORY_INTERVAL_DAYS))
        if store_history:
            num_hist_points = ((self.days - 1) // history_interval) + 1
            nlv_file = f"temp_nlv_hist_{target_leverage:.3f}.dat"
            lev_file = f"temp_lev_hist_{target_leverage:.3f}.dat"
            nlv_hist = np.memmap(nlv_file, dtype=np.float32, mode="w+", shape=(NUM_PATHS, num_hist_points))
            lev_hist = np.memmap(lev_file, dtype=np.float32, mode="w+", shape=(NUM_PATHS, num_hist_points))
        else:
            nlv_hist = np.empty((0, 0), dtype=np.float32)
            lev_hist = np.empty((0, 0), dtype=np.float32)

        with Progress(
            TextColumn("[dim]{task.description}[/dim]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Extracting Path Details ({target_leverage:.3f}x)...", total=NUM_PATHS)
            for start_idx in range(0, NUM_PATHS, 10000):
                end_idx = min(start_idx + 10000, NUM_PATHS)
                _fused_hmm_simulation_chunk(
                    start_idx, end_idx, int(RNG_SEED), self.days, np.float32(CURRENT_DEBT), np.float32(CURRENT_SMA),
                    np.float32(self.initial_withdrawal_amount), np.float32(MARGIN_INTEREST_RATE / 365.25),
                    np.float32(target_leverage), self.num_factors, self.num_assets,
                    self.pca_loadings, self.residual_scales, self.asset_means, self.asset_scales,
                    self.P_matrix, self.chol_0, self.chol_1, self.init_prob_regime_1,
                    self.daily_jump_prob, self.mu_jump, self.sigma_jump,
                    self.asset_v0, self.asset_mmr, self.asset_imr, self.deposits_arr, self.withdrawals_arr,
                    ruined, final_nlv, final_gross, final_debt, final_cash, store_history, history_interval,
                    nlv_hist, lev_hist, self.asset_type, self.div_mean, self.div_vol, self.div_yield_init,
                    self.is_div_day, self.div_weight, np.float32(DIV_MEAN_REVERSION_SPEED),
                    self.bond_proxy_dur, self.bond_initial_ytm, self.bond_cf_days, self.bond_cf_coupons,
                    self.bond_cf_principals, self.bond_maturity_day, self.bond_fx_idx,
                    np.float32(BOND_YIELD_MR_SPEED), np.float32(BOND_YIELD_FLOOR), int(REBALANCE_INTERVAL_DAYS)
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
                "Final_Ruined": ruined
            })
        if store_history:
            nlv_hist.flush()
            lev_hist.flush()
            hist_dates = [CURRENT_DATE + datetime.timedelta(days=t) for t in range(0, self.days, history_interval)]
            result["history_paths"] = {
                "nlv_file": nlv_file,
                "lev_file": lev_file,
                "shape": (NUM_PATHS, num_hist_points),
                "dates": hist_dates
            }
        return result