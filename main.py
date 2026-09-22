import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter, PercentFormatter
from dateutil.relativedelta import relativedelta
from rich import box
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, NUM_PATHS, 
    MAX_MARGIN_CALL_PROBABILITY, POST_LAST_WITHDRAWAL_BUFFER_DAYS, 
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS, ACTIVE_ASSET, WITHDRAWAL_SCHEDULE, BASE_CURRENCY,
    NUM_FACTORS, MCMC_DRAWS, MCMC_TUNE, MCMC_CORES, IM_REQUIREMENTS
)
from data_fetcher import DataEngine
from calibration import BayesianFactorCalibrator
from engine import MarketSimulator
from optimizer import CRNGridOptimizer

console = Console(highlight=False)

def _safe_float(value) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")

def _fmt_base_ccy(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    return f"{x:,.{decimals}f} {BASE_CURRENCY}" if np.isfinite(x) else "n/a"

def _fmt_pct(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    return f"{x:.{decimals}%}" if np.isfinite(x) else "n/a"

def _fmt_x(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    if np.isposinf(x): return "∞x"
    return f"{x:.{decimals}f}x" if np.isfinite(x) else "n/a"

def _binomial_upper_95(p_hat: float, n: int) -> float:
    p_hat = float(p_hat)
    if not np.isfinite(p_hat) or n <= 0: return float("nan")
    se = np.sqrt(max(p_hat * (1.0 - p_hat), 1e-12) / n)
    return min(1.0, p_hat + 1.96 * se)

def print_banner(title: str, subtitle: str | None = None) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if subtitle: console.print(f"{subtitle}")
    console.print("────────────────────────────────────────────────────────────\n")

def print_table(title: str, columns: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> None:
    right_align = right_align or set()
    table = Table(title=title, box=box.SIMPLE, header_style="bold", show_lines=False, title_style="bold")
    for column in columns: table.add_column(column, justify="right" if column in right_align else "left")
    for row in rows: table.add_row(*[str(cell) for cell in row])
    console.print(table)
    console.print()

def print_initial_state(state: dict) -> float:
    assets = state["assets_dict"]
    gross_assets = sum(info["v0"] for info in assets.values())
    active_value = assets[ACTIVE_ASSET]["v0"]
    
    total_mmr = sum(info["v0"] * info["mmr"] for info in assets.values())
    nlv_pre_trade = gross_assets - CURRENT_DEBT
    el_pre_trade = nlv_pre_trade - total_mmr
    current_leverage = (gross_assets / nlv_pre_trade) if nlv_pre_trade > 0 else float("inf")
    sma_buying_power = CURRENT_SMA / float(IM_REQUIREMENTS.get(ACTIVE_ASSET, 0.50))

    rows = [
        ["Active Rebalancing Asset", f"{ACTIVE_ASSET} ({assets[ACTIVE_ASSET].get('currency', BASE_CURRENCY)})"],
        ["Gross Position Value", _fmt_base_ccy(gross_assets)],
        ["Net Liquidation Value (NLV)", _fmt_base_ccy(nlv_pre_trade)],
        ["Excess Liquidity (EL)", _fmt_base_ccy(el_pre_trade)],
        ["Reg T Special Memorandum (SMA)", _fmt_base_ccy(CURRENT_SMA)],
        ["Available SMA Buying Power", _fmt_base_ccy(sma_buying_power)],
        ["Pre-Trade Portfolio Leverage", _fmt_x(current_leverage)],
    ]
    print_table("INITIAL MARGIN ENVIRONMENT", ["Metric", "Value"], rows, {"Value"})
    return current_leverage

def print_simulation_horizon(simulator, final_date) -> None:
    rows = [
        ["Start date", str(CURRENT_DATE)],
        ["Final simulation date", str(final_date)],
        ["Total scheduled deposits", _fmt_base_ccy(float(np.sum(simulator.deposits_arr)))],
        ["Total scheduled liabilities", _fmt_base_ccy(float(np.sum(simulator.withdrawals_arr) + simulator.initial_withdrawal_amount))],
        ["Simulated paths", f"{NUM_PATHS:,}"],
    ]
    print_table("SIMULATION BOUNDARY", ["Parameter", "Configuration"], rows, {"Configuration"})

def print_execution_directive(optimal_results: dict, state: dict) -> None:
    prob_ruin = float(optimal_results["prob_ruin"])
    optimal_lev = optimal_results["optimal_target_leverage"]
    
    debt = float(CURRENT_DEBT) + sum(float(w["amount"]) for w in WITHDRAWAL_SCHEDULE if (w["date"] - CURRENT_DATE).days == 0)
    gross = sum(info["v0"] for info in state["assets_dict"].values())
    nlv = gross - debt
    
    target_gross = nlv * optimal_lev if nlv > 0.0 else gross
    target_purchase = max(0.0, target_gross - gross)
    actual_immediate_purchase = min(target_purchase, CURRENT_SMA / float(IM_REQUIREMENTS.get(ACTIVE_ASSET, 0.50)))

    deposit_needed = ((gross / optimal_lev) - nlv) if (gross / nlv if nlv > 0.0 else float("inf")) > optimal_lev and optimal_lev > 0.0 else 0.0

    upper_95 = _binomial_upper_95(prob_ruin, NUM_PATHS)
    status = "BREACH" if prob_ruin > MAX_MARGIN_CALL_PROBABILITY else "MARGINAL" if upper_95 > MAX_MARGIN_CALL_PROBABILITY else "PASS"

    rows = [
        ["Ruin Status", status],
        ["Optimal Target Leverage", _fmt_x(optimal_lev)],
        ["Risk Budget Allocation (Upper 95%)", f"{_fmt_pct(upper_95)} / {_fmt_pct(MAX_MARGIN_CALL_PROBABILITY)}"],
        ["Required Purchase (Unconstrained)", _fmt_base_ccy(target_purchase)],
        [f"Immediate Executable {ACTIVE_ASSET} Order", f"[bold]{_fmt_base_ccy(actual_immediate_purchase)}[/bold]"],
        ["Required Cash Deposit (Deleverage)", f"[bold]{_fmt_base_ccy(deposit_needed)}[/bold]"],
    ]
    print_table("EXECUTION DIRECTIVE", ["Metric", "Value"], rows, {"Value"})

def _extract_terminal_nlv(results: dict, drop_invalid: bool = True, survivors_only: bool = True) -> np.ndarray:
    nlv = np.asarray(results["Final_NLV"], dtype=np.float64)
    if survivors_only: nlv = nlv[~np.asarray(results.get("Final_Ruined", np.zeros(nlv.shape, dtype=bool)), dtype=bool)]
    return nlv[np.isfinite(nlv)] if drop_invalid else nlv

def calculate_annualized_twrr(simulator, sim_results: dict) -> float:
    survivor_nlv = _extract_terminal_nlv(sim_results, survivors_only=True)
    if len(survivor_nlv) == 0: return float("nan")
        
    v_terminal = float(np.median(survivor_nlv))
    v_0 = float(sum(float(info["v0"]) for info in simulator.assets_dict.values()) - CURRENT_DEBT - simulator.initial_withdrawal_amount)
    days = float(simulator.days)
    
    if days <= 0 or v_0 <= 0: return float("nan")
        
    net_daily_cf = simulator.deposits_arr - simulator.withdrawals_arr
    time_weights = (days - np.arange(simulator.days, dtype=np.float64)) / days
    weighted_cf = float(np.sum(net_daily_cf * time_weights))
    
    avg_cap = v_0 + weighted_cf
    period_return = (v_terminal - v_0 - float(np.sum(net_daily_cf))) / avg_cap if avg_cap > 0 else -1.0
    return ((1.0 + period_return) ** (1.0 / (days / 365.0))) - 1.0 if (1.0 + period_return) > 0 else float("nan")

def print_terminal_comparison(opt_res: dict, bench_res: dict, opt_twrr: float, bench_twrr: float) -> None:
    opt_nlv, bench_nlv = _extract_terminal_nlv(opt_res), _extract_terminal_nlv(bench_res)
    
    rows = [
        ["Median Terminal NLV", _fmt_base_ccy(float(np.median(opt_nlv))), _fmt_base_ccy(float(np.median(bench_nlv)))],
        ["Annualized TWRR", _fmt_pct(opt_twrr), _fmt_pct(bench_twrr)]
    ]
    print_table("PERFORMANCE COMPARISON (Median Survivor)", ["Metric", "Optimized", "Benchmark (1.0x)"], rows, {"Optimized", "Benchmark (1.0x)"})

# =============================================================================
# MATPLOTLIB VISUALIZATIONS
# =============================================================================

def show_visualizations(opt_res: dict, bench_res: dict):
    console.print("Generating structural diagnostic charts...")
    plt.style.use('default')
    
    # Pre-extract data to avoid keeping memory maps open
    has_history = "history_paths" in opt_res
    if has_history:
        shape = opt_res["history_paths"]["shape"]
        dates = opt_res["history_paths"]["dates"]
        num_paths, num_days = shape
        
        nlv_hist = np.memmap(opt_res["history_paths"]["nlv_file"], dtype=np.float32, mode='r', shape=shape)
        lev_hist = np.memmap(opt_res["history_paths"]["lev_file"], dtype=np.float32, mode='r', shape=shape)
        
        # Calculate percentiles vertically to save RAM
        nlv_p = np.zeros((5, num_days), dtype=np.float32)
        lev_p = np.zeros((5, num_days), dtype=np.float32)
        
        chunk_days = 50
        for t_start in range(0, num_days, chunk_days):
            t_end = min(t_start + chunk_days, num_days)
            nlv_chunk = np.array(nlv_hist[:, t_start:t_end])
            lev_chunk = np.array(lev_hist[:, t_start:t_end])
            nlv_p[:, t_start:t_end] = np.percentile(nlv_chunk, [5, 25, 50, 75, 95], axis=0)
            with np.errstate(invalid='ignore'):
                lev_p[:, t_start:t_end] = np.nanpercentile(lev_chunk, [5, 25, 50, 75, 95], axis=0)
            
        del nlv_hist
        del lev_hist
        
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(dates, nlv_p[2], color="#1E293B", linewidth=2, label="Median NLV")
        ax1.fill_between(dates, nlv_p[1], nlv_p[3], color="#475569", alpha=0.4, label="25th - 75th Percentile")
        ax1.fill_between(dates, nlv_p[0], nlv_p[4], color="#94A3B8", alpha=0.2, label="5th - 95th Percentile")
        ax1.set_title("Portfolio NLV & Dynamic Leverage Trajectories", loc="left", fontsize=12, fontweight="bold")
        ax1.set_ylabel(f"Net Liquidation Value ({BASE_CURRENCY})")
        ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        optimal_lev = opt_res["optimal_target_leverage"]
        ax2.plot(dates, lev_p[2], color="#0F172A", linewidth=2, label="Median Leverage")
        ax2.fill_between(dates, lev_p[1], lev_p[3], color="#334155", alpha=0.4)
        ax2.fill_between(dates, lev_p[0], lev_p[4], color="#64748B", alpha=0.2)
        ax2.axhline(optimal_lev, color="#B91C1C", linestyle=":", linewidth=2, label=f"Rebalancing Target ({optimal_lev:.2f}x)")
        ax2.set_ylabel("Portfolio Leverage (x)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left")
        fig1.tight_layout()

    # Risk Frontier Curve
    if "grid_leverages" in opt_res:
        fig2, ax3 = plt.subplots(figsize=(8, 5))
        levs = opt_res["grid_leverages"]
        probs = opt_res["grid_ruin_probs"]
        optimal_lev = opt_res["optimal_target_leverage"]
        
        ax3.plot(levs, probs, color="#1E293B", linewidth=2)
        ax3.axhline(MAX_MARGIN_CALL_PROBABILITY, color="#B91C1C", linestyle="--", linewidth=1.5, label=f"Risk Budget ({MAX_MARGIN_CALL_PROBABILITY:.1%})")
        
        optimal_prob = probs[np.where(np.isclose(levs, optimal_lev))[0][0]]
        ax3.scatter([optimal_lev], [optimal_prob], color="#B91C1C", s=60, zorder=5, label=f"Optimal Target ({optimal_lev:.2f}x)")
        
        ax3.set_title("Systemic Risk Profile (Ruin Frontier)", loc="left", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Target Portfolio Leverage (x)")
        ax3.set_ylabel("Probability of Margin Call")
        ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax3.set_ylim(0, min(1.0, max(0.1, optimal_prob * 3)))
        ax3.grid(alpha=0.3)
        ax3.legend(loc="upper left")
        fig2.tight_layout()

    # Terminal Distribution Histogram
    strat_nlv = _extract_terminal_nlv(opt_res, survivors_only=True)
    bench_nlv = _extract_terminal_nlv(bench_res, survivors_only=True)
    
    if len(strat_nlv) > 0 and len(bench_nlv) > 0:
        fig3, ax4 = plt.subplots(figsize=(9, 5))
        x_low, x_high = np.percentile(np.concatenate([strat_nlv, bench_nlv]), [0.5, 99.5])
        bins = np.linspace(x_low, x_high, 100)
        
        ax4.hist(bench_nlv, bins=bins, alpha=0.5, color="#94A3B8", label="Benchmark (1.0x)")
        ax4.hist(strat_nlv, bins=bins, alpha=0.5, color="#0F172A", label=f"Optimized ({opt_res.get('optimal_target_leverage', 1.0):.2f}x)")
        
        ax4.axvline(np.median(bench_nlv), color="#64748B", linestyle="--", linewidth=1.5)
        ax4.axvline(np.median(strat_nlv), color="#1E293B", linestyle="--", linewidth=1.5)
        
        ax4.set_title("Terminal Wealth Distribution (Survivors)", loc="left", fontsize=12, fontweight="bold")
        ax4.set_xlabel(f"Net Liquidation Value ({BASE_CURRENCY})")
        ax4.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax4.get_yaxis().set_visible(False)
        ax4.legend(loc="upper right")
        fig3.tight_layout()

    plt.show()

# =============================================================================
# ORCHESTRATION
# =============================================================================

def main() -> None:
    print_banner("LDI OPTIMIZATION SYSTEM", "Bayesian Factors & Dynamic Cash Flows")
    
    skip_deposit = Confirm.ask("Skip next scheduled deposit?", default=False)
    show_plots = Confirm.ask("Display interactive charts?", default=False)
    console.print()

    engine = DataEngine()
    engine.fetch_data()
    state = engine.build_current_state()
    print_initial_state(state)
    
    calibrator = BayesianFactorCalibrator(engine.data, num_factors=NUM_FACTORS)
    mcmc_params = calibrator.fit(draws=MCMC_DRAWS, tune=MCMC_TUNE, cores=MCMC_CORES)

    final_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE) + relativedelta(months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS, days=POST_LAST_WITHDRAWAL_BUFFER_DAYS)
    simulator = MarketSimulator(state, mcmc_params, final_date, skip_next_deposit=skip_deposit)
    print_simulation_horizon(simulator, final_date)

    opt_results = CRNGridOptimizer(simulator).optimize()

    console.print("Extracting benchmark (1.00x) base cases...")
    bench_results = simulator.simulate(target_leverage=1.0, store_paths=True, store_history=show_plots)
    console.print()

    print_execution_directive(opt_results, state)
    print_terminal_comparison(opt_results, bench_results, calculate_annualized_twrr(simulator, opt_results), calculate_annualized_twrr(simulator, bench_results))

    if show_plots:
        show_visualizations(opt_results, bench_results)
        
    def safe_clean(d):
        if "history_paths" in d:
            for f in [d["history_paths"]["nlv_file"], d["history_paths"]["lev_file"]]:
                try: os.remove(f) 
                except: pass
    
    safe_clean(opt_results)
    safe_clean(bench_results)

if __name__ == "__main__":
    main()