import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from dateutil.relativedelta import relativedelta
from rich import box
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt

# --- Shared Config & Parameters ---
from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, NUM_PATHS, 
    MAX_MARGIN_CALL_PROBABILITY, POST_LAST_WITHDRAWAL_BUFFER_DAYS, 
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS, ACTIVE_ASSET, WITHDRAWAL_SCHEDULE, BASE_CURRENCY,
    NUM_FACTORS, MCMC_DRAWS, MCMC_TUNE, MCMC_CORES, IM_REQUIREMENTS
)

# --- Dynamic Bayesian Method Modules ---
from data_fetcher import DataEngine as DynamicDataEngine
from calibration import BayesianFactorCalibrator
from engine import MarketSimulator as DynamicMarketSimulator
from optimizer import CRNGridOptimizer as DynamicCRNGridOptimizer

# --- Static Method Modules (Copied from Archive) ---
from data_fetcher_static_method import DataEngine as StaticDataEngine
from engine_static_method import MarketSimulator as StaticMarketSimulator
from optimizer_static_method import CRNGridOptimizer as StaticCRNGridOptimizer

console = Console()

# =============================================================================
# FORMATTING & REPORTING HELPERS
# =============================================================================

def _safe_float(value) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")

def _fmt_base_ccy(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    if not np.isfinite(x): return "n/a"
    return f"{x:,.{decimals}f} {BASE_CURRENCY}"

def _fmt_pct(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    if not np.isfinite(x): return "n/a"
    return f"{x:.{decimals}%}"

def _fmt_x(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    if np.isposinf(x): return "∞x"
    if not np.isfinite(x): return "n/a"
    return f"{x:.{decimals}f}x"

def _binomial_upper_95(p_hat: float, n: int) -> float:
    p_hat = float(p_hat)
    if not np.isfinite(p_hat) or n <= 0: return float("nan")
    se = np.sqrt(max(p_hat * (1.0 - p_hat), 1e-12) / n)
    return min(1.0, p_hat + 1.96 * se)

def _risk_status(prob_ruin: float) -> str:
    upper_95 = _binomial_upper_95(prob_ruin, NUM_PATHS)
    if prob_ruin > MAX_MARGIN_CALL_PROBABILITY:
        return "[red]BREACH[/red]"
    if upper_95 > MAX_MARGIN_CALL_PROBABILITY:
        return "[yellow]MARGINAL (Passes but close to budget)[/yellow]"
    return "[green]PASS[/green]"

def print_banner(title: str, subtitle: str | None = None) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if subtitle: console.print(f"[dim]{subtitle}[/dim]")
    console.print("─" * 65 + "\n")

def print_table(title: str, columns: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> None:
    right_align = right_align or set()
    table = Table(title=title, box=box.SIMPLE, header_style="bold", show_lines=False, title_style="bold")
    for column in columns: 
        table.add_column(column, justify="right" if column in right_align else "left", overflow="fold")
    for row in rows: 
        table.add_row(*[str(cell) for cell in row])
    console.print(table)
    console.print()

def print_initial_state(state: dict) -> float:
    assets = state["assets_dict"]
    gross_assets = sum(info["v0"] for info in assets.values())
    active_value = assets[ACTIVE_ASSET]["v0"]
    passive_value = gross_assets - active_value
    
    total_mmr = sum(info["v0"] * info["mmr"] for info in assets.values())
    nlv_pre_trade = gross_assets - CURRENT_DEBT
    el_pre_trade = nlv_pre_trade - total_mmr
    current_leverage = (gross_assets / nlv_pre_trade) if nlv_pre_trade > 0 else float("inf")
    
    active_imr = float(IM_REQUIREMENTS.get(ACTIVE_ASSET, 0.50))
    sma_buying_power = CURRENT_SMA / active_imr

    balance_rows = [
        ["Active Rebalancing Asset", f"{ACTIVE_ASSET} ({assets[ACTIVE_ASSET]['currency']})"],
        ["Active Asset Current Value", _fmt_base_ccy(active_value)],
        ["Passive Portfolio Value", _fmt_base_ccy(passive_value)],
        ["Gross Position Value", _fmt_base_ccy(gross_assets)],
        ["Current Margin Debt", _fmt_base_ccy(CURRENT_DEBT)],
        ["Net Liquidation Value (NLV)", _fmt_base_ccy(nlv_pre_trade)],
        ["Excess Liquidity (EL)", _fmt_base_ccy(el_pre_trade)],
        ["Reg T Special Memorandum (SMA)", _fmt_base_ccy(CURRENT_SMA)],
        ["Available SMA Buying Power", _fmt_base_ccy(sma_buying_power)],
        ["Pre-Trade Portfolio Leverage", _fmt_x(current_leverage)],
    ]
    print_table("Initial Margin Sheet", ["Metric", "Value"], balance_rows, {"Value"})
    return current_leverage

def print_simulation_horizon(simulator, final_date, last_withdrawal_date) -> None:
    total_future_deposits = float(np.sum(simulator.deposits_arr))
    total_withdrawals = float(np.sum(simulator.withdrawals_arr) + simulator.initial_withdrawal_amount)
    deposit_days_count = int(np.count_nonzero(simulator.deposits_arr))

    horizon_rows = [
        ["Start date", str(CURRENT_DATE)],
        ["Final simulation date", str(final_date)],
        ["Calendar days", f"{simulator.days:,}"],
        ["Future deposit events", f"{deposit_days_count:,}"],
        ["Total scheduled deposits", _fmt_base_ccy(total_future_deposits)],
        ["Total scheduled liabilities", _fmt_base_ccy(total_withdrawals)],
        ["Simulated paths", f"{NUM_PATHS:,}"],
    ]
    print_table("Simulation Horizon", ["Metric", "Value"], horizon_rows, {"Value"})

def print_execution_directive(optimal_results: dict, state: dict) -> None:
    prob_ruin = float(optimal_results["prob_ruin"])
    upper_95_ruin = _binomial_upper_95(prob_ruin, NUM_PATHS)
    optimal_lev = optimal_results["optimal_target_leverage"]
    
    cash = 0.0
    debt = float(CURRENT_DEBT)
    init_withdrawal = sum(float(w["amount"]) for w in WITHDRAWAL_SCHEDULE if (w["date"] - CURRENT_DATE).days == 0)
    if init_withdrawal > 0.0:
        debt += init_withdrawal
    
    gross_assets = sum(info["v0"] for info in state["assets_dict"].values())
    gross = gross_assets + cash
    nlv = gross - debt
    
    target_gross = nlv * optimal_lev if nlv > 0.0 else gross
    target_purchase = max(0.0, target_gross - gross)
    
    active_imr = float(IM_REQUIREMENTS.get(ACTIVE_ASSET, 0.50))
    max_executable_purchase = CURRENT_SMA / active_imr
    actual_immediate_purchase = min(target_purchase, max_executable_purchase)

    current_leverage = gross / nlv if nlv > 0.0 else float("inf")
    deposit_needed = 0.0
    if current_leverage > optimal_lev and optimal_lev > 0.0:
        deposit_needed = (gross / optimal_lev) - nlv

    rows = [
        ["Ruin Status", _risk_status(prob_ruin)],
        ["Optimal Target Leverage", _fmt_x(optimal_lev)],
        ["Required Purchase (to hit target)", _fmt_base_ccy(target_purchase)],
        ["Max Executable Today (SMA limit)", _fmt_base_ccy(max_executable_purchase)],
        [f"Immediate Executable {ACTIVE_ASSET} Order", f"[bold green]{_fmt_base_ccy(actual_immediate_purchase)}[/bold green]"],
        ["Required Cash Deposit (to deleverage)", f"[bold red]{_fmt_base_ccy(deposit_needed)}[/bold red]"],
        ["Dual-Constraint Ruin Prob.", _fmt_pct(prob_ruin)],
        ["Breach Upper 95% Bound", _fmt_pct(upper_95_ruin)],
        ["Risk Budget Limit", _fmt_pct(MAX_MARGIN_CALL_PROBABILITY)],
    ]
    print_table("Execution Directive", ["Metric", "Value"], rows, {"Value"})

def _extract_terminal_nlv(results: dict, drop_invalid: bool = True, survivors_only: bool = True) -> np.ndarray:
    nlv = np.asarray(results["Final_NLV"], dtype=np.float64)
    if survivors_only:
        ruined = np.asarray(results.get("Final_Ruined", np.zeros(nlv.shape, dtype=bool)), dtype=bool)
        nlv = nlv[~ruined]
    if drop_invalid: nlv = nlv[np.isfinite(nlv)]
    return nlv

def _nlv_summary(nlv: np.ndarray) -> dict:
    if len(nlv) == 0: return {"mean": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(nlv)), 
        "p05": float(np.percentile(nlv, 5)), 
        "median": float(np.percentile(nlv, 50)), 
        "p95": float(np.percentile(nlv, 95))
    }

def calculate_annualized_twrr(simulator, sim_results: dict) -> float:
    """
    Computes the standard Modified Dietz Time-Weighted Rate of Return (TWRR)
    neutralizing the timing and size of external deposits and scheduled liabilities.
    """
    survivor_nlv = _extract_terminal_nlv(sim_results, survivors_only=True)
    if len(survivor_nlv) == 0:
        return float("nan")
        
    v_terminal = float(np.median(survivor_nlv))
    
    # Starting Net Liquidation Value (after Day 0 rebalancing and initial withdrawal)
    gross_v0 = sum(float(info["v0"]) for info in simulator.assets_dict.values())
    v_0 = float(gross_v0 - CURRENT_DEBT - simulator.initial_withdrawal_amount)
    
    days = float(simulator.days)
    if days <= 0 or v_0 <= 0:
        return float("nan")
        
    # Net daily cash flows: Deposits (+) and Withdrawals (-)
    net_daily_cf = simulator.deposits_arr - simulator.withdrawals_arr
    total_net_cf = float(np.sum(net_daily_cf))
    
    # Calculate time-weighted cash flow contribution
    day_indices = np.arange(simulator.days, dtype=np.float64)
    time_weights = (days - day_indices) / days
    weighted_cf = float(np.sum(net_daily_cf * time_weights))
    
    average_capital_base = v_0 + weighted_cf
    net_investment_gain = v_terminal - v_0 - total_net_cf
    
    if average_capital_base <= 0:
        return float("nan")
        
    period_return = net_investment_gain / average_capital_base
    if (1.0 + period_return) <= 0:
        return float("nan")
        
    years = days / 365.0
    annualized_twrr = ((1.0 + period_return) ** (1.0 / years)) - 1.0
    
    return float(annualized_twrr)

def print_terminal_nlv_comparison(strategy_results: dict, benchmark_results: dict, strategy_twrr: float, benchmark_twrr: float) -> None:
    strategy_nlv = _extract_terminal_nlv(strategy_results, survivors_only=True)
    benchmark_nlv = _extract_terminal_nlv(benchmark_results, survivors_only=True)
    
    strategy_stats, benchmark_stats = _nlv_summary(strategy_nlv), _nlv_summary(benchmark_nlv)
    twrr_uplift = strategy_twrr - benchmark_twrr if not np.isnan(strategy_twrr) and not np.isnan(benchmark_twrr) else float('nan')

    rows = [
        ["Mean Terminal NLV", _fmt_base_ccy(strategy_stats["mean"]), _fmt_base_ccy(benchmark_stats["mean"]), _fmt_base_ccy(strategy_stats["mean"] - benchmark_stats["mean"])],
        ["Median Terminal NLV", _fmt_base_ccy(strategy_stats["median"]), _fmt_base_ccy(benchmark_stats["median"]), _fmt_base_ccy(strategy_stats["median"] - benchmark_stats["median"])],
        ["SMA/EL Ruin Prob.", _fmt_pct(strategy_results["prob_ruin"]), _fmt_pct(benchmark_results["prob_ruin"]), "-"],
        ["Annualized TWRR (Median)", _fmt_pct(strategy_twrr), _fmt_pct(benchmark_twrr), _fmt_pct(twrr_uplift)]
    ]
    print_table("Terminal NLV & Performance Comparison (Survivors)", ["Metric", "Optimized Rebalancing", "Benchmark (1.0x)", "Uplift"], rows, {"Optimized Rebalancing", "Benchmark (1.0x)", "Uplift"})

# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def plot_time_series_bands(sim_results: dict) -> None:
    if "history_paths" not in sim_results: return
    hist_data = sim_results["history_paths"]
    dates = hist_data["dates"]
    shape = hist_data["shape"]
    
    nlv_file, lev_file = hist_data["nlv_file"], hist_data["lev_file"]
    nlv_hist = np.memmap(nlv_file, dtype=np.float32, mode='r', shape=shape)
    lev_hist = np.memmap(lev_file, dtype=np.float32, mode='r', shape=shape)
    
    num_paths, num_days = shape
    nlv_p = np.zeros((5, num_days), dtype=np.float32)
    lev_p = np.zeros((5, num_days), dtype=np.float32)
    
    temp_nlv_t = nlv_file.replace('.dat', '_transposed.dat')
    temp_lev_t = lev_file.replace('.dat', '_transposed.dat')
    nlv_transposed = np.memmap(temp_nlv_t, dtype=np.float32, mode='w+', shape=(num_days, num_paths))
    lev_transposed = np.memmap(temp_lev_t, dtype=np.float32, mode='w+', shape=(num_days, num_paths))
    
    chunk_size = 25000 
    for start_idx in range(0, num_paths, chunk_size):
        end_idx = min(start_idx + chunk_size, num_paths)
        nlv_transposed[:, start_idx:end_idx] = nlv_hist[start_idx:end_idx, :].T
        lev_transposed[:, start_idx:end_idx] = lev_hist[start_idx:end_idx, :].T
            
    chunk_days = 25  
    for t_start in range(0, num_days, chunk_days):
        t_end = min(t_start + chunk_days, num_days)
        nlv_chunk = np.array(nlv_transposed[t_start:t_end, :])
        lev_chunk = np.array(lev_transposed[t_start:t_end, :])
        nlv_p[:, t_start:t_end] = np.percentile(nlv_chunk, [5, 25, 50, 75, 95], axis=1)
        with np.errstate(invalid='ignore'):
            lev_p[:, t_start:t_end] = np.nanpercentile(lev_chunk, [5, 25, 50, 75, 95], axis=1)
        del nlv_chunk, lev_chunk

    if hasattr(nlv_transposed, '_mmap'): nlv_transposed._mmap.close()
    if hasattr(lev_transposed, '_mmap'): lev_transposed._mmap.close()
    del nlv_transposed, lev_transposed
    if hasattr(nlv_hist, '_mmap'): nlv_hist._mmap.close()
    if hasattr(lev_hist, '_mmap'): lev_hist._mmap.close()
    del nlv_hist, lev_hist
    if os.path.exists(temp_nlv_t): os.remove(temp_nlv_t)
    if os.path.exists(temp_lev_t): os.remove(temp_lev_t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(dates, nlv_p[2], color="#047857", linewidth=2, label="Median NLV")
    ax1.fill_between(dates, nlv_p[1], nlv_p[3], color="#10B981", alpha=0.4, label="25th - 75th Percentile")
    ax1.fill_between(dates, nlv_p[0], nlv_p[4], color="#10B981", alpha=0.15, label="5th - 95th Percentile")
    ax1.set_title("Portfolio NLV & Dynamic Leverage Trajectories", loc="left", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"Net Liquidation Value ({BASE_CURRENCY})")
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_ylim(np.min(nlv_p[0]), np.max(nlv_p[4]))

    optimal_lev = sim_results["optimal_target_leverage"]
    ax2.plot(dates, lev_p[2], color="#1E3A8A", linewidth=2, label="Median Leverage")
    ax2.fill_between(dates, lev_p[1], lev_p[3], color="#3B82F6", alpha=0.4, label="25th - 75th Percentile")
    ax2.fill_between(dates, lev_p[0], lev_p[4], color="#3B82F6", alpha=0.15, label="5th - 95th Percentile")
    ax2.axhline(optimal_lev, color="#D97706", linestyle=":", linewidth=2, label=f"Rebalancing Target ({optimal_lev:.2f}x)")
    ax2.set_ylabel("Portfolio Leverage (x)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(alpha=0.3)
    min_lev = min(np.min(lev_p[0]), optimal_lev)
    max_lev = max(np.max(lev_p[4]), optimal_lev)
    y_range = max_lev - min_lev if max_lev > min_lev else 0.1
    ax2.set_ylim(min_lev - (y_range * 0.02), max_lev + (y_range * 0.02))
    ax2.legend(loc="upper left")

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)

def plot_risk_curve(sim_results: dict) -> None:
    if "grid_leverages" not in sim_results: return
    levs = sim_results["grid_leverages"]
    probs = sim_results["grid_ruin_probs"] * 100.0 
    optimal_lev = sim_results["optimal_target_leverage"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(levs, probs, color="#991B1B", linewidth=2.5, label="SMA/EL Ruin Probability")
    ax.axhline(MAX_MARGIN_CALL_PROBABILITY * 100, color="black", linestyle="--", linewidth=1.5, label="Risk Budget Limit (3%)")
    
    optimal_prob = probs[np.where(np.isclose(levs, optimal_lev))[0][0]]
    ax.scatter([optimal_lev], [optimal_prob], color="#D97706", s=100, zorder=5, label=f"Optimal Target ({optimal_lev:.2f}x)")
    ax.vlines(x=optimal_lev, ymin=0, ymax=optimal_prob, color="#D97706", linestyle=":", linewidth=2)
    ax.set_title("Systemic Risk Profile Across Leverage Constraints", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Portfolio Leverage (x)")
    ax.set_ylabel("Probability of Margin Call (%)")
    ax.set_ylim(0, min(100.0, max(10.0, optimal_prob * 3)))
    ax.set_xlim(levs[0], levs[-1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()

def plot_terminal_nlv_distribution(strategy_results: dict, benchmark_results: dict, bins: int = 150) -> None:
    strategy_nlv = _extract_terminal_nlv(strategy_results, survivors_only=True)
    benchmark_nlv = _extract_terminal_nlv(benchmark_results, survivors_only=True)
    combined = np.concatenate([strategy_nlv, benchmark_nlv])
    
    x_low, x_high = np.percentile(combined, [0.5, 99.5]) if len(combined) > 0 else (0.0, 1.0)
    strat_clipped = np.clip(strategy_nlv, x_low, x_high)
    bench_clipped = np.clip(benchmark_nlv, x_low, x_high)
    bin_edges = np.linspace(x_low, x_high, bins + 1)

    strategy_median = float(np.median(strategy_nlv))
    benchmark_median = float(np.median(benchmark_nlv))
    median_uplift = strategy_median - benchmark_median
    optimal_target = strategy_results.get("optimal_target_leverage", 1.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(bench_clipped, bins=bin_edges, alpha=0.45, color="#64748B", label="Benchmark (1.0x)")
    ax.hist(strat_clipped, bins=bin_edges, alpha=0.45, color="#047857", label=f"Optimized Rebalancing ({optimal_target:.2f}x)")

    ax.axvline(benchmark_median, color="#334155", linestyle="--", linewidth=2, label=f"Benchmark median: {benchmark_median:,.0f} {BASE_CURRENCY}")
    ax.axvline(strategy_median, color="#065F46", linestyle="--", linewidth=2, label=f"Optimized median: {strategy_median:,.0f} {BASE_CURRENCY}")

    ax.set_title("Terminal NLV Distribution Comparison", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Net Liquidation Value ({BASE_CURRENCY})\n*Outer boundaries act as overflow bins for extreme 1% outliers")
    ax.set_ylabel("Count")
    ax.set_xlim(x_low, x_high)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    annotation = (
        f"Median uplift: {median_uplift:,.0f} {BASE_CURRENCY}\n"
        f"Optimized breach prob.: {strategy_results['prob_ruin']:.2%}\n"
        f"Benchmark breach prob.: {benchmark_results['prob_ruin']:.2%}\n"
    )
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left", bbox={"boxstyle": "round", "alpha": 0.85, "facecolor": "white", "edgecolor": "gray"})
    plt.tight_layout()

# =============================================================================
# PIPELINE EXECUTORS
# =============================================================================

def run_bayesian_pipeline(show_plots: bool, skip_deposit: bool) -> None:
    print_banner("DYNAMIC METHOD: Bayesian HMM Multifactor Framework")
    
    data_engine = DynamicDataEngine()
    data_engine.fetch_data()
    state = data_engine.build_current_state()
    print_initial_state(state)
    
    calibrator = BayesianFactorCalibrator(data_engine.data, num_factors=NUM_FACTORS)
    mcmc_params = calibrator.fit(draws=MCMC_DRAWS, tune=MCMC_TUNE, cores=MCMC_CORES)

    last_withdrawal_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE)
    final_date = last_withdrawal_date + relativedelta(months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS, days=POST_LAST_WITHDRAWAL_BUFFER_DAYS)

    simulator = DynamicMarketSimulator(state, mcmc_params, final_date, skip_next_deposit=skip_deposit)
    print_simulation_horizon(simulator, final_date, last_withdrawal_date)

    optimizer = DynamicCRNGridOptimizer(simulator)
    optimal_results = optimizer.optimize()

    console.print("[dim]Extracting benchmark (1.00x) paths for statistical comparison...[/dim]")
    no_leverage_results = simulator.simulate(target_leverage=1.0, store_paths=True, store_history=show_plots)

    strat_twrr = calculate_annualized_twrr(simulator, optimal_results)
    bench_twrr = calculate_annualized_twrr(simulator, no_leverage_results)

    print_execution_directive(optimal_results, state)
    print_terminal_nlv_comparison(optimal_results, no_leverage_results, strat_twrr, bench_twrr)

    if show_plots:
        plot_time_series_bands(optimal_results)
        plot_terminal_nlv_distribution(optimal_results, no_leverage_results)
        plot_risk_curve(optimal_results)
        console.print("\n[dim]Visualizations generated. Close the plotting windows to continue.[/dim]")
        plt.show()

    def safe_remove(filepath):
        try:
            if os.path.exists(filepath): os.remove(filepath)
        except Exception: pass

    if "history_paths" in optimal_results:
        safe_remove(optimal_results["history_paths"]["nlv_file"])
        safe_remove(optimal_results["history_paths"]["lev_file"])
    if "history_paths" in no_leverage_results:
        safe_remove(no_leverage_results["history_paths"]["nlv_file"])
        safe_remove(no_leverage_results["history_paths"]["lev_file"])


def run_static_pipeline(show_plots: bool, skip_deposit: bool) -> None:
    print_banner("STATIC METHOD: Historical Covariance & Heston-Merton Jump Engine")
    
    data_engine = StaticDataEngine()
    data_engine.fetch_data()
    state = data_engine.build_current_state()
    params = data_engine.estimate_parameters()
    print_initial_state(state)

    last_withdrawal_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE)
    final_date = last_withdrawal_date + relativedelta(months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS, days=POST_LAST_WITHDRAWAL_BUFFER_DAYS)

    simulator = StaticMarketSimulator(state, params, final_date, skip_next_deposit=skip_deposit)
    print_simulation_horizon(simulator, final_date, last_withdrawal_date)

    optimizer = StaticCRNGridOptimizer(simulator)
    optimal_results = optimizer.optimize()

    console.print("[dim]Extracting benchmark (1.00x) paths for statistical comparison...[/dim]")
    no_leverage_results = simulator.simulate(target_leverage=1.0, store_paths=True, store_history=show_plots)

    strat_twrr = calculate_annualized_twrr(simulator, optimal_results)
    bench_twrr = calculate_annualized_twrr(simulator, no_leverage_results)

    print_execution_directive(optimal_results, state)
    print_terminal_nlv_comparison(optimal_results, no_leverage_results, strat_twrr, bench_twrr)

    if show_plots:
        plot_time_series_bands(optimal_results)
        plot_terminal_nlv_distribution(optimal_results, no_leverage_results)
        plot_risk_curve(optimal_results)
        console.print("\n[dim]Visualizations generated. Close the plotting windows to continue.[/dim]")
        plt.show()

# =============================================================================
# MAIN ORCHESTRATION & INTERACTIVE DISPATCHER
# =============================================================================

def main() -> None:
    print_banner("IBKR LDI Optimization System", "Unified Dual-Engine Rebalancing Framework")
    
    console.print("[bold cyan]Select Calibration & Simulation Engine:[/bold cyan]")
    console.print("  [1] Dynamic Bayesian Regime-Switching Factor Model ([bold green]Recommended[/bold green])")
    console.print("  [2] Static Covariance Matrix with Heston-Merton (Archived Baseline)")
    
    choice = Prompt.ask("\nEnter selection", choices=["1", "2"], default="1")
    skip_deposit = Confirm.ask("\n[bold yellow]?[/bold yellow] Skip the next scheduled month-end deposit?", default=False)
    show_plots = Confirm.ask("[bold yellow]?[/bold yellow] Generate and display interactive diagnostic charts?", default=True)
    console.print()

    # Run primary selected engine
    if choice == "1":
        run_bayesian_pipeline(show_plots=show_plots, skip_deposit=skip_deposit)
        alternate_choice = "2"
        alt_name = "Static Heston-Merton Model"
    else:
        run_static_pipeline(show_plots=show_plots, skip_deposit=skip_deposit)
        alternate_choice = "1"
        alt_name = "Dynamic Bayesian Factor Model"

    # Offer side-by-side comparison run using identical deposit/plot settings
    console.print("\n" + "═" * 65)
    run_other = Confirm.ask(f"[bold yellow]?[/bold yellow] Would you like to run the alternative engine ({alt_name}) for comparison?", default=False)
    
    if run_other:
        console.print()
        if alternate_choice == "1":
            run_bayesian_pipeline(show_plots=show_plots, skip_deposit=skip_deposit)
        else:
            run_static_pipeline(show_plots=show_plots, skip_deposit=skip_deposit)

    console.print("\n[bold green]✓ Optimization workflow complete.[/bold green]\n")

if __name__ == "__main__":
    main()