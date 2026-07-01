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
from rich.prompt import Confirm

from data_fetcher import DataEngine
from engine import MarketSimulator
from optimizer import CRNGridOptimizer
from config import (
    CURRENT_DATE, CURRENT_DEBT, CURRENT_SMA, TODAY_DEPOSIT, NUM_PATHS, 
    MAX_MARGIN_CALL_PROBABILITY, POST_LAST_WITHDRAWAL_BUFFER_DAYS, 
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS, ACTIVE_ASSET, WITHDRAWAL_SCHEDULE, BASE_CURRENCY
)

console = Console()

# =============================================================================
# Formatting helpers
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
    return "PASS"

def print_banner(title: str, subtitle: str | None = None) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if subtitle: console.print(f"[dim]{subtitle}[/dim]")
    console.print("─" * 65 + "\n")

def print_table(title: str, columns: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> None:
    right_align = right_align or set()
    table = Table(title=title, box=box.SIMPLE, header_style="bold", show_lines=False, title_style="bold")
    for column in columns: table.add_column(column, justify="right" if column in right_align else "left", overflow="fold")
    for row in rows: table.add_row(*[str(cell) for cell in row])
    console.print(table)
    console.print()

# =============================================================================
# Initial state and parameter reporting
# =============================================================================

def print_initial_state(state: dict) -> float:
    assets = state["assets_dict"]
    gross_assets = sum(info["v0"] for info in assets.values())
    active_value = assets[ACTIVE_ASSET]["v0"]
    passive_value = gross_assets - active_value
    
    total_mmr = sum(info["v0"] * info["mmr"] for info in assets.values())
    total_imr = sum(info["v0"] * info["imr"] for info in assets.values())

    c = TODAY_DEPOSIT
    d = CURRENT_DEBT
    
    if c > 0:
        rep = min(c, d)
        c -= rep
        d -= rep
        
    gross_position = gross_assets + c
    nlv_pre_trade = gross_position - d
    el_pre_trade = nlv_pre_trade - total_mmr
    current_leverage = (gross_position / nlv_pre_trade) if nlv_pre_trade > 0 else float("inf")

    balance_rows = [
        ["Active Rebalancing Asset", f"{ACTIVE_ASSET} ({assets[ACTIVE_ASSET]['currency']})"],
        ["Active Asset Current Value", _fmt_base_ccy(active_value)],
        ["Passive Portfolio Value", _fmt_base_ccy(passive_value)],
        ["Gross Position Value", _fmt_base_ccy(gross_position)],
        ["Current Margin Debt", _fmt_base_ccy(CURRENT_DEBT)],
        ["Net Liquidation Value (NLV)", _fmt_base_ccy(nlv_pre_trade)],
        ["Configured Cash Deposit (Today)", _fmt_base_ccy(TODAY_DEPOSIT)],
        ["Excess Liquidity (EL)", _fmt_base_ccy(el_pre_trade)],
        ["Reg T Special Memorandum (SMA)", _fmt_base_ccy(CURRENT_SMA)],
        ["Pre-Trade Portfolio Leverage", _fmt_x(current_leverage)],
    ]
    print_table("Initial Margin Sheet", ["Metric", "Value"], balance_rows, {"Value"})
    return current_leverage

def print_simulation_horizon(simulator: MarketSimulator, final_date, last_withdrawal_date) -> None:
    total_future_deposits = float(np.sum(simulator.deposits_arr))
    total_withdrawals = float(np.sum(simulator.withdrawals_arr) + simulator.initial_withdrawal_amount)
    deposit_days_count = int(np.count_nonzero(simulator.deposits_arr))

    horizon_rows = [
        ["Start date", str(CURRENT_DATE)],
        ["Final simulation date", str(final_date)],
        ["Calendar days", f"{simulator.days:,}"],
        ["Future deposit events", f"{deposit_days_count:,}"],
        ["Total future scheduled deposits", _fmt_base_ccy(total_future_deposits)],
        ["Total future scheduled withdrawals", _fmt_base_ccy(total_withdrawals)],
        ["Monte Carlo simulated paths", f"{NUM_PATHS:,}"],
    ]
    print_table("Simulation Horizon", ["Metric", "Value"], horizon_rows, {"Value"})

# =============================================================================
# Result reporting & Plotting
# =============================================================================

def print_execution_directive(optimal_results: dict, state: dict) -> None:
    prob_ruin = float(optimal_results["prob_ruin"])
    upper_95_ruin = _binomial_upper_95(prob_ruin, NUM_PATHS)
    optimal_lev = optimal_results["optimal_target_leverage"]
    
    cash = 0.0
    debt = float(CURRENT_DEBT)
    
    init_withdrawal = sum(float(w["amount"]) for w in WITHDRAWAL_SCHEDULE if (w["date"] - CURRENT_DATE).days == 0)
    if init_withdrawal > 0.0:
        cash_used = min(cash, init_withdrawal)
        cash -= cash_used
        debt += (init_withdrawal - cash_used)

    if TODAY_DEPOSIT > 0.0: cash += float(TODAY_DEPOSIT)
        
    debt_repay = min(cash, debt)
    cash -= debt_repay
    debt -= debt_repay
    
    gross_assets = sum(info["v0"] for info in state["assets_dict"].values())
    gross = gross_assets + cash
    nlv = gross - debt
    
    target_gross = nlv * optimal_lev if nlv > 0.0 else gross
    purchase_amount = max(0.0, target_gross - gross)
    
    cash_used_for_purchase = min(cash, purchase_amount)
    borrowed_amount = purchase_amount - cash_used_for_purchase

    rows = [
        ["Ruin Status", _risk_status(prob_ruin)],
        ["Optimal Target Leverage", _fmt_x(optimal_lev)],
        ["Today's Deposit Processed", _fmt_base_ccy(TODAY_DEPOSIT)],
        ["Required Margin Borrowing", _fmt_base_ccy(borrowed_amount)],
        [f"Total {ACTIVE_ASSET} Purchase", f"[bold green]{_fmt_base_ccy(purchase_amount)}[/bold green]"],
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
    return {"mean": float(np.mean(nlv)), "p05": float(np.percentile(nlv, 5)), "median": float(np.percentile(nlv, 50)), "p95": float(np.percentile(nlv, 95))}

def calculate_trajectory_twrr(simulator: MarketSimulator, sim_results: dict, use_median: bool = True) -> float:
    if "history_paths" not in sim_results:
        return float('nan')
    
    shape = sim_results["history_paths"]["shape"]
    nlv_file = sim_results["history_paths"]["nlv_file"]
    
    # Read-only memmap to prevent data corruption
    nlv_hist = np.memmap(nlv_file, dtype=np.float32, mode='r', shape=shape)
    ruined = np.asarray(sim_results.get("Final_Ruined", np.zeros(shape[0], dtype=bool)))
    
    # Filter out paths that breached margin limits
    survivors_idx = np.where(~ruined)[0]
    num_survivors = len(survivors_idx)
    
    if num_survivors == 0:
        return 0.0
        
    num_days = shape[1]
    trajectory = np.zeros(num_days, dtype=np.float64)
    
    # --- Out-of-Core Memory Optimization ---
    # To avoid massive RAM usage OR disk thrashing, we transpose the data in chunks
    # into a temporary Day-Major memory map.
    temp_transposed_file = nlv_file.replace('.dat', '_transposed.dat')
    transposed_hist = np.memmap(temp_transposed_file, dtype=np.float32, mode='w+', shape=(num_days, num_survivors))
    
    # Strict RAM cap: 25k paths * 2000 days * 4 bytes = ~200 MB max RAM usage
    chunk_size = 25000  
    
    with console.status(f"Performing out-of-core transpose on {num_survivors:,} paths...", spinner="dots"):
        for start_idx in range(0, num_survivors, chunk_size):
            end_idx = min(start_idx + chunk_size, num_survivors)
            
            # Fast contiguous read from disk (Path-Major)
            path_chunk = nlv_hist[survivors_idx[start_idx:end_idx], :] 
            
            # Fast contiguous write to disk (Day-Major)
            transposed_hist[:, start_idx:end_idx] = path_chunk.T 
            
    with console.status("Computing continuous TWRR trajectory...", spinner="dots"):
        for t in range(num_days):
            # Now, reading a single day is a perfect contiguous block read!
            day_data = transposed_hist[t, :]
            
            if use_median:
                trajectory[t] = np.median(day_data)
            else:
                trajectory[t] = np.mean(day_data)
    
    if hasattr(transposed_hist, '_mmap'):
        transposed_hist._mmap.close()
    del transposed_hist
    
    if hasattr(nlv_hist, '_mmap'):
        nlv_hist._mmap.close()
    del nlv_hist
    
    if os.path.exists(temp_transposed_file):
        os.remove(temp_transposed_file)
        
    # Isolate deterministic net cash flows: Deposits - Withdrawals
    net_cf = np.zeros(num_days, dtype=np.float64)
    net_cf[1:] = simulator.deposits_arr[1:] - simulator.withdrawals_arr[1:]
    
    twrr_index = 1.0
    for t in range(1, num_days):
        v_start = trajectory[t-1]
        v_end = trajectory[t]
        cf = net_cf[t]
        
        if v_start <= 0:
            continue # Safe-guard against calculation breaks
            
        daily_ret = (v_end - cf) / v_start
        twrr_index *= daily_ret
        
    years = num_days / 365.0
    annualized_twrr = (twrr_index ** (1.0 / years)) - 1.0
    
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
        ["Annualized TWRR (Median Path)", _fmt_pct(strategy_twrr), _fmt_pct(benchmark_twrr), _fmt_pct(twrr_uplift)]
    ]
    print_table("Terminal NLV & Performance Comparison (Survivors)", ["Metric", "Optimized Rebalancing", "Benchmark (1.0x)", "Uplift"], rows, {"Optimized Rebalancing", "Benchmark (1.0x)", "Uplift"})


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
    
    # --- Out-of-Core Transpose ---
    temp_nlv_t = nlv_file.replace('.dat', '_transposed.dat')
    temp_lev_t = lev_file.replace('.dat', '_transposed.dat')
    
    nlv_transposed = np.memmap(temp_nlv_t, dtype=np.float32, mode='w+', shape=(num_days, num_paths))
    lev_transposed = np.memmap(temp_lev_t, dtype=np.float32, mode='w+', shape=(num_days, num_paths))
    
    chunk_size = 25000 
    
    with console.status("Pivoting memory maps for fast quantile extraction...", spinner="dots"):
        for start_idx in range(0, num_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, num_paths)
            
            # Fast contiguous read -> transpose -> fast contiguous write
            nlv_transposed[:, start_idx:end_idx] = nlv_hist[start_idx:end_idx, :].T
            lev_transposed[:, start_idx:end_idx] = lev_hist[start_idx:end_idx, :].T
            
    # Process 25 days at a time. 
    # For 1M paths, this strictly caps RAM usage at ~200MB per iteration.
    chunk_days = 25  
    
    with console.status(f"Vectorizing daily quantiles in chunks of {chunk_days} days...", spinner="dots"):
        for t_start in range(0, num_days, chunk_days):
            t_end = min(t_start + chunk_days, num_days)
            
            # 1. Pull the chunk entirely into RAM as a standard array.
            # This detaches it from the memmap, preventing Windows from locking up your page file.
            nlv_chunk = np.array(nlv_transposed[t_start:t_end, :])
            lev_chunk = np.array(lev_transposed[t_start:t_end, :])
            
            # 2. Compute across the entire chunk simultaneously (axis=1)
            # NLV doesn't contain NaNs, so standard np.percentile is 5x faster and uses half the RAM.
            nlv_p[:, t_start:t_end] = np.percentile(nlv_chunk, [5, 25, 50, 75, 95], axis=1)
            
            # Leverage does contain NaNs, so we must use nanpercentile here.
            with np.errstate(invalid='ignore'):
                lev_p[:, t_start:t_end] = np.nanpercentile(lev_chunk, [5, 25, 50, 75, 95], axis=1)
            
            # 3. Explicitly nuke the RAM arrays to keep usage perfectly flat at ~200MB
            del nlv_chunk, lev_chunk

    # Safely release the memory map locks (Explicit Windows Unlocking)
    if hasattr(nlv_transposed, '_mmap'): nlv_transposed._mmap.close()
    if hasattr(lev_transposed, '_mmap'): lev_transposed._mmap.close()
    del nlv_transposed, lev_transposed
    
    if hasattr(nlv_hist, '_mmap'): nlv_hist._mmap.close()
    if hasattr(lev_hist, '_mmap'): lev_hist._mmap.close()
    del nlv_hist, lev_hist
    
    # Clean up only the transposed temp files (leave the originals for the global cleanup at the end of main)
    if os.path.exists(temp_nlv_t): os.remove(temp_nlv_t)
    if os.path.exists(temp_lev_t): os.remove(temp_lev_t)

    # Change to a 2x1 vertical layout with a shared X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Top Plot: NLV ---
    ax1.plot(dates, nlv_p[2], color="#047857", linewidth=2, label="Median NLV")
    ax1.fill_between(dates, nlv_p[1], nlv_p[3], color="#10B981", alpha=0.4, label="25th - 75th Percentile")
    ax1.fill_between(dates, nlv_p[0], nlv_p[4], color="#10B981", alpha=0.15, label="5th - 95th Percentile")
    
    ax1.set_title("Portfolio NLV & Dynamic Leverage Trajectories", loc="left", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"Net Liquidation Value ({BASE_CURRENCY})")
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Strictly bind the Y-axis to the actual data bounds of the 5th and 95th percentiles
    ax1.set_ylim(np.min(nlv_p[0]), np.max(nlv_p[4]))

    # --- Bottom Plot: Leverage ---
    optimal_lev = sim_results["optimal_target_leverage"]
    ax2.plot(dates, lev_p[2], color="#1E3A8A", linewidth=2, label="Median Leverage")
    ax2.fill_between(dates, lev_p[1], lev_p[3], color="#3B82F6", alpha=0.4, label="25th - 75th Percentile")
    ax2.fill_between(dates, lev_p[0], lev_p[4], color="#3B82F6", alpha=0.15, label="5th - 95th Percentile")
    ax2.axhline(optimal_lev, color="#D97706", linestyle=":", linewidth=2, label="Rebalancing Target")
    
    ax2.set_ylabel("Portfolio Leverage (x)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(alpha=0.3)
    
    # Strictly bind the Y-axis to the actual data bounds, including the target line
    min_lev = min(np.min(lev_p[0]), optimal_lev)
    max_lev = max(np.max(lev_p[4]), optimal_lev)
    
    # Add a tiny 2% visual buffer so the outer edges aren't clipped by the plot frame
    y_range = max_lev - min_lev if max_lev > min_lev else 0.1
    ax2.set_ylim(min_lev - (y_range * 0.02), max_lev + (y_range * 0.02))
    
    ax2.legend(loc="upper left")

    # Tight layout with zero vertical pad to merge the shared axis cleanly
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)

def plot_risk_curve(sim_results: dict) -> None:
    if "grid_leverages" not in sim_results: return
    
    levs = sim_results["grid_leverages"]
    probs = sim_results["grid_ruin_probs"] * 100.0 
    optimal_lev = sim_results["optimal_target_leverage"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(levs, probs, color="#991B1B", linewidth=2.5, label="SMA/EL Ruin Probability")
    ax.axhline(MAX_MARGIN_CALL_PROBABILITY * 100, color="black", linestyle="--", linewidth=1.5, label="Risk Budget Limit")
    
    optimal_prob = probs[np.where(np.isclose(levs, optimal_lev))[0][0]]
    ax.scatter([optimal_lev], [optimal_prob], color="#D97706", s=100, zorder=5, label=f"Optimal Target ({optimal_lev:.2f}x)")
    ax.vlines(x=optimal_lev, ymin=0, ymax=optimal_prob, color="#D97706", linestyle=":", linewidth=2)
    
    ax.set_title("Systemic Risk Profile Across Leverage Constraints", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Portfolio Leverage (x)")
    ax.set_ylabel("Probability of Margin Call (%)")
    
    max_y = min(100.0, max(10.0, optimal_prob * 3))
    ax.set_ylim(0, max_y)
    ax.set_xlim(levs[0], levs[-1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    
    plt.tight_layout()

def plot_terminal_diagnostics(sim_results: dict) -> None:
    gross_assets = np.asarray(sim_results["Final_Gross_Assets"])
    nlv = np.asarray(sim_results["Final_NLV"])
    ruined = np.asarray(sim_results["Final_Ruined"], dtype=bool)

    gross_survivors = gross_assets[~ruined]
    nlv_survivors = nlv[~ruined]

    with np.errstate(divide='ignore', invalid='ignore'):
        leverage = np.where(nlv_survivors > 0, gross_survivors / nlv_survivors, np.nan)
    leverage = leverage[np.isfinite(leverage)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Gross Assets Plot (with Overflow Clipping) ---
    if len(gross_survivors) > 0:
        g_low, g_high = np.percentile(gross_survivors, [0.5, 99.5])
        gross_clipped = np.clip(gross_survivors, g_low, g_high)
        
        ax1.hist(gross_clipped, bins=90, color="#047857", alpha=0.6)
        median_gross = np.median(gross_survivors) # Median uses original unclipped data
        ax1.axvline(median_gross, color="#065F46", linestyle="--", linewidth=2, label=f"Median: {median_gross:,.0f} {BASE_CURRENCY}")
        
    ax1.set_title("Terminal Gross Assets (Survivors)", loc="left", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"Gross Assets ({BASE_CURRENCY})\n*Outer bins contain clipped extreme outliers")
    ax1.set_ylabel("Count")
    ax1.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.grid(alpha=0.3)
    ax1.legend()

    # --- Leverage Plot (with Overflow Clipping) ---
    if len(leverage) > 0:
        # Leverage doesn't usually go negative, so we only cap the extreme upper tail
        l_high = np.percentile(leverage, 99.5)
        # Ensure the plot at least reaches the target leverage
        max_plot_lev = max(sim_results["optimal_target_leverage"] * 1.5, l_high) 
        leverage_clipped = np.clip(leverage, 1.0, max_plot_lev)

        ax2.hist(leverage_clipped, bins=np.linspace(1.0, max_plot_lev, 90), color="#1E3A8A", alpha=0.6)
        median_lev = np.median(leverage) # Median uses original unclipped data
        
        ax2.axvline(median_lev, color="#1E3A8A", linestyle="--", linewidth=2, label=f"Median Final Leverage: {median_lev:.2f}x")
        ax2.axvline(sim_results["optimal_target_leverage"], color="#D97706", linestyle=":", linewidth=2, label=f"Target L: {sim_results['optimal_target_leverage']:.2f}x")
        ax2.set_xlim(1.0, max_plot_lev)

    ax2.set_title("Terminal Portfolio Leverage (Survivors)", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Leverage (x)\n*Outer bins contain clipped extreme outliers")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()


def plot_terminal_nlv_distribution(
    strategy_results: dict, benchmark_results: dict, bins: int = 150, survivors_only: bool = True,
) -> None:
    strategy_nlv = _extract_terminal_nlv(strategy_results, survivors_only=survivors_only)
    benchmark_nlv = _extract_terminal_nlv(benchmark_results, survivors_only=survivors_only)

    combined = np.concatenate([strategy_nlv, benchmark_nlv])
    
    # Identify the global boundaries for the 0.5% and 99.5% percentiles across BOTH datasets
    if len(combined) > 0:
        x_low, x_high = np.percentile(combined, [0.5, 99.5])
    else:
        x_low, x_high = 0.0, 1.0

    # Clip both arrays to these exact boundaries. 
    # This creates the "Overflow Bin" effect at the extreme left and right.
    strat_clipped = np.clip(strategy_nlv, x_low, x_high)
    bench_clipped = np.clip(benchmark_nlv, x_low, x_high)

    bin_edges = np.linspace(x_low, x_high, bins + 1)

    # Medians should always be calculated on the pure, unclipped data
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
# Main orchestration
# =============================================================================

def main() -> None:
    print_banner("IBKR LDI Optimization Engine", "Dual-Constraint Ruin Engine (Reg T SMA + Maintenance Margin)")

    data_engine = DataEngine()
    data_engine.fetch_data()
    state = data_engine.build_current_state()
    params = data_engine.estimate_parameters()

    print_initial_state(state)

    skip_deposit = Confirm.ask("\n[bold yellow]?[/bold yellow] Skip the next scheduled month-end deposit? (e.g., if already manually funded)", default=False)
    console.print()

    last_withdrawal_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE)
    final_date = last_withdrawal_date + relativedelta(months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS, days=POST_LAST_WITHDRAWAL_BUFFER_DAYS)

    simulator = MarketSimulator(state, params, final_date, skip_next_deposit=skip_deposit)
    print_simulation_horizon(simulator, final_date, last_withdrawal_date)

    optimizer = CRNGridOptimizer(simulator)
    optimal_results = optimizer.optimize()

    console.print("[dim]Extracting benchmark (1.00x) paths for statistical comparison...[/dim]")
    no_leverage_results = simulator.simulate(target_leverage=1.0, store_paths=True, store_history=True)

    strat_twrr = calculate_trajectory_twrr(simulator, optimal_results, use_median=True)
    bench_twrr = calculate_trajectory_twrr(simulator, no_leverage_results, use_median=True)

    print_execution_directive(optimal_results, state)
    print_terminal_nlv_comparison(optimal_results, no_leverage_results, strat_twrr, bench_twrr)

    plot_time_series_bands(optimal_results)
    plot_terminal_nlv_distribution(optimal_results, no_leverage_results)
    plot_terminal_diagnostics(optimal_results)
    plot_risk_curve(optimal_results)
    
    console.print("\n[dim]Visualizations generated. Close the plotting windows to exit the script.[/dim]")
    plt.show()

    def safe_remove(filepath):
        try:
            if os.path.exists(filepath): os.remove(filepath)
        except Exception: 
            pass

    console.print("[dim]Cleaning up memory-mapped cache files...[/dim]")
    if "history_paths" in optimal_results:
        safe_remove(optimal_results["history_paths"]["nlv_file"])
        safe_remove(optimal_results["history_paths"]["lev_file"])
    if "history_paths" in no_leverage_results:
        safe_remove(no_leverage_results["history_paths"]["nlv_file"])
        safe_remove(no_leverage_results["history_paths"]["lev_file"])

if __name__ == "__main__":
    main()