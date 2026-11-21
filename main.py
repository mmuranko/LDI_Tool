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
    NUM_FACTORS, MCMC_DRAWS, MCMC_TUNE, MCMC_CORES, IM_REQUIREMENTS, MM_REQUIREMENTS
)
from data_fetcher import DataEngine
from calibration import BayesianFactorCalibrator
from engine import MarketSimulator
from optimizer import CRNGridOptimizer

console = Console(highlight=False)


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_base_ccy(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    return f"{x:,.{decimals}f} {BASE_CURRENCY}" if np.isfinite(x) else "n/a"


def _fmt_pct(value, decimals: int = 2) -> str:
    x = _safe_float(value)
    return f"{x:.{decimals}%}" if np.isfinite(x) else "n/a"


def _fmt_x(value, decimals: int = 3) -> str:
    x = _safe_float(value)
    if np.isposinf(x):
        return "∞x"
    return f"{x:.{decimals}f}x" if np.isfinite(x) else "n/a"


def _binomial_upper_95(p_hat: float, n: int) -> float:
    p_hat = float(p_hat)
    if not np.isfinite(p_hat) or n <= 0:
        return float("nan")
    se = np.sqrt(max(p_hat * (1.0 - p_hat), 1e-12) / n)
    return min(1.0, p_hat + 1.96 * se)


def print_banner(title: str, subtitle: str | None = None) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if subtitle:
        console.print(f"{subtitle}")
    console.print("────────────────────────────────────────────────────────────\n")


def print_table(title: str, columns: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> None:
    right_align = right_align or set()
    table = Table(title=title, box=box.SIMPLE, header_style="bold", show_lines=False, title_style="bold")
    for column in columns:
        table.add_column(column, justify="right" if column in right_align else "left")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)
    console.print()


def print_initial_state(state: dict) -> float:
    assets = state["assets_dict"]
    gross_assets = sum(info["v0"] for info in assets.values() if not info.get("is_fx_tracker", False))
    total_mmr = sum(info["v0"] * info["mmr"] for info in assets.values() if not info.get("is_fx_tracker", False))

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
        ["Pre-Trade Portfolio Leverage", _fmt_x(current_leverage, decimals=2)],
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
    optimal_lev = float(optimal_results["optimal_target_leverage"])

    init_w = sum(float(w["amount"]) for w in WITHDRAWAL_SCHEDULE if (w["date"] - CURRENT_DATE).days == 0)
    debt = float(CURRENT_DEBT) + init_w
    sma_avail = float(CURRENT_SMA) - init_w

    gross = sum(info["v0"] for info in state["assets_dict"].values() if not info.get("is_fx_tracker", False))
    total_mmr = sum(info["v0"] * info["mmr"] for info in state["assets_dict"].values() if not info.get("is_fx_tracker", False))
    nlv = gross - debt

    target_gross = nlv * optimal_lev if nlv > 0.0 else gross
    target_purchase = max(0.0, target_gross - gross)

    imr_active = float(IM_REQUIREMENTS.get(ACTIVE_ASSET, 0.50))
    mmr_active = float(MM_REQUIREMENTS.get(ACTIVE_ASSET, 0.25))
    max_sma_purchase = max(0.0, sma_avail / imr_active)
    max_el_purchase = max(0.0, (nlv - total_mmr) / mmr_active)
    actual_immediate_purchase = min(target_purchase, max_sma_purchase, max_el_purchase)

    curr_lev = gross / nlv if nlv > 0.0 else float("inf")
    deposit_needed = ((gross / optimal_lev) - nlv) if curr_lev > optimal_lev and optimal_lev > 0.0 else 0.0

    upper_95 = _binomial_upper_95(prob_ruin, NUM_PATHS)
    status = (
        "BREACH" if prob_ruin > MAX_MARGIN_CALL_PROBABILITY
        else "MARGINAL" if upper_95 > MAX_MARGIN_CALL_PROBABILITY
        else "PASS"
    )

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
    if survivors_only:
        nlv = nlv[~np.asarray(results.get("Final_Ruined", np.zeros(nlv.shape, dtype=bool)), dtype=bool)]
    return nlv[np.isfinite(nlv)] if drop_invalid else nlv


def calculate_annualized_mwrr(simulator, sim_results: dict) -> float:
    """Computes exact annualized Money-Weighted Rate of Return (IRR) on the median survivor."""
    survivor_nlv = _extract_terminal_nlv(sim_results, survivors_only=True)
    if len(survivor_nlv) == 0:
        return float("nan")

    v_terminal = float(np.median(survivor_nlv))
    v_0 = float(
        sum(float(info["v0"]) for info in simulator.assets_dict.values() if not info.get("is_fx_tracker", False))
        - CURRENT_DEBT
        - simulator.initial_withdrawal_amount
    )
    days = float(simulator.days)
    if days <= 0 or v_0 <= 0:
        return float("nan")

    net_daily_cf = np.asarray(simulator.deposits_arr - simulator.withdrawals_arr, dtype=np.float64)
    cf_indices = np.nonzero(net_daily_cf)[0]
    cf_amounts = net_daily_cf[cf_indices]
    t_years_rem = (days - 1.0 - cf_indices) / 365.25
    total_years = (days - 1.0) / 365.25

    # Solve for continuous rate r where FV(v_0) + FV(cash_flows) == v_terminal via bisection
    low, high = -0.95, 1.50
    for _ in range(80):
        mid = 0.5 * (low + high)
        fv = v_0 * np.exp(mid * total_years) + float(np.sum(cf_amounts * np.exp(mid * t_years_rem)))
        if fv < v_terminal:
            low = mid
        else:
            high = mid

    return float(np.exp(0.5 * (low + high)) - 1.0)


def print_terminal_comparison(opt_res: dict, bench_res: dict, opt_irr: float, bench_irr: float) -> None:
    opt_nlv = _extract_terminal_nlv(opt_res)
    bench_nlv = _extract_terminal_nlv(bench_res)

    rows = [
        [
            "Median Terminal NLV",
            _fmt_base_ccy(float(np.median(opt_nlv))) if len(opt_nlv) else "n/a",
            _fmt_base_ccy(float(np.median(bench_nlv))) if len(bench_nlv) else "n/a"
        ],
        ["Annualized MWRR (IRR)", _fmt_pct(opt_irr), _fmt_pct(bench_irr)]
    ]
    print_table(
        "PERFORMANCE COMPARISON (Median Survivor)",
        ["Metric", "Optimized", "Benchmark (1.0x)"],
        rows,
        {"Optimized", "Benchmark (1.0x)"}
    )


def show_visualizations(opt_res: dict, bench_res: dict) -> None:
    console.print("Generating structural diagnostic charts...")
    plt.style.use("default")

    if "history_paths" in opt_res:
        shape = opt_res["history_paths"]["shape"]
        dates = opt_res["history_paths"]["dates"]
        _, num_points = shape

        nlv_hist = np.memmap(opt_res["history_paths"]["nlv_file"], dtype=np.float32, mode="r", shape=shape)
        lev_hist = np.memmap(opt_res["history_paths"]["lev_file"], dtype=np.float32, mode="r", shape=shape)

        nlv_p = np.zeros((5, num_points), dtype=np.float32)
        lev_p = np.zeros((5, num_points), dtype=np.float32)

        chunk_cols = 50
        for c_start in range(0, num_points, chunk_cols):
            c_end = min(c_start + chunk_cols, num_points)
            nlv_chunk = np.array(nlv_hist[:, c_start:c_end])
            lev_chunk = np.array(lev_hist[:, c_start:c_end])
            with np.errstate(invalid="ignore"):
                nlv_p[:, c_start:c_end] = np.nanpercentile(nlv_chunk, [5, 25, 50, 75, 95], axis=0)
                lev_p[:, c_start:c_end] = np.nanpercentile(lev_chunk, [5, 25, 50, 75, 95], axis=0)

        del nlv_hist
        del lev_hist

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(dates, nlv_p[2], color="#1E293B", linewidth=2, label="Median NLV (Survivors)")
        ax1.fill_between(dates, nlv_p[1], nlv_p[3], color="#475569", alpha=0.4, label="25th - 75th Percentile")
        ax1.fill_between(dates, nlv_p[0], nlv_p[4], color="#94A3B8", alpha=0.2, label="5th - 95th Percentile")
        ax1.set_title("Portfolio NLV & Dynamic Leverage Trajectories", loc="left", fontsize=12, fontweight="bold")
        ax1.set_ylabel(f"Net Liquidation Value ({BASE_CURRENCY})")
        ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        optimal_lev = float(opt_res["optimal_target_leverage"])
        ax2.plot(dates, lev_p[2], color="#0F172A", linewidth=2, label="Median Leverage (Survivors)")
        ax2.fill_between(dates, lev_p[1], lev_p[3], color="#334155", alpha=0.4)
        ax2.fill_between(dates, lev_p[0], lev_p[4], color="#64748B", alpha=0.2)
        ax2.axhline(optimal_lev, color="#B91C1C", linestyle=":", linewidth=2, label=f"Rebalancing Target ({optimal_lev:.3f}x)")
        ax2.set_ylabel("Portfolio Leverage (x)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left")
        fig1.tight_layout()

    if "grid_leverages" in opt_res:
        fig2, ax3 = plt.subplots(figsize=(8, 5))
        levs = np.asarray(opt_res["grid_leverages"], dtype=np.float64)
        probs = np.asarray(opt_res["grid_ruin_probs"], dtype=np.float64)
        optimal_lev = float(opt_res["optimal_target_leverage"])

        ax3.plot(levs, probs, color="#1E293B", linewidth=2)
        ax3.axhline(
            MAX_MARGIN_CALL_PROBABILITY, color="#B91C1C", linestyle="--",
            linewidth=1.5, label=f"Risk Budget ({MAX_MARGIN_CALL_PROBABILITY:.1%})"
        )

        closest_idx = int(np.argmin(np.abs(levs - optimal_lev)))
        optimal_prob = float(probs[closest_idx])
        ax3.scatter([optimal_lev], [optimal_prob], color="#B91C1C", s=60, zorder=5, label=f"Optimal Target ({optimal_lev:.3f}x)")

        ax3.set_title("Systemic Risk Profile (Ruin Frontier)", loc="left", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Target Portfolio Leverage (x)")
        ax3.set_ylabel("Probability of Margin Call")
        ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax3.set_ylim(0, min(1.0, max(0.1, optimal_prob * 3.0)))
        ax3.grid(alpha=0.3)
        ax3.legend(loc="upper left")
        fig2.tight_layout()

    strat_nlv = _extract_terminal_nlv(opt_res, survivors_only=True)
    bench_nlv = _extract_terminal_nlv(bench_res, survivors_only=True)

    if len(strat_nlv) > 0 and len(bench_nlv) > 0:
        fig3, ax4 = plt.subplots(figsize=(9, 5))
        x_low, x_high = np.percentile(np.concatenate([strat_nlv, bench_nlv]), [0.5, 99.5])
        bins = np.linspace(x_low, x_high, 100)

        ax4.hist(bench_nlv, bins=bins, alpha=0.5, color="#94A3B8", label="Benchmark (1.000x)")
        ax4.hist(strat_nlv, bins=bins, alpha=0.5, color="#0F172A", label=f"Optimized ({opt_res.get('optimal_target_leverage', 1.0):.3f}x)")

        ax4.axvline(np.median(bench_nlv), color="#64748B", linestyle="--", linewidth=1.5)
        ax4.axvline(np.median(strat_nlv), color="#1E293B", linestyle="--", linewidth=1.5)

        ax4.set_title("Terminal Wealth Distribution (Survivors)", loc="left", fontsize=12, fontweight="bold")
        ax4.set_xlabel(f"Net Liquidation Value ({BASE_CURRENCY})")
        ax4.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax4.get_yaxis().set_visible(False)
        ax4.legend(loc="upper right")
        fig3.tight_layout()

    plt.show()


def _safe_clean(res_dict: dict | None) -> None:
    if isinstance(res_dict, dict) and "history_paths" in res_dict:
        for f in (res_dict["history_paths"]["nlv_file"], res_dict["history_paths"]["lev_file"]):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


def main() -> None:
    print_banner("LDI OPTIMIZATION SYSTEM", "Bayesian Factors & Dynamic Cash Flows")

    skip_deposit = Confirm.ask("Skip next scheduled deposit?", default=False)
    show_plots = Confirm.ask("Display interactive charts?", default=False)
    console.print()

    opt_results = None
    bench_results = None

    try:
        engine = DataEngine()
        engine.fetch_data()
        state = engine.build_current_state()
        print_initial_state(state)

        calibrator = BayesianFactorCalibrator(engine.data, num_factors=NUM_FACTORS)
        mcmc_params = calibrator.fit(draws=MCMC_DRAWS, tune=MCMC_TUNE, cores=MCMC_CORES)

        final_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE) + relativedelta(
            months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS, days=POST_LAST_WITHDRAWAL_BUFFER_DAYS
        )
        simulator = MarketSimulator(state, mcmc_params, final_date, skip_next_deposit=skip_deposit)
        print_simulation_horizon(simulator, final_date)

        opt_results = CRNGridOptimizer(simulator).optimize(store_history=show_plots)

        console.print("Extracting benchmark (1.000x) base cases...")
        bench_results = simulator.simulate(target_leverage=1.0, store_paths=True, store_history=False)
        console.print()

        print_execution_directive(opt_results, state)
        print_terminal_comparison(
            opt_results,
            bench_results,
            calculate_annualized_mwrr(simulator, opt_results),
            calculate_annualized_mwrr(simulator, bench_results)
        )

        if show_plots:
            show_visualizations(opt_results, bench_results)
    finally:
        _safe_clean(opt_results)
        _safe_clean(bench_results)


if __name__ == "__main__":
    main()