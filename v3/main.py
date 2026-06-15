import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from dateutil.relativedelta import relativedelta
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_fetcher import DataEngine
from engine import MarketSimulator
from optimizer import MarginOptimizer
from config import (
    CURRENT_DATE,
    CURRENT_DEBT,
    TODAY_DEPOSIT,
    NUM_PATHS,
    MAX_MARGIN_CALL_PROBABILITY,
    POST_LAST_WITHDRAWAL_BUFFER_DAYS,
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS,
    TARGET_ASSET,
    WITHDRAWAL_SCHEDULE,
    CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX,
    CONTRIBUTION_POLICY_NO_INVEST_MIN,
)


console = Console()


# =============================================================================
# Formatting helpers
# =============================================================================

def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_chf(value, decimals: int = 2) -> str:
    x = _safe_float(value)

    if not np.isfinite(x):
        return "n/a"

    return f"{x:,.{decimals}f} CHF"


def _fmt_pct(value, decimals: int = 2) -> str:
    x = _safe_float(value)

    if not np.isfinite(x):
        return "n/a"

    return f"{x:.{decimals}%}"


def _fmt_x(value, decimals: int = 2) -> str:
    x = _safe_float(value)

    if np.isposinf(x):
        return "∞x"

    if not np.isfinite(x):
        return "n/a"

    return f"{x:.{decimals}f}x"


def _fmt_bool(value) -> str:
    return "[green]Yes[/green]" if bool(value) else "[red]No[/red]"


def _binomial_upper_95(p_hat: float, n: int) -> float:
    """
    Normal-approximation upper 95% Monte Carlo confidence bound.

    Used only for reporting. The optimizer itself still uses the point estimate
    unless you explicitly change optimizer.py.
    """
    p_hat = float(p_hat)

    if not np.isfinite(p_hat):
        return float("nan")

    if n <= 0:
        return float("nan")

    se = np.sqrt(max(p_hat * (1.0 - p_hat), 1e-12) / n)
    return min(1.0, p_hat + 1.96 * se)


def _get_optimal_contribution_leverage(results: dict) -> float:
    """
    Transitional helper while migrating from optimal_target_leverage to
    optimal_contribution_leverage.
    """
    for key in (
        "optimal_contribution_leverage",
        "optimal_target_leverage",
        "contribution_leverage_tested",
    ):
        if key in results and results[key] is not None:
            return float(results[key])

    return float("nan")


def _risk_status(prob_ruin: float) -> str:
    upper_95 = _binomial_upper_95(prob_ruin, NUM_PATHS)

    if prob_ruin > MAX_MARGIN_CALL_PROBABILITY:
        return "[bold red]BREACH[/bold red]"

    if upper_95 > MAX_MARGIN_CALL_PROBABILITY:
        return "[bold yellow]PASS, but close to risk budget[/bold yellow]"

    return "[bold green]PASS[/bold green]"


def print_banner(title: str, subtitle: str | None = None) -> None:
    text = f"[bold]{title}[/bold]"

    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"

    console.print()
    console.print(Panel.fit(text, border_style="cyan"))
    console.print()


def print_table(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    right_align: set[str] | None = None,
) -> None:
    right_align = right_align or set()

    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
        title_style="bold",
    )

    for column in columns:
        table.add_column(
            column,
            justify="right" if column in right_align else "left",
            overflow="fold",
        )

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)


# =============================================================================
# Initial state and parameter reporting
# =============================================================================

def print_initial_state(state: dict, params: dict) -> None:
    target = params["target_factor"]

    v_target = float(state["v_target_0"])
    v_legacy = float(state["v_legacy_0"])
    gross_assets = v_target + v_legacy
    nav_before_today_deposit = gross_assets - CURRENT_DEBT

    current_leverage = (
        gross_assets / nav_before_today_deposit
        if nav_before_today_deposit > 0.0
        else float("inf")
    )

    nav_after_today_deposit_before_trade = nav_before_today_deposit + TODAY_DEPOSIT

    balance_rows = [
        ["Target asset", f"{TARGET_ASSET} ({target['currency']})"],
        ["Target value", _fmt_chf(v_target)],
        ["Legacy value", _fmt_chf(v_legacy)],
        ["Gross assets", _fmt_chf(gross_assets)],
        ["Current margin debt", _fmt_chf(CURRENT_DEBT)],
        ["NAV before today's deposit", _fmt_chf(nav_before_today_deposit)],
        ["Today's configured deposit", _fmt_chf(TODAY_DEPOSIT)],
        [
            "NAV after today's deposit before trade",
            _fmt_chf(nav_after_today_deposit_before_trade),
        ],
        ["Portfolio leverage before today's deposit", _fmt_x(current_leverage)],
    ]

    print_table(
        title="Initial Balance Sheet",
        columns=["Metric", "Value"],
        rows=balance_rows,
        right_align={"Value"},
    )

    asset_rows = []

    for ticker in state["legacy_asset_order"]:
        asset_state = state["legacy_assets"][ticker]
        param_state = params["legacy_assets"].get(ticker, {})
        sigma = param_state.get("sigma", float("nan"))

        asset_rows.append(
            [
                ticker,
                _fmt_chf(asset_state["v0"]),
                asset_state["currency"],
                _fmt_pct(asset_state["m"]),
                _fmt_pct(sigma),
            ]
        )

    if not asset_rows:
        asset_rows = [["None", "-", "-", "-", "-"]]

    print_table(
        title="Legacy Assets",
        columns=["Asset", "Value", "CCY", "Margin req.", "Vol"],
        rows=asset_rows,
        right_align={"Value", "Margin req.", "Vol"},
    )

    currency_rows = []

    for ccy, bucket in sorted(state["legacy_by_currency"].items()):
        currency_rows.append(
            [
                ccy,
                _fmt_chf(bucket["v0"]),
                _fmt_pct(bucket["m"]),
                ", ".join(bucket["tickers"]),
            ]
        )

    if not currency_rows:
        currency_rows = [["None", "-", "-", "-"]]

    print_table(
        title="Legacy Currency Exposure",
        columns=["CCY", "Value", "Avg. margin req.", "Assets"],
        rows=currency_rows,
        right_align={"Value", "Avg. margin req."},
    )


def print_parameter_summary(params: dict) -> None:
    target = params["target_factor"]

    factor_rows = [
        [
            "Target",
            target["ticker"],
            target["currency"],
            _fmt_pct(target["mu_raw"]),
            _fmt_pct(target["mu"]),
            _fmt_pct(target["sigma"]),
        ]
    ]

    for ticker, info in params["legacy_assets"].items():
        factor_rows.append(
            [
                "Legacy",
                ticker,
                info["currency"],
                _fmt_pct(info["mu_raw"]),
                _fmt_pct(info["mu"]),
                _fmt_pct(info["sigma"]),
            ]
        )

    for ccy, info in params["fx_factors"].items():
        factor_rows.append(
            [
                "FX",
                f"{ccy}/{params['base_currency']}",
                ccy,
                _fmt_pct(info["mu_raw"]),
                _fmt_pct(info["mu"]),
                _fmt_pct(info["sigma"]),
            ]
        )

    print_table(
        title="Drift / Volatility Estimates",
        columns=["Type", "Name", "CCY", "Raw drift", "Used drift", "Vol"],
        rows=factor_rows,
        right_align={"Raw drift", "Used drift", "Vol"},
    )

    engine_rows = [
        ["Number of factors", f"{len(params['factor_names']):,}"],
        ["Aligned observations", f"{params['aligned_observations']:,}"],
        ["Base currency", params["base_currency"]],
        ["Factor order", ", ".join(params["factor_names"])],
    ]

    print_table(
        title="Correlation Engine",
        columns=["Metric", "Value"],
        rows=engine_rows,
        right_align={"Value"},
    )


def print_simulation_horizon(
    simulator: MarketSimulator,
    final_date,
    last_withdrawal_date,
) -> None:
    total_future_deposits = float(sum(simulator.deposit_amounts))
    total_withdrawals = float(sum(simulator.withdrawal_amounts))

    horizon_rows = [
        ["Start date", str(CURRENT_DATE)],
        ["Last withdrawal date", str(last_withdrawal_date)],
        ["Final simulation date", str(final_date)],
        ["Calendar days", f"{simulator.days:,}"],
        ["Future scheduled deposit dates", f"{len(simulator.deposit_days):,}"],
        ["Total future scheduled deposits", _fmt_chf(total_future_deposits)],
        ["Future withdrawal dates", f"{len(simulator.withdrawal_days):,}"],
        ["Total future withdrawals", _fmt_chf(total_withdrawals)],
        ["Monte Carlo paths", f"{NUM_PATHS:,}"],
    ]

    print_table(
        title="Simulation Horizon",
        columns=["Metric", "Value"],
        rows=horizon_rows,
        right_align={"Value"},
    )


# =============================================================================
# Result reporting
# =============================================================================

def print_execution_directive(optimal_results: dict) -> None:
    optimal_contribution_leverage = _get_optimal_contribution_leverage(optimal_results)

    prob_ruin = float(optimal_results["prob_ruin"])
    upper_95_ruin = _binomial_upper_95(prob_ruin, NUM_PATHS)

    today_order = float(optimal_results["optimal_purchase_chf"])
    approximate_net_borrowing = today_order - TODAY_DEPOSIT

    rows = [
        ["Status", _risk_status(prob_ruin)],
        ["Recommended target-asset order today", _fmt_chf(today_order)],
        ["Today's deposit", _fmt_chf(TODAY_DEPOSIT)],
        [
            "Approx. net borrowing from today's order",
            _fmt_chf(approximate_net_borrowing),
        ],
        ["Optimal contribution leverage policy", _fmt_x(optimal_contribution_leverage)],
        [
            "Today pre-contribution portfolio leverage",
            _fmt_x(optimal_results["today_pre_contribution_leverage"]),
        ],
        [
            "Today contribution multiplier applied",
            _fmt_x(optimal_results["today_contribution_multiplier"]),
        ],
        ["Today policy action", str(optimal_results["today_policy_action"])],
        ["Full-leverage cutoff X", _fmt_x(CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX)],
        ["No-invest cutoff Y", _fmt_x(CONTRIBUTION_POLICY_NO_INVEST_MIN)],
        ["Margin-breach probability", _fmt_pct(prob_ruin)],
        ["Margin-breach upper 95% MC bound", _fmt_pct(upper_95_ruin)],
        ["Risk budget", _fmt_pct(MAX_MARGIN_CALL_PROBABILITY)],
        ["Max median leverage observed", _fmt_x(optimal_results["max_median_leverage"])],
        ["Optimizer method", str(optimal_results["optimizer_method"])],
        ["Constraint binding", _fmt_bool(optimal_results["constraint_binding"])],
        [
            "Non-monotonic risk curve",
            _fmt_bool(optimal_results["risk_curve_non_monotonic"]),
        ],
    ]

    print_table(
        title="Execution Directive",
        columns=["Metric", "Value"],
        rows=rows,
        right_align={"Value"},
    )


def _extract_terminal_nav(
    results: dict,
    label: str,
    survivors_only: bool = False,
    drop_invalid: bool = True,
) -> np.ndarray:
    """Extracts terminal NAV values from a simulation result."""
    if "Final_NAV" not in results:
        raise KeyError(
            f"[!] {label} result does not contain 'Final_NAV'. "
            "Make sure engine.simulate(...) returns terminal NAV."
        )

    nav = np.asarray(results["Final_NAV"], dtype=np.float64)

    if survivors_only:
        ruined = np.asarray(
            results.get("Final_Ruined", np.zeros(nav.shape, dtype=bool)),
            dtype=bool,
        )

        if ruined.shape != nav.shape:
            raise ValueError(
                f"[!] {label} Final_Ruined shape does not match Final_NAV shape."
            )

        nav = nav[~ruined]

    if drop_invalid:
        nav = nav[np.isfinite(nav)]

    if nav.size == 0:
        raise ValueError(f"[!] No terminal NAV values available for {label}.")

    return nav


def _nav_summary(nav: np.ndarray) -> dict:
    """Compact terminal NAV summary statistics."""
    return {
        "mean": float(np.mean(nav)),
        "p05": float(np.percentile(nav, 5)),
        "p25": float(np.percentile(nav, 25)),
        "median": float(np.percentile(nav, 50)),
        "p75": float(np.percentile(nav, 75)),
        "p95": float(np.percentile(nav, 95)),
    }


def print_terminal_nav_comparison(
    strategy_results: dict,
    benchmark_results: dict,
) -> None:
    """
    Prints paired terminal NAV comparison.

    Because both simulations use the same MarketSimulator instance, the market
    paths are the same. Therefore strategy NAV minus benchmark NAV is a pathwise
    policy comparison.
    """
    strategy_nav = _extract_terminal_nav(
        strategy_results,
        label="optimized strategy",
        survivors_only=False,
        drop_invalid=False,
    )

    benchmark_nav = _extract_terminal_nav(
        benchmark_results,
        label="no-future-leverage benchmark",
        survivors_only=False,
        drop_invalid=False,
    )

    if strategy_nav.shape != benchmark_nav.shape:
        raise ValueError(
            "[!] Strategy and benchmark terminal NAV arrays have different shapes.\n"
            f"    Strategy:  {strategy_nav.shape}\n"
            f"    Benchmark: {benchmark_nav.shape}"
        )

    valid_pair = np.isfinite(strategy_nav) & np.isfinite(benchmark_nav)

    if not np.any(valid_pair):
        raise ValueError("[!] No valid paired terminal NAV observations available.")

    strategy_nav = strategy_nav[valid_pair]
    benchmark_nav = benchmark_nav[valid_pair]

    nav_diff = strategy_nav - benchmark_nav

    strategy_stats = _nav_summary(strategy_nav)
    benchmark_stats = _nav_summary(benchmark_nav)
    diff_stats = _nav_summary(nav_diff)

    prob_outperform = float(np.mean(nav_diff > 0.0))

    strategy_prob_ruin = float(strategy_results["prob_ruin"])
    benchmark_prob_ruin = float(benchmark_results["prob_ruin"])
    margin_breach_delta = strategy_prob_ruin - benchmark_prob_ruin

    rows = [
        [
            "Mean terminal NAV",
            _fmt_chf(strategy_stats["mean"]),
            _fmt_chf(benchmark_stats["mean"]),
            _fmt_chf(diff_stats["mean"]),
        ],
        [
            "5th percentile NAV / uplift",
            _fmt_chf(strategy_stats["p05"]),
            _fmt_chf(benchmark_stats["p05"]),
            _fmt_chf(diff_stats["p05"]),
        ],
        [
            "25th percentile NAV / uplift",
            _fmt_chf(strategy_stats["p25"]),
            _fmt_chf(benchmark_stats["p25"]),
            _fmt_chf(diff_stats["p25"]),
        ],
        [
            "Median terminal NAV / uplift",
            _fmt_chf(strategy_stats["median"]),
            _fmt_chf(benchmark_stats["median"]),
            _fmt_chf(diff_stats["median"]),
        ],
        [
            "75th percentile NAV / uplift",
            _fmt_chf(strategy_stats["p75"]),
            _fmt_chf(benchmark_stats["p75"]),
            _fmt_chf(diff_stats["p75"]),
        ],
        [
            "95th percentile NAV / uplift",
            _fmt_chf(strategy_stats["p95"]),
            _fmt_chf(benchmark_stats["p95"]),
            _fmt_chf(diff_stats["p95"]),
        ],
        [
            "Margin-breach probability",
            _fmt_pct(strategy_prob_ruin),
            _fmt_pct(benchmark_prob_ruin),
            _fmt_pct(margin_breach_delta),
        ],
        [
            "Margin-breach upper 95% MC bound",
            _fmt_pct(_binomial_upper_95(strategy_prob_ruin, NUM_PATHS)),
            _fmt_pct(_binomial_upper_95(benchmark_prob_ruin, NUM_PATHS)),
            "-",
        ],
        [
            "P(optimized NAV > benchmark NAV)",
            "-",
            "-",
            _fmt_pct(prob_outperform),
        ],
    ]

    print_table(
        title="Terminal NAV Policy Comparison",
        columns=[
            "Metric",
            "Optimized policy",
            "Benchmark: 1.00x future contributions",
            "Pathwise uplift",
        ],
        rows=rows,
        right_align={
            "Optimized policy",
            "Benchmark: 1.00x future contributions",
            "Pathwise uplift",
        },
    )


# =============================================================================
# Plots
# =============================================================================

def plot_diagnostics(sim_results: dict, withdrawal_days: list[int]) -> None:
    """Generates a dual-panel diagnostic visualization."""
    t = sim_results["time_axis"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # --- Top Panel: Gross asset dynamics ---
    cash_paths = sim_results.get("Cash")

    if cash_paths is None:
        v_total = sim_results["V_target"] + sim_results["V_legacy"]
    else:
        v_total = sim_results["V_target"] + sim_results["V_legacy"] + cash_paths

    v_5 = np.percentile(v_total, 5, axis=0)
    v_25 = np.percentile(v_total, 25, axis=0)
    v_50 = np.median(v_total, axis=0)
    v_75 = np.percentile(v_total, 75, axis=0)
    v_95 = np.percentile(v_total, 95, axis=0)

    sample_value_lines = ax1.plot(
        t,
        v_total[0:5].T,
        color="black",
        alpha=0.20,
        linewidth=1,
    )
    sample_value_lines[0].set_label("First 5 gross-asset paths")

    ax1.plot(
        t,
        v_50,
        color="#047857",
        label="Median gross assets",
        linewidth=2,
    )
    ax1.fill_between(
        t,
        v_25,
        v_75,
        color="#047857",
        alpha=0.35,
        label="50% interval",
    )
    ax1.fill_between(
        t,
        v_5,
        v_95,
        color="#047857",
        alpha=0.15,
        label="90% interval",
    )

    ax1.set_title(
        "Projected Gross Assets Including Cash",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Gross Assets incl. Cash (CHF)")
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    # --- Bottom Panel: Leverage dynamics ---
    leverage_paths = sim_results["Leverage"]

    lev_5 = np.nanpercentile(leverage_paths, 5, axis=0)
    lev_25 = np.nanpercentile(leverage_paths, 25, axis=0)
    lev_50 = np.nanmedian(leverage_paths, axis=0)
    lev_75 = np.nanpercentile(leverage_paths, 75, axis=0)
    lev_95 = np.nanpercentile(leverage_paths, 95, axis=0)

    ax2.fill_between(
        t,
        lev_25,
        lev_75,
        color="#1E3A8A",
        alpha=0.35,
        label="50% interval",
    )
    ax2.fill_between(
        t,
        lev_5,
        lev_95,
        color="#1E3A8A",
        alpha=0.15,
        label="90% interval",
    )

    ax2.plot(
        t,
        leverage_paths[0:5].T,
        color="black",
        alpha=0.25,
        linewidth=1,
    )

    ax2.plot(
        t,
        lev_50,
        color="#1E3A8A",
        label="Median leverage",
        linewidth=2,
    )

    ax2.axhline(
        y=CONTRIBUTION_POLICY_FULL_LEVERAGE_MAX,
        color="#D97706",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label="X: full contribution-leverage cutoff",
    )

    ax2.axhline(
        y=CONTRIBUTION_POLICY_NO_INVEST_MIN,
        color="#7F1D1D",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label="Y: no-invest cutoff",
    )

    for wd in withdrawal_days:
        ax2.axvline(x=wd, color="#B91C1C", linestyle="--", alpha=0.7)

    if withdrawal_days:
        ax2.axvline(
            x=-100,
            color="#B91C1C",
            linestyle="--",
            alpha=0.7,
            label="Liability withdrawal",
        )

    ax2.set_xlim(0, t[-1])
    ax2.set_title(
        "Simulated Total Portfolio Leverage",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Total Leverage (x)")
    ax2.set_xlabel("Simulation Horizon (Days)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_terminal_nav_distribution(
    strategy_results: dict,
    benchmark_results: dict,
    bins: int = 90,
    trim_percentiles: tuple[float, float] = (0.5, 99.5),
    survivors_only: bool = False,
) -> None:
    """
    Overlays terminal NAV distributions for:
      1. optimized contribution-leverage policy;
      2. no-future-leverage benchmark.
    """
    strategy_nav = _extract_terminal_nav(
        strategy_results,
        label="optimized strategy",
        survivors_only=survivors_only,
    )

    benchmark_nav = _extract_terminal_nav(
        benchmark_results,
        label="no-future-leverage benchmark",
        survivors_only=survivors_only,
    )

    combined = np.concatenate([strategy_nav, benchmark_nav])
    x_low, x_high = np.percentile(combined, trim_percentiles)

    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low >= x_high:
        x_low = float(np.min(combined))
        x_high = float(np.max(combined))

    bin_edges = np.linspace(x_low, x_high, bins + 1)

    strategy_median = float(np.median(strategy_nav))
    benchmark_median = float(np.median(benchmark_nav))
    median_uplift = strategy_median - benchmark_median

    optimal_contribution_leverage = _get_optimal_contribution_leverage(strategy_results)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        benchmark_nav,
        bins=bin_edges,
        density=True,
        alpha=0.45,
        color="#64748B",
        label="Benchmark: future contributions at 1.00x",
    )

    ax.hist(
        strategy_nav,
        bins=bin_edges,
        density=True,
        alpha=0.45,
        color="#047857",
        label=(
            f"Optimized policy: {optimal_contribution_leverage:.2f}x "
            "when guardrails allow"
        ),
    )

    ax.axvline(
        benchmark_median,
        color="#334155",
        linestyle="--",
        linewidth=2,
        label=f"Benchmark median: {benchmark_median:,.0f} CHF",
    )

    ax.axvline(
        strategy_median,
        color="#065F46",
        linestyle="--",
        linewidth=2,
        label=f"Optimized median: {strategy_median:,.0f} CHF",
    )

    ax.set_title(
        "Terminal NAV Distribution: Optimized Policy vs No-Future-Leverage Benchmark",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Terminal NAV / Equity (CHF)")
    ax.set_ylabel("Density")
    ax.set_xlim(x_low, x_high)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    annotation = (
        f"Median uplift: {median_uplift:,.0f} CHF\n"
        f"Optimized breach prob.: {strategy_results['prob_ruin']:.2%}\n"
        f"Benchmark breach prob.: {benchmark_results['prob_ruin']:.2%}\n"
        f"Display range: p{trim_percentiles[0]:g} to p{trim_percentiles[1]:g}"
    )

    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main orchestration
# =============================================================================

def main() -> None:
    print_banner(
        "LDI Optimization Engine",
        "Contribution-leverage policy simulator",
    )

    data_engine = DataEngine()
    data_engine.fetch_data()

    state = data_engine.build_current_state()
    params = data_engine.estimate_parameters()

    print_initial_state(state, params)
    print_parameter_summary(params)

    if not WITHDRAWAL_SCHEDULE:
        raise ValueError("[!] WITHDRAWAL_SCHEDULE is empty. Cannot infer simulation horizon.")

    last_withdrawal_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE)

    final_date = last_withdrawal_date + relativedelta(
        months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS,
        days=POST_LAST_WITHDRAWAL_BUFFER_DAYS,
    )

    simulator = MarketSimulator(state, params, final_date)

    print_simulation_horizon(
        simulator=simulator,
        final_date=final_date,
        last_withdrawal_date=last_withdrawal_date,
    )

    optimizer = MarginOptimizer(simulator)
    optimal_results = optimizer.optimize()

    console.print(
        "\n[bold cyan][*][/bold cyan] Running no-future-leverage benchmark "
        "on the same market paths..."
    )

    no_leverage_results = simulator.simulate(
        1.0,
        store_paths=False,
        store_final_nav=True,
        contribution_policy_mode="always_unlevered",
    )

    print_execution_directive(optimal_results)

    print_terminal_nav_comparison(
        strategy_results=optimal_results,
        benchmark_results=no_leverage_results,
    )

    plot_diagnostics(
        sim_results=optimal_results,
        withdrawal_days=simulator.withdrawal_days,
    )

    plot_terminal_nav_distribution(
        strategy_results=optimal_results,
        benchmark_results=no_leverage_results,
        survivors_only=False,
    )


if __name__ == "__main__":
    main()