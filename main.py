import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta

from config import (
    CURRENT_DATE,
    POST_LAST_WITHDRAWAL_BUFFER_DAYS,
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS,
    TARGET_ASSET,
    WITHDRAWAL_SCHEDULE,
)
from data_fetcher import DataEngine
from engine import MarketSimulator
from optimizer import MarginOptimizer


def plot_diagnostics(sim_results: dict, withdrawal_days: list):
    """Generates a dual-panel diagnostic visualization."""
    t = sim_results["time_axis"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # --- Top Panel: Total Portfolio Value Dynamics ---
    v_total = sim_results["V_target"] + sim_results["V_legacy"]

    v_5 = np.percentile(v_total, 5, axis=0)
    v_25 = np.percentile(v_total, 25, axis=0)
    v_50 = np.median(v_total, axis=0)
    v_75 = np.percentile(v_total, 75, axis=0)
    v_95 = np.percentile(v_total, 95, axis=0)

    sample_value_lines = ax1.plot(t, v_total[0:5].T, color="black", alpha=0.20, linewidth=1)
    sample_value_lines[0].set_label("First 5 value paths")

    ax1.plot(t, v_50, color="#047857", label="Median Total Value", linewidth=2)
    ax1.fill_between(t, v_25, v_75, color="#047857", alpha=0.35, label="50% CI")
    ax1.fill_between(t, v_5, v_95, color="#047857", alpha=0.15, label="90% CI")

    ax1.set_title("Projected Gross Asset Value", loc="left", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Gross Asset Value (CHF)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    # --- Bottom Panel: Leverage Ratio Dynamics ---
    leverage_paths = sim_results["Leverage"]

    lev_5 = np.nanpercentile(leverage_paths, 5, axis=0)
    lev_25 = np.nanpercentile(leverage_paths, 25, axis=0)
    lev_50 = np.nanmedian(leverage_paths, axis=0)
    lev_75 = np.nanpercentile(leverage_paths, 75, axis=0)
    lev_95 = np.nanpercentile(leverage_paths, 95, axis=0)

    ax2.fill_between(t, lev_25, lev_75, color="#1E3A8A", alpha=0.35, label="50% CI")
    ax2.fill_between(t, lev_5, lev_95, color="#1E3A8A", alpha=0.15, label="90% CI")

    ax2.plot(t, leverage_paths[0:5].T, color='black', alpha=0.25, linewidth=1)

    ax2.plot(t, lev_50, color="#1E3A8A", label="Median Leverage", linewidth=2)

    for wd in withdrawal_days:
        ax2.axvline(x=wd, color="#B91C1C", linestyle="--", alpha=0.7)

    ax2.axvline(x=-100, color="#B91C1C", linestyle="--", alpha=0.7, label="Liability Drawdown")

    ax2.set_xlim(0, t[-1])
    ax2.set_title("Simulated Leverage Ratio Dynamics", loc="left", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Total Leverage (x)")
    ax2.set_xlabel("Simulation Horizon (Days)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def print_initial_state(state: dict, params: dict) -> None:
    target = params["target_factor"]

    print("\n[*] Initial Balance Sheet:")
    print(f"    Target Base: {state['v_target_0']:,.2f} CHF ({target['currency']})")
    print(f"    Legacy Base: {state['v_legacy_0']:,.2f} CHF")

    print("[*] Legacy Assets:")
    for ticker in state["legacy_asset_order"]:
        asset_state = state["legacy_assets"][ticker]
        param_state = params["legacy_assets"].get(ticker, {})
        sigma = param_state.get("sigma", float("nan"))
        print(
            f"    {ticker:<14} {asset_state['v0']:>12,.2f} CHF | "
            f"{asset_state['currency']:<3} | m = {asset_state['m']:.2%} | vol = {sigma:.2%}"
        )

    if not state["legacy_asset_order"]:
        print("    None")

    print("[*] Currency Exposure Summary:")
    for ccy, bucket in sorted(state["legacy_by_currency"].items()):
        print(
            f"    {ccy:<3}: {bucket['v0']:>12,.2f} CHF | "
            f"m = {bucket['m']:.2%} | {', '.join(bucket['tickers'])}"
        )


def print_parameter_summary(params: dict) -> None:
    target = params["target_factor"]

    print("[*] Drift / Volatility Estimates:")
    print(
        f"    Target local {target['ticker']} ({target['currency']}): "
        f"raw drift = {target['mu_raw']:.2%}, used = {target['mu']:.2%}, "
        f"vol = {target['sigma']:.2%}"
    )

    for ticker, info in params["legacy_assets"].items():
        print(
            f"    Legacy asset {ticker:<14} ({info['currency']}): "
            f"raw drift = {info['mu_raw']:.2%}, used = {info['mu']:.2%}, "
            f"vol = {info['sigma']:.2%}"
        )

    if not params["legacy_assets"]:
        print("    Legacy assets: none")

    for ccy, info in params["fx_factors"].items():
        print(
            f"    FX {ccy}/{params['base_currency']}: "
            f"raw drift = {info['mu_raw']:.2%}, used = {info['mu']:.2%}, "
            f"vol = {info['sigma']:.2%}"
        )

    print(
        f"[*] Correlation Engine: {len(params['factor_names'])}x{len(params['factor_names'])} "
        f"Cholesky matrix loaded from {params['aligned_observations']} aligned observations"
    )
    print(f"    Factor order: {', '.join(params['factor_names'])}")


def main():
    print("===================================================")
    print("      LDI OPTIMIZATION ENGINE INITIALIZING         ")
    print("===================================================")

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

    print(f"[*] Simulation Horizon: {CURRENT_DATE} to {final_date} ({simulator.days} calendar days)")

    optimizer = MarginOptimizer(simulator)
    optimal_results = optimizer.optimize()

    print("\n===================================================")
    print("                 EXECUTION DIRECTIVE               ")
    print("===================================================")
    print(f"Optimal Target Asset Order:    {optimal_results['optimal_purchase_chf']:,.2f} CHF")
    print(f"Optimal Unified Leverage L*:   {optimal_results['optimal_target_leverage']:.2f}x")
    print(f"Path Ruin Probability (EL<0):  {optimal_results['prob_ruin']:.2%}")
    print(f"Optimizer Method:              {optimal_results['optimizer_method']}")
    print(f"Constraint Binding:            {optimal_results['constraint_binding']}")
    print(f"Non-Monotonic Risk Curve:      {optimal_results['risk_curve_non_monotonic']}")
    print("===================================================")

    plot_diagnostics(optimal_results, simulator.withdrawal_days)


if __name__ == "__main__":
    main()
