import matplotlib.pyplot as plt
import numpy as np
from data_fetcher import DataEngine
from engine import MarketSimulator
from optimizer import MarginOptimizer
from dateutil.relativedelta import relativedelta
from config import (
    WITHDRAWAL_SCHEDULE,
    TARGET_ASSET,
    CURRENT_DATE,
    POST_LAST_WITHDRAWAL_BUFFER_MONTHS,
    POST_LAST_WITHDRAWAL_BUFFER_DAYS
)

def plot_diagnostics(sim_results: dict, withdrawal_days: list):
    """Generates a professional dual-panel visualization."""
    t = sim_results["time_axis"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    # --- Top Panel: Total Portfolio Value Dynamics ---
    # Aggregate both asset matrices into a single Total Value matrix
    v_total = sim_results["V_target"] + sim_results["V_legacy"]
    
    # Calculate confidence percentiles for the Total Value
    v_5 = np.percentile(v_total, 5, axis=0)
    v_25 = np.percentile(v_total, 25, axis=0)
    v_50 = np.median(v_total, axis=0)
    v_75 = np.percentile(v_total, 75, axis=0)
    v_95 = np.percentile(v_total, 95, axis=0)
    
    # Plot the first 5 individual random paths to show the actual chaotic volatility
    ax1.plot(t, v_total[0:5].T, color='black', alpha=0.2, linewidth=1)

    # Plot the asset cones (using a distinct green to separate from the blue leverage plot)
    ax1.plot(t, v_50, color='#047857', label='Median Total Value', linewidth=2)
    ax1.fill_between(t, v_25, v_75, color='#047857', alpha=0.35, label='50% CI')
    ax1.fill_between(t, v_5, v_95, color='#047857', alpha=0.15, label='90% CI')
    
    ax1.set_title("Projected Gross Asset Value", loc='left', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Gross Asset Value (CHF)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')
    
    # --- Bottom Panel: Leverage Ratio Dynamics ---
    lev_5 = np.nanpercentile(sim_results["Leverage"], 5, axis=0)
    lev_25 = np.nanpercentile(sim_results["Leverage"], 25, axis=0)
    lev_50 = np.nanmedian(sim_results["Leverage"], axis=0)
    lev_75 = np.nanpercentile(sim_results["Leverage"], 75, axis=0)
    lev_95 = np.nanpercentile(sim_results["Leverage"], 95, axis=0)
    
    ax2.plot(t, lev_50, color='#1E3A8A', label='Median Leverage', linewidth=2)
    ax2.fill_between(t, lev_25, lev_75, color='#1E3A8A', alpha=0.35, label='50% CI')
    ax2.fill_between(t, lev_5, lev_95, color='#1E3A8A', alpha=0.15, label='90% CI')
    
    # Mark liabilities
    for wd in withdrawal_days:
        ax2.axvline(x=wd, color='#B91C1C', linestyle='--', alpha=0.7)
    
    ax2.axvline(x=-100, color='#B91C1C', linestyle='--', alpha=0.7, label='Liability Drawdown')
    
    ax2.set_xlim(0, t[-1])
    ax2.set_title("Simulated Leverage Ratio Dynamics", loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Total Leverage (x)")
    ax2.set_xlabel("Simulation Horizon (Days)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def main():
    print("===================================================")
    print("      LDI OPTIMIZATION ENGINE INITIALIZING         ")
    print("===================================================")
    
    # 1. Pipeline Initiation
    data_engine = DataEngine()
    data_engine.fetch_data()
    
    state = data_engine.build_current_state()
    params = data_engine.estimate_parameters()
    
    print("\n[*] Initial Balance Sheet:")
    print(f"    Target Base: {state['v_target_0']:,.2f} CHF")
    print(f"    Legacy Base: {state['v_legacy_0']:,.2f} CHF")
    print("[*] Drift Estimates:")
    print(f"    Target:       raw = {params['mu_target_raw']:.2%}, used/capped = {params['mu_target']:.2%}")
    print(f"    Legacy local: raw = {params['mu_legacy_loc_raw']:.2%}, used/capped = {params['mu_legacy_loc']:.2%}")
    print(f"    Legacy FX:    raw = {params['mu_fx_raw']:.2%}, used assumption = {params['mu_fx']:.2%}")
    print(f"[*] Correlation Engine: 3x3 Cholesky Matrix Loaded")
    
    # 2. Setup Simulator Engine
    if not WITHDRAWAL_SCHEDULE:
        raise ValueError("[!] WITHDRAWAL_SCHEDULE is empty. Cannot infer simulation horizon.")

    last_withdrawal_date = max(w["date"] for w in WITHDRAWAL_SCHEDULE)

    final_date = last_withdrawal_date + relativedelta(
        months=POST_LAST_WITHDRAWAL_BUFFER_MONTHS,
        days=POST_LAST_WITHDRAWAL_BUFFER_DAYS
    )

    simulator = MarketSimulator(state, params, final_date)

    print(f"[*] Simulation Horizon: {CURRENT_DATE} to {final_date} ({simulator.days} calendar days)")
    
    # 3. Run Optmizer
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
    
    # 4. Diagnostics Generation
    plot_diagnostics(optimal_results, simulator.withdrawal_days)

if __name__ == "__main__":
    main()