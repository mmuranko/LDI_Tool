import numpy as np
import pandas as pd
import datetime
from config import (CURRENT_DATE, CURRENT_DEBT, DEFAULT_MONTHLY_DEPOSIT_2026, DEFAULT_MONTHLY_DEPOSIT_FUTURE,
                    MARGIN_INTEREST_RATE, WITHDRAWAL_SCHEDULE, NUM_PATHS,
                    JUMP_FREQUENCY_PER_YEAR, JUMP_MEAN_SIZE, JUMP_VOLATILITY, TODAY_DEPOSIT, 
                    HESTON_KAPPA, HESTON_XI, HESTON_RHO)

class MarketSimulator:
    """Vectorized Stochastic Asset Engine with Merton Jump-Diffusion."""
    
    def __init__(self, state: dict, params: dict, end_date: datetime.date):
        self.state = state
        self.params = params
        
        self.days = (end_date - CURRENT_DATE).days
        self.dt = 1 / 365.0  
        
        # --- Pre-allocate 3D Tensor for Cholesky Decomposition ---
        np.random.seed(42)
        # Shape: (3 Factors, NUM_PATHS, days)
        self.Z = np.random.normal(0, np.sqrt(self.dt), (3, NUM_PATHS, self.days))

        # --- Independent Noise for Heston Variance Paths ---
        self.Zv_acwi = np.random.normal(0, np.sqrt(self.dt), (NUM_PATHS, self.days))
        self.Zv_leg = np.random.normal(0, np.sqrt(self.dt), (NUM_PATHS, self.days))
        
        # --- Pre-allocate Jump-Diffusion Random Variables (Fat Tails) ---
        # 1. Poisson process: Does a jump happen on this specific day? (0 or 1)
        jump_lambda = JUMP_FREQUENCY_PER_YEAR * self.dt
        self.jump_occurrences = np.random.poisson(jump_lambda, (NUM_PATHS, self.days))
        
        # 2. Jump sizes: If a jump happens, how severe is it?
        self.jump_sizes = np.random.normal(JUMP_MEAN_SIZE, JUMP_VOLATILITY, (NUM_PATHS, self.days))
        
        # Identify schedule indices
        self.withdrawal_days = [(w["date"] - CURRENT_DATE).days for w in WITHDRAWAL_SCHEDULE]
        self.withdrawal_amounts = [w["amount"] for w in WITHDRAWAL_SCHEDULE]

        # --- Institutional Calendar: Last Business Day of the Month ---
        business_month_ends = pd.date_range(start=CURRENT_DATE, end=end_date, freq='BME').date
        
        self.deposit_days = []
        self.deposit_amounts = []
        
        for bme in business_month_ends:
            days_from_start = (bme - CURRENT_DATE).days
            if 0 < days_from_start < self.days:
                self.deposit_days.append(days_from_start)
                # Assign the correct deposit amount based on the calendar year
                if bme.year == 2026:
                    self.deposit_amounts.append(DEFAULT_MONTHLY_DEPOSIT_2026)
                else:
                    self.deposit_amounts.append(DEFAULT_MONTHLY_DEPOSIT_FUTURE)

    def simulate(self, target_leverage: float) -> dict:
        # Matrices for Margin calculations
        V_acwi = np.zeros((NUM_PATHS, self.days))
        V_legacy = np.zeros((NUM_PATHS, self.days))
        Debt = np.zeros((NUM_PATHS, self.days))
        
        # New: Isolated index paths for the legacy portfolio
        Local_Index = np.ones((NUM_PATHS, self.days))
        FX_Index = np.ones((NUM_PATHS, self.days))
        
        # --- Day 0 Mechanics ---
        V_0 = self.state["v_acwi_0"] + self.state["v_legacy_0"]
        E_0 = V_0 - CURRENT_DEBT + TODAY_DEPOSIT
        target_assets = E_0 * target_leverage
        purchase_amount = max(0.0, target_assets - V_0)
        
        V_acwi[:, 0] = self.state["v_acwi_0"] + purchase_amount
        V_legacy[:, 0] = self.state["v_legacy_0"]
        Debt[:, 0] = CURRENT_DEBT + purchase_amount - TODAY_DEPOSIT
        
        # --- Multi-Factor Cholesky Decomposition ---
        L = np.linalg.cholesky(self.params["corr_matrix"])
        
        # Einstein Summation to rapidly multiply the 3x3 Cholesky matrix across the (3, Paths, Days) tensor
        # This correlates the Wiener processes for all 3 risk factors simultaneously
        dW = np.einsum('ij,jkl->ikl', L, self.Z)
        
        dW_acwi = dW[0]
        dW_leg_loc = dW[1]
        dW_fx = dW[2]
        
        # --- Heston Setup: Initial Variance & Long-Term Mean (Theta) ---
        theta_acwi = self.params["sigma_acwi"]**2
        theta_leg = self.params["sigma_legacy_loc"]**2
        
        # Initialize Variance Paths matrix
        V_vol_acwi = np.full((NUM_PATHS, self.days), theta_acwi)
        V_vol_leg = np.full((NUM_PATHS, self.days), theta_leg)
        
        expected_jump = JUMP_FREQUENCY_PER_YEAR * (np.exp(JUMP_MEAN_SIZE + 0.5 * JUMP_VOLATILITY**2) - 1)
        daily_rate = MARGIN_INTEREST_RATE / 365.0
        jump_multipliers = np.exp(self.jump_occurrences * self.jump_sizes)
        
        # Pull Heston constants locally for faster loop execution
        kappa = HESTON_KAPPA
        xi = HESTON_XI
        rho_sv = HESTON_RHO
        rho_sv_comp = np.sqrt(1 - rho_sv**2)
        
        # --- The Main Vectorized Loop (Heston + Jump Diffusion) ---
        for t in range(1, self.days):
            
            # 1. Apply Full Truncation to previous variance (prevents negative variance errors)
            v_prev_acwi = np.maximum(V_vol_acwi[:, t-1], 0)
            v_prev_leg = np.maximum(V_vol_leg[:, t-1], 0)
            
            # 2. Correlate Variance Noise to Asset Noise (The Leverage Effect)
            # This ensures that if the asset crashes this turn (dW is very negative), volatility spikes!
            dW_v_acwi = rho_sv * dW_acwi[:, t] + rho_sv_comp * self.Zv_acwi[:, t]
            dW_v_leg = rho_sv * dW_leg_loc[:, t] + rho_sv_comp * self.Zv_leg[:, t]
            
            # 3. Step Forward the Variance Paths (Euler-Maruyama)
            V_vol_acwi[:, t] = V_vol_acwi[:, t-1] + kappa * (theta_acwi - v_prev_acwi) * self.dt + xi * np.sqrt(v_prev_acwi) * dW_v_acwi
            V_vol_leg[:, t] = V_vol_leg[:, t-1] + kappa * (theta_leg - v_prev_leg) * self.dt + xi * np.sqrt(v_prev_leg) * dW_v_leg
            
            # 4. Extract instantaneous volatility for this exact day
            inst_vol_acwi = np.sqrt(np.maximum(V_vol_acwi[:, t], 0))
            inst_vol_leg = np.sqrt(np.maximum(V_vol_leg[:, t], 0))
            
            # 5. Calculate Dynamic Drift (drift changes daily because 0.5 * sigma^2 is no longer constant)
            drift_acwi = (self.params["mu_acwi"] - expected_jump - 0.5 * inst_vol_acwi**2) * self.dt
            drift_leg = (self.params["mu_legacy_loc"] - expected_jump - 0.5 * inst_vol_leg**2) * self.dt
            drift_fx = (self.params["mu_fx"] - 0.5 * self.params["sigma_fx"]**2) * self.dt
            
            # 6. Step Forward Asset Paths using the Dynamic Volatility
            V_acwi[:, t] = V_acwi[:, t-1] * np.exp(drift_acwi + inst_vol_acwi * dW_acwi[:, t]) * jump_multipliers[:, t]
            Local_Index[:, t] = Local_Index[:, t-1] * np.exp(drift_leg + inst_vol_leg * dW_leg_loc[:, t]) * jump_multipliers[:, t]
            FX_Index[:, t] = FX_Index[:, t-1] * np.exp(drift_fx + self.params["sigma_fx"] * dW_fx[:, t]) # FX remains standard GBM
            
            # Recombine and Handle Debt
            V_legacy[:, t] = self.state["v_legacy_0"] * Local_Index[:, t] * FX_Index[:, t]
            Debt[:, t] = Debt[:, t-1] * (1 + daily_rate)
            
            # --- Future Interventions: Deploy at Target Leverage ---
            if t in self.deposit_days:
                # Find which deposit number this is
                idx = self.deposit_days.index(t)
                base_deposit = self.deposit_amounts[idx]
                
                # Apply the target leverage to the specific month's cash injection
                leveraged_purchase = base_deposit * target_leverage
                new_debt = leveraged_purchase - base_deposit
                
                V_acwi[:, t] += leveraged_purchase
                Debt[:, t] += new_debt
                
            if t in self.withdrawal_days:
                idx = self.withdrawal_days.index(t)
                Debt[:, t] += self.withdrawal_amounts[idx]
        
        Equity = (V_acwi + V_legacy) - Debt
        Maintenance_Margin = (self.state["m_acwi"] * V_acwi) + (self.state["m_legacy"] * V_legacy)
        Excess_Liquidity = Equity - Maintenance_Margin
        Leverage = (V_acwi + V_legacy) / np.maximum(Equity, 1e-6)
        
        ruined_paths = np.any(Excess_Liquidity < 0, axis=1)
        prob_ruin = np.sum(ruined_paths) / NUM_PATHS
        
        median_leverage_path = np.median(Leverage, axis=0)
        max_median_leverage = np.max(median_leverage_path)
        
        return {
            "prob_ruin": prob_ruin,
            "max_median_leverage": max_median_leverage,
            "V_acwi": V_acwi,
            "V_legacy": V_legacy,
            "Leverage": Leverage,
            "time_axis": np.arange(self.days),
            "optimal_purchase_chf": purchase_amount # Pass this back out for logging
        }