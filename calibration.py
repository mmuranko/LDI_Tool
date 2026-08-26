import os
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytensor.tensor.linalg as ptlinalg
import arviz as az
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console

from config import DRIFT_CAP

console = Console()
warnings.filterwarnings("ignore", category=RuntimeWarning)

class BayesianFactorCalibrator:
    def __init__(self, raw_data: pd.DataFrame, num_factors: int = 5):
        self.num_factors = num_factors
        
        # 1. Compute Base-Currency Log Returns
        returns = np.log(raw_data / raw_data.shift(1)).dropna()
        self.asset_names = returns.columns.tolist()
        
        # 2. Standardize & Extract PCA Factors
        self.scaler = StandardScaler()
        scaled_returns = self.scaler.fit_transform(returns)
        
        self.pca = PCA(n_components=num_factors)
        self.factor_returns = self.pca.fit_transform(scaled_returns)
        self.loadings = self.pca.components_.T  # Shape: (N_assets, N_factors)
        
        # 3. Calculate Idiosyncratic Residual Variances (Preserves 100% Volatility)
        explained_var_per_asset = np.sum(self.loadings ** 2, axis=1)
        self.residual_scales = np.sqrt(np.maximum(1e-4, 1.0 - explained_var_per_asset)).astype(np.float32)
        
        # 4. Convert Intuitive Simple Return Cap (DRIFT_CAP) to Daily Log Drift
        raw_annual_log_means = self.scaler.mean_ * 252.0
        annual_sigmas = self.scaler.scale_ * np.sqrt(252.0)
        raw_annual_simple = np.exp(raw_annual_log_means + 0.5 * (annual_sigmas ** 2)) - 1.0
        
        capped_annual_log = np.where(
            raw_annual_simple > DRIFT_CAP,
            np.log(1.0 + DRIFT_CAP) - 0.5 * (annual_sigmas ** 2),
            raw_annual_log_means
        )
        
        self.asset_means = (capped_annual_log / 252.0).astype(np.float32)
        self.asset_scales = self.scaler.scale_.astype(np.float32)
        self.T = len(self.factor_returns)

    def fit(self, draws: int = 500, tune: int = 600, cores: int = 8, chains: int | None = None) -> dict:
        if chains is None:
            chains = cores
        console.print(f"Configuring Bayesian MCMC for {self.num_factors} factors ({chains} parallel chains)...")
        N = self.num_factors
        
        with pm.Model() as model:
            # Transition Matrix Priors
            p0 = pm.Dirichlet("p0", a=np.array([2.0, 1.0]))
            p1 = pm.Dirichlet("p1", a=np.array([1.0, 2.0]))
            P = pt.stack([p0, p1])
            
            # FACTOR CHOLESKY MATRICES: Unpack Index 0 (chol)
            chol_0_raw, _, _ = pm.LKJCholeskyCov("lkj_0", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_0 = pm.Deterministic("chol_0", chol_0_raw)
            
            chol_1_raw, _, _ = pm.LKJCholeskyCov("lkj_1", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_1 = pm.Deterministic("chol_1", chol_1_raw)
            
            def step_forward(obs_t, prev_gamma, P_mat, c0, c1):
                gamma_pred = pt.dot(prev_gamma, P_mat)
                
                def mvn_logp(y, chol):
                    k = y.shape[0]
                    log_det = pt.sum(pt.log(pt.diag(chol)))
                    z = ptlinalg.solve(chol, y)
                    mahalanobis = pt.sum(z ** 2)
                    return -0.5 * (k * np.log(2 * np.pi) + 2.0 * log_det + mahalanobis)
                
                ll_0 = mvn_logp(obs_t, c0)
                ll_1 = mvn_logp(obs_t, c1)
                
                log_gamma_update = pt.stack([pt.log(gamma_pred[0]) + ll_0, pt.log(gamma_pred[1]) + ll_1])
                max_log = pt.max(log_gamma_update)
                marginal_ll = max_log + pt.log(pt.sum(pt.exp(log_gamma_update - max_log)))
                
                next_gamma = pt.exp(log_gamma_update - marginal_ll)
                return next_gamma, marginal_ll

            pi_0 = pt.constant(np.array([0.5, 0.5]))
            
            results, _ = pytensor.scan(
                fn=step_forward,
                sequences=[self.factor_returns],
                outputs_info=[pi_0, None],
                non_sequences=[P, chol_0, chol_1],
                strict=True
            )
            gammas, log_likes = results
            pm.Potential("hmm_loglike", pt.sum(log_likes))
            
            trace = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=chains, 
                target_accept=0.85,
                return_inferencedata=True, 
                progressbar=True
            )

            summary = az.summary(trace)
            max_rhat = float(summary["r_hat"].max())
            console.print(f"MCMC Convergence Summary (Max R-hat): [bold]{max_rhat:.4f}[/bold]")
        
        # Canonical Per-Draw Sorting (Eliminates Cross-Chain Label Switching)
        posteriors = trace.posterior
        p0_vals = posteriors["p0"].values.copy()
        p1_vals = posteriors["p1"].values.copy()
        c0_vals = posteriors["chol_0"].values.copy()
        c1_vals = posteriors["chol_1"].values.copy()

        num_chains, num_draws = c0_vals.shape[0], c0_vals.shape[1]
        for c in range(num_chains):
            for d in range(num_draws):
                v0 = np.trace(c0_vals[c, d] @ c0_vals[c, d].T)
                v1 = np.trace(c1_vals[c, d] @ c1_vals[c, d].T)
                if v0 > v1:
                    tmp_c = c0_vals[c, d].copy()
                    c0_vals[c, d] = c1_vals[c, d]
                    c1_vals[c, d] = tmp_c
                    tmp_p0 = p0_vals[c, d].copy()
                    p0_vals[c, d] = p1_vals[c, d][[1, 0]]
                    p1_vals[c, d] = tmp_p0[[1, 0]]

        P_mean = np.mean(np.stack([p0_vals, p1_vals], axis=-2), axis=(0, 1))
        chol_0_mean = np.mean(c0_vals, axis=(0, 1))
        chol_1_mean = np.mean(c1_vals, axis=(0, 1))

        console.print("✓ [green]Bayesian HMM Factor Calibration complete.[/green]")
        return {
            "P": np.asarray(P_mean, dtype=np.float32),
            "chol_0": np.asarray(chol_0_mean, dtype=np.float32),
            "chol_1": np.asarray(chol_1_mean, dtype=np.float32),
            "pca_loadings": np.asarray(self.loadings, dtype=np.float32),
            "residual_scales": self.residual_scales,
            "asset_means": self.asset_means,
            "asset_scales": self.asset_scales,
            "asset_names": self.asset_names
        }