import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytensor.tensor.linalg as ptlinalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console

console = Console()

class BayesianFactorCalibrator:
    def __init__(self, raw_data: pd.DataFrame, num_factors: int = 5):
        self.num_factors = num_factors
        
        # 1. Compute Log Returns (raw_data is already converted to BASE_CURRENCY)
        returns = np.log(raw_data / raw_data.shift(1)).dropna()
        self.asset_names = returns.columns.tolist()
        
        # 2. Standardize & Extract PCA Factors
        self.scaler = StandardScaler()
        scaled_returns = self.scaler.fit_transform(returns)
        
        self.pca = PCA(n_components=num_factors)
        self.factor_returns = self.pca.fit_transform(scaled_returns)
        self.loadings = self.pca.components_.T  # Shape: (N_assets, N_factors)
        
        self.asset_means = self.scaler.mean_
        self.asset_scales = self.scaler.scale_
        self.T = len(self.factor_returns)

    def fit(self, draws: int = 1000, tune: int = 1000, cores: int = 2) -> dict:
        console.print(f"Configuring Bayesian MCMC for {self.num_factors} factors...")
        N = self.num_factors
        
        with pm.Model() as model:
            # Transition Matrix (2 Regimes) using Dirichlet Priors
            p0 = pm.Dirichlet("p0", a=np.ones(2))
            p1 = pm.Dirichlet("p1", a=np.ones(2))
            P = pt.stack([p0, p1])
            
            # Factor Covariance Matrices using LKJ Cholesky Priors
            # LKJ returns (packed, cholesky_matrix, cov_matrix). We safely extract index 1.
            _, chol_0_raw, _ = pm.LKJCholeskyCov("lkj_0", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_0 = pm.Deterministic("chol_0", chol_0_raw)
            
            _, chol_1_raw, _ = pm.LKJCholeskyCov("lkj_1", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_1 = pm.Deterministic("chol_1", chol_1_raw)
            
            # Forward Algorithm to marginalize hidden states for NUTS sampler
            def step_forward(obs_t, prev_gamma, P_mat, c0, c1):
                gamma_pred = pt.dot(prev_gamma, P_mat)
                
                # --- MANUAL MVN LOG-DENSITY ---
                # Bypasses PyMC's CheckParameterValue which crashes on NaN dummy values during scan shape inference
                def mvn_logp(y, chol):
                    k = y.shape[0]
                    # Log-determinant of Cholesky is the sum of the log of the diagonal
                    log_det = pt.sum(pt.log(pt.diag(chol)))
                    # Mahalanobis distance: solve L * z = y for z
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
            
            # Accumulate the log-likelihood of the HMM sequence
            pm.Potential("hmm_loglike", pt.sum(log_likes))
            
            console.print(f"[dim]Starting NUTS Sampler (Draws: {draws}, Tune: {tune})...[/dim]")
            trace = pm.sample(draws=draws, tune=tune, cores=cores, return_inferencedata=True, progressbar=True)
            
        # Extract Posterior Means
        posteriors = trace.posterior
        P_mean = np.mean(np.stack([posteriors["p0"].values, posteriors["p1"].values], axis=-2), axis=(0, 1))
        chol_0_mean = np.mean(posteriors["chol_0"].values, axis=(0, 1))
        chol_1_mean = np.mean(posteriors["chol_1"].values, axis=(0, 1))
        
        # Enforce structural consistency (Regime 1 is defined as the high-volatility/stress regime)
        vol_0 = np.trace(chol_0_mean @ chol_0_mean.T)
        vol_1 = np.trace(chol_1_mean @ chol_1_mean.T)
        
        if vol_0 > vol_1:
            chol_0_mean, chol_1_mean = chol_1_mean, chol_0_mean
            P_mean[[0, 1]] = P_mean[[1, 0]]
            P_mean[:, [0, 1]] = P_mean[:, [1, 0]]

        console.print("✓ [green]Bayesian HMM Calibration complete.[/green]")
        return {
            "P": np.asarray(P_mean, dtype=np.float32),
            "chol_0": np.asarray(chol_0_mean, dtype=np.float32),
            "chol_1": np.asarray(chol_1_mean, dtype=np.float32),
            "pca_loadings": np.asarray(self.loadings, dtype=np.float32),
            "asset_means": np.asarray(self.asset_means, dtype=np.float32),
            "asset_scales": np.asarray(self.asset_scales, dtype=np.float32),
            "asset_names": self.asset_names
        }