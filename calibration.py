import warnings
warnings.filterwarnings("ignore", message=".*PyTensor could not link to a BLAS installation.*")

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

console = Console(highlight=False)

class BayesianFactorCalibrator:
    def __init__(self, raw_data: pd.DataFrame, num_factors: int = 3):
        self.num_factors = num_factors
        returns = np.log(raw_data / raw_data.shift(1)).dropna()
        self.asset_names = returns.columns.tolist()
        
        self.scaler = StandardScaler()
        scaled_returns = self.scaler.fit_transform(returns)
        self.pca = PCA(n_components=num_factors)
        self.factor_returns = self.pca.fit_transform(scaled_returns)
        self.loadings = self.pca.components_.T  
        
        explained_var = np.sum(self.loadings ** 2, axis=1)
        self.residual_scales = np.sqrt(np.maximum(1e-4, 1.0 - explained_var)).astype(np.float32)
        
        raw_annual_log = self.scaler.mean_ * 252.0
        annual_sigmas = self.scaler.scale_ * np.sqrt(252.0)
        raw_annual_simple = np.exp(raw_annual_log + 0.5 * (annual_sigmas ** 2)) - 1.0
        capped_log = np.where(raw_annual_simple > DRIFT_CAP, np.log(1.0 + DRIFT_CAP) - 0.5 * (annual_sigmas ** 2), raw_annual_log)
        
        self.asset_means = (capped_log / 252.0).astype(np.float32)
        self.asset_scales = self.scaler.scale_.astype(np.float32)

    def fit(self, draws: int = 1500, tune: int = 1500, cores: int = 8) -> dict:
        console.print(f"Configuring MCMC environment for {self.num_factors} PCA factors...")
        N = self.num_factors
        
        with pm.Model() as model:
            p0 = pm.Dirichlet("p0", a=np.array([95.0, 5.0]))
            p1 = pm.Dirichlet("p1", a=np.array([5.0, 95.0]))
            P = pt.stack([p0, p1])
            
            chol_0_raw, _, _ = pm.LKJCholeskyCov("lkj_0", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_0 = pm.Deterministic("chol_0", chol_0_raw)
            
            chol_1_raw, _, _ = pm.LKJCholeskyCov("lkj_1", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.0))
            chol_1 = pm.Deterministic("chol_1", chol_1_raw)
            
            var_0 = pt.sum(chol_0_raw**2)
            var_1 = pt.sum(chol_1_raw**2)
            diff = var_0 - var_1
            pm.Potential("order_constraint", pt.switch(diff < 0, 0.0, -100.0 * (diff**4)))
            
            def step_forward(obs_t, prev_gamma, P_mat, c0, c1):
                gamma_pred = pt.dot(prev_gamma, P_mat)
                def mvn_logp(y, chol):
                    k = y.shape[0]
                    log_det = pt.sum(pt.log(pt.diag(chol)))
                    z = ptlinalg.solve(chol, y)
                    return -0.5 * (k * np.log(2 * np.pi) + 2.0 * log_det + pt.sum(z ** 2))
                
                log_gamma_update = pt.stack([pt.log(gamma_pred[0]) + mvn_logp(obs_t, c0), pt.log(gamma_pred[1]) + mvn_logp(obs_t, c1)])
                max_log = pt.max(log_gamma_update)
                marginal_ll = max_log + pt.log(pt.sum(pt.exp(log_gamma_update - max_log)))
                return pt.exp(log_gamma_update - marginal_ll), marginal_ll

            results, _ = pytensor.scan(
                fn=step_forward, sequences=[self.factor_returns],
                outputs_info=[pt.constant(np.array([1.0, 0.0])), None],
                non_sequences=[P, chol_0, chol_1], strict=True
            )
            pm.Potential("hmm_loglike", pt.sum(results[1]))
            
            trace = pm.sample(
                draws=draws, 
                tune=tune, 
                chains=cores, 
                init="adapt_diag",
                target_accept=0.95, 
                return_inferencedata=True, 
                progressbar=True
            )
            console.print(f"MCMC Calibration Complete (Max R-hat: {float(az.summary(trace)['r_hat'].max()):.4f})\n")
        
        posteriors = trace.posterior
        P_mean = np.mean(np.stack([posteriors["p0"].values, posteriors["p1"].values], axis=-2), axis=(0, 1))
        chol_0_mean = np.mean(posteriors["chol_0"].values, axis=(0, 1))
        chol_1_mean = np.mean(posteriors["chol_1"].values, axis=(0, 1))

        return {
            "P": np.asarray(P_mean, dtype=np.float32), "chol_0": np.asarray(chol_0_mean, dtype=np.float32),
            "chol_1": np.asarray(chol_1_mean, dtype=np.float32), "pca_loadings": np.asarray(self.loadings, dtype=np.float32),
            "residual_scales": self.residual_scales, "asset_means": self.asset_means,
            "asset_scales": self.asset_scales, "asset_names": self.asset_names
        }