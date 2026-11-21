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
from rich.console import Console

from config import DRIFT_CAP, HMM_STICKY_PRIOR, MAX_ACCEPTABLE_RHAT, ASSET_CURRENCIES, BASE_CURRENCY

console = Console(highlight=False)


class BayesianFactorCalibrator:
    def __init__(self, raw_data: pd.DataFrame, num_factors: int = 4):
        self.num_factors = num_factors
        clean_prices = raw_data.ffill()
        full_log_returns = np.log(clean_prices / clean_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
        self.asset_names = full_log_returns.columns.tolist()

        # 1. Calibrate each asset's drift (asset_means) and volatility (asset_scales) on its
        #    FULL 20-year history so one short-history ETF does not restrict vol/drift to 2024-2026
        num_assets = len(self.asset_names)
        full_means = np.zeros(num_assets, dtype=np.float64)
        full_stds = np.zeros(num_assets, dtype=np.float64)
        history_counts = {}

        for idx, col in enumerate(self.asset_names):
            s = full_log_returns[col].dropna()
            if len(s) < 60:
                raise ValueError(f"Insufficient historical data (<60 days) for '{col}'.")
            full_stds[idx] = max(float(s.std(ddof=1)), 1e-6)
            history_counts[col] = len(s)

            # Zero out secular FX spot drift for FX trackers and remove 2006-2026 FX trend from local equity drift
            if col.startswith("FX_"):
                full_means[idx] = 0.0
            else:
                ccy = ASSET_CURRENCIES.get(col, BASE_CURRENCY)
                fx_col = f"FX_{ccy}"
                if ccy != BASE_CURRENCY and fx_col in full_log_returns.columns:
                    local_ret = (full_log_returns[col] - full_log_returns[fx_col]).dropna()
                    full_means[idx] = float(local_ret.mean())
                else:
                    full_means[idx] = float(s.mean())

        shortest_ticker = min(history_counts, key=history_counts.get)
        longest_count = max(history_counts.values())
        if history_counts[shortest_ticker] < longest_count - 10:
            console.print(
                f"[dim]Note: Common PCA window limited to {history_counts[shortest_ticker]:,} days by "
                f"'{shortest_ticker}' (full {longest_count:,}-day history used for individual asset vols/drifts).[/dim]"
            )

        # 2. Fit PCA & HMM on standardized overlapping returns
        overlap_returns = full_log_returns.dropna(how="any")
        if len(overlap_returns) < 60:
            raise ValueError("Insufficient overlapping historical return data (<60 days) across assets.")

        overlap_means = overlap_returns.mean(axis=0).values
        overlap_stds = np.maximum(overlap_returns.std(axis=0, ddof=1).values, 1e-6)
        scaled_returns = (overlap_returns.values - overlap_means) / overlap_stds

        self.pca = PCA(n_components=num_factors)
        raw_factors = self.pca.fit_transform(scaled_returns).astype(np.float64)
        eig_stds = np.sqrt(np.maximum(self.pca.explained_variance_, 1e-8))

        # Standardize factors to unit variance so HMM Cholesky matrices are isotropic;
        # absorb eigenvalue scales directly into loadings so (loadings @ F) is unchanged
        self.factor_returns = (raw_factors / eig_stds).astype(np.float64)
        self.loadings = (self.pca.components_.T * eig_stds).astype(np.float32)

        explained_var = np.sum(self.loadings ** 2, axis=1)
        self.residual_scales = np.sqrt(np.maximum(1e-4, 1.0 - explained_var)).astype(np.float32)

        span_years = max((clean_prices.index[-1] - clean_prices.index[0]).days / 365.25, 1.0)
        obs_per_year = len(clean_prices) / span_years

        raw_annual_log = full_means * obs_per_year
        annual_sigmas = full_stds * np.sqrt(obs_per_year)
        raw_annual_simple = np.exp(raw_annual_log + 0.5 * (annual_sigmas ** 2)) - 1.0
        capped_log = np.where(
            raw_annual_simple > DRIFT_CAP,
            np.log(1.0 + DRIFT_CAP) - 0.5 * (annual_sigmas ** 2),
            raw_annual_log
        )

        self.asset_means = (capped_log / 365.25).astype(np.float32)
        self.asset_scales = (annual_sigmas / np.sqrt(365.25)).astype(np.float32)

    def fit(self, draws: int = 1000, tune: int = 1500, cores: int = 4) -> dict:
        console.print(
            f"Configuring MCMC environment for {self.num_factors} PCA factors "
            f"({len(self.factor_returns):,} overlapping trading days)..."
        )
        N = self.num_factors
        T_obs = float(len(self.factor_returns))
        stay_alpha, switch_alpha = float(HMM_STICKY_PRIOR[0]), float(HMM_STICKY_PRIOR[1])
        obs_T = pt.constant(self.factor_returns.T, dtype="float64")  # Shape: (N, T)

        with pm.Model() as model:
            p0 = pm.Dirichlet("p0", a=np.array([stay_alpha, switch_alpha], dtype=np.float64))
            p1 = pm.Dirichlet("p1", a=np.array([switch_alpha, stay_alpha], dtype=np.float64))
            P = pt.stack([p0, p1])

            # Regime 0 (Calm): SDs < 1.0; Regime 1 (Crisis): SDs > 1.0
            chol_0_raw, _, _ = pm.LKJCholeskyCov("lkj_0", n=N, eta=2.0, sd_dist=pm.Exponential.dist(lam=1.5))
            chol_0 = pm.Deterministic("chol_0", chol_0_raw)

            chol_1_raw, _, _ = pm.LKJCholeskyCov("lkj_1", n=N, eta=2.0, sd_dist=pm.Gamma.dist(alpha=4.0, beta=2.0))
            chol_1 = pm.Deterministic("chol_1", chol_1_raw)

            # Exact trace separation at N: Tr(Sigma_0) < N < Tr(Sigma_1) prevents label switching
            var_0 = pt.sum(chol_0_raw ** 2)
            var_1 = pt.sum(chol_1_raw ** 2)
            pen_0 = pt.switch(var_0 < float(N), 0.0, -T_obs * ((var_0 - float(N)) ** 2))
            pen_1 = pt.switch(var_1 > float(N), 0.0, -T_obs * ((float(N) - var_1) ** 2))
            pm.Potential("order_constraint", pen_0 + pen_1)

            def _vectorized_mvn_logp(obs_matrix, chol):
                log_det = pt.sum(pt.log(pt.diag(chol)))
                z = ptlinalg.solve_triangular(chol, obs_matrix, lower=True)
                return -0.5 * (N * np.log(2.0 * np.pi) + 2.0 * log_det + pt.sum(z ** 2, axis=0))

            ll_0 = _vectorized_mvn_logp(obs_T, chol_0)
            ll_1 = _vectorized_mvn_logp(obs_T, chol_1)
            emission_lls = pt.stack([ll_0, ll_1], axis=1)  # Shape: (T, 2)

            def step_forward(em_ll_t, prev_gamma, P_mat):
                gamma_pred = pt.dot(prev_gamma, P_mat)
                log_gamma_update = pt.log(pt.clip(gamma_pred, 1e-16, 1.0)) + em_ll_t
                max_log = pt.max(log_gamma_update)
                marginal_ll = max_log + pt.log(pt.sum(pt.exp(log_gamma_update - max_log)))
                return pt.exp(log_gamma_update - marginal_ll), marginal_ll

            results, _ = pytensor.scan(
                fn=step_forward,
                sequences=[emission_lls],
                outputs_info=[pt.constant(np.array([1.0, 0.0], dtype=np.float64)), None],
                non_sequences=[P],
                strict=True
            )
            pm.Potential("hmm_loglike", pt.sum(results[1]))
            pm.Deterministic("regime_1_prob_today", results[0][-1, 1])

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=cores,
                cores=cores,
                init="adapt_diag",
                target_accept=0.92,
                return_inferencedata=True,
                progressbar=True
            )

            posteriors = trace.posterior
            init_prob_regime_1 = float(np.clip(np.mean(posteriors["regime_1_prob_today"].values), 0.0, 1.0))
            max_rhat = float(az.summary(trace, var_names=["p0", "p1", "lkj_0", "lkj_1"])["r_hat"].max())

            if max_rhat > MAX_ACCEPTABLE_RHAT:
                console.print(
                    f"[yellow]Warning: MCMC Max R-hat ({max_rhat:.4f}) exceeds {MAX_ACCEPTABLE_RHAT:.2f}.[/yellow]"
                )
            console.print(
                f"MCMC Calibration Complete (Max R-hat: {max_rhat:.4f} | "
                f"Live High-Vol Regime Prob: {init_prob_regime_1:.1%})\n"
            )

        P_mean = np.mean(np.stack([posteriors["p0"].values, posteriors["p1"].values], axis=-2), axis=(0, 1))
        chol_0_samples = posteriors["chol_0"].values
        chol_1_samples = posteriors["chol_1"].values
        cov_0_mean = np.mean(chol_0_samples @ np.swapaxes(chol_0_samples, -1, -2), axis=(0, 1))
        cov_1_mean = np.mean(chol_1_samples @ np.swapaxes(chol_1_samples, -1, -2), axis=(0, 1))
        chol_0_mean = np.linalg.cholesky(cov_0_mean)
        chol_1_mean = np.linalg.cholesky(cov_1_mean)

        return {
            "P": np.asarray(P_mean, dtype=np.float32),
            "chol_0": np.asarray(chol_0_mean, dtype=np.float32),
            "chol_1": np.asarray(chol_1_mean, dtype=np.float32),
            "init_prob_regime_1": np.float32(init_prob_regime_1),
            "pca_loadings": np.asarray(self.loadings, dtype=np.float32),
            "residual_scales": self.residual_scales,
            "asset_means": self.asset_means,
            "asset_scales": self.asset_scales,
            "asset_names": self.asset_names
        }