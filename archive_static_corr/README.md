# Legacy LDI Engine (Heston / Merton SDE)

## The Old Design

This archive contains the previous iteration of the Liability-Driven Investment (LDI) margin simulator. It operated as a continuous-time Monte Carlo engine using an Euler-Maruyama discretization. The core risk architecture relied on:

Heston Stochastic Volatility: Modeled individual asset volatility dynamically using mean-reversion, volatility-of-volatility, and leverage effects (kappa, xi, rho).

Merton Jump-Diffusion: Injected tail risk via discrete Poisson jumps to simulate sudden market crashes.

Static Correlation: Cross-asset dependency was driven by a static Cholesky decomposition of historical returns, calculated exactly once at the beginning of the simulation and held constant across all time steps.  

## Differences to the Current Framework

The new tool replaces the independent asset SDEs with a Bayesian Regime-Switching Factor Model.

Dimensionality: Instead of simulating 14+ independent assets, the new engine extracts underlying PCA macro-factors and maps asset returns via static betas.

Dynamic Correlation (Regime Switching): The static Cholesky matrix has been replaced. The new engine uses a PyMC-calibrated Hidden Markov Model (HMM) to dynamically switch between normal and high-stress covariance regimes.

Volatility Handling: Rather than using a continuous Heston process for every single asset, volatility clustering and tail risks are now naturally captured by the Markov chain probabilistically jumping into the high-volatility/high-correlation state.