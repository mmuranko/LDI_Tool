# IBKR margin leverage simulator

This project estimates how much margin leverage a portfolio can use while keeping the simulated probability of a margin breach below a configured limit.

It downloads market and FX history, converts the portfolio to a base currency, estimates return and risk parameters, simulates many future daily paths, and searches for the highest target leverage that stays within the risk budget.

The simulation treats a path as failed when either maintenance-margin excess liquidity is negative or Reg T SMA is negative.

## What the tool does

1. Reads the portfolio, currencies, margin rules, cash-flow schedule, current margin debt, and current SMA from `config.py`.
2. Downloads historical prices and FX rates with `yfinance`.
3. Converts all holdings to the base currency.
4. Estimates annual drift, volatility, and correlations from historical log returns.
5. Simulates daily portfolio paths until a date after the last scheduled withdrawal.
6. Applies scheduled deposits, scheduled withdrawals, margin interest, FX movements, stochastic volatility, and jump shocks.
7. Buys the configured active asset when cash flows allow the portfolio to move toward the tested target leverage.
8. Checks maintenance margin and SMA on every simulated day.
9. Searches a leverage grid and selects the highest target leverage whose simulated breach probability is at or below the configured limit.
10. Runs a 1.00x benchmark and prints terminal-value and performance comparisons.
11. Shows plots for portfolio value paths, terminal value distributions, terminal leverage, and breach probability by leverage level.

The tool does not place trades. It prints an execution directive that states the selected leverage target and the implied active-asset purchase amount.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Orchestrates data loading, simulation, optimization, reporting, plots, and cleanup. |
| `config.py` | User settings: holdings, currencies, broker margin rules, current debt and SMA, deposits, withdrawals, risk limits, and model parameters. |
| `data_fetcher.py` | Downloads prices and FX rates, converts holdings to the base currency, and estimates factor parameters. |
| `engine.py` | Runs the Monte Carlo simulation. The heavy loops are compiled with `numba`. |
| `optimizer.py` | Searches the leverage grid and selects the highest acceptable leverage. |
| `requirements.txt` | Python dependencies. |

The Python imports expect these exact module names. If the files were downloaded with names such as `main(2).py`, rename them before running the project.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Running

Edit `config.py` first, then run:

```bash
python main.py
```

At startup the script asks whether to skip the next scheduled month-end deposit. This is useful if that deposit is already reflected in the current portfolio state.

The script then prints:

- the initial margin sheet;
- the simulation horizon;
- the selected leverage target;
- the implied active-asset purchase;
- the estimated breach probability;
- a comparison against the 1.00x benchmark.

It also opens matplotlib charts. Close the chart windows to let the script finish cleanup.

## Configuration

The main settings are in `config.py`:

- `HOLDINGS`: asset quantities.
- `ASSET_CURRENCIES`: trading currency for each asset.
- `MM_REQUIREMENTS`: maintenance margin requirement by asset.
- `IM_REQUIREMENTS`: initial margin requirement by asset.
- `CURRENT_DEBT`: current margin loan.
- `CURRENT_SMA`: current Reg T SMA.
- `WITHDRAWAL_SCHEDULE`: dated future withdrawals.
- `DEFAULT_MONTHLY_DEPOSIT_2026` and `DEFAULT_MONTHLY_DEPOSIT_FUTURE`: scheduled month-end deposits.
- `ACTIVE_ASSET`: asset bought when the simulation increases exposure.
- `NUM_PATHS`: number of Monte Carlo paths.
- `MAX_MARGIN_CALL_PROBABILITY`: maximum allowed simulated breach probability.
- `MAX_TARGET_LEVERAGE` and `OPTIMIZER_TOLERANCE`: leverage grid bounds and step size.

The default configuration uses CHF as the base currency and buys `VT` as the active asset.

## Main equations

### Log returns

Historical prices are converted to log returns:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

These returns are used to estimate annual volatility, drift, and correlations.

### Annual volatility and drift

For each asset factor:

$$
\sigma = \operatorname{std}(r_t) \sqrt{252}
$$

$$
\mu = 252 \operatorname{mean}(r_t) + \frac{1}{2}\sigma^2
$$

The drift is capped by `DRIFT_CAP`. FX factors use historical volatility and zero drift.

### Correlated daily shocks

The simulator builds a correlation matrix from factor returns and uses its Cholesky factor `C`:

$$
\Delta W = C z \sqrt{\Delta t}, \quad z \sim N(0, I)
$$

This gives correlated daily shocks for assets and FX rates.

### Stochastic variance

Each asset has a variance process:

$$
v_{t+\Delta t} = \max\left(v_t + \kappa(\theta - v_t)\Delta t + \xi\sqrt{v_t}\Delta W_v, 0\right)
$$

The variance shock is linked to the asset return shock:

$$
\Delta W_v = \rho \Delta W_s + \sqrt{1 - \rho^2}\Delta W_\perp
$$

This lets volatility rise or fall along each path instead of staying fixed.

### Jump shock

The number of jumps in a day is sampled as:

$$
N_t \sim \operatorname{Poisson}(\lambda \Delta t)
$$

If jumps occur, the multiplicative jump is:

$$
J_t = \exp(Y_t), \quad Y_t \sim N(N_t \mu_J, \sqrt{N_t}\sigma_J)
$$

The expected jump effect is subtracted from normal drift:

$$
\lambda \left(\exp\left(\mu_J + \frac{1}{2}\sigma_J^2\right) - 1\right)
$$

This keeps the non-jump drift from double-counting the average jump return.

### Asset update

For each asset factor:

$$
V_{t+\Delta t} = V_t \exp\left((\mu - \text{expected jump} - \frac{1}{2}v_t)\Delta t + \sqrt{v_t}\Delta W_s\right) J_t
$$

Foreign assets are also multiplied by the simulated FX move into the base currency.

### FX update

For each required FX factor:

$$
F_{t+\Delta t} = F_t \exp\left((\mu_{FX} - \frac{1}{2}\sigma_{FX}^2)\Delta t + \sigma_{FX}\Delta W_{FX}\right)
$$

In this project, `mu_FX` is set to zero.

### Net liquidation value

Gross assets are portfolio assets plus cash:

$$
G_t = \sum_i V_{i,t} + \text{cash}_t
$$

Net liquidation value is gross assets minus margin debt:

$$
NLV_t = G_t - D_t
$$

Portfolio leverage is:

$$
L_t = \frac{G_t}{NLV_t}
$$

### Target leverage purchase

For a tested target leverage `L*`, the simulator computes:

$$
G_t^* = L^* \cdot NLV_t
$$

$$
\text{purchase}_t = \max(G_t^* - G_t, 0)
$$

The purchase is made in the active asset. Cash is used first; any remaining amount increases margin debt. The simulator does not sell assets to force leverage back down to the target.

### Maintenance margin and excess liquidity

Maintenance margin is:

$$
MM_t = \sum_i m_i^{MM} V_{i,t}
$$

Excess liquidity is:

$$
EL_t = NLV_t - MM_t
$$

A path fails if:

$$
EL_t < 0
$$

### Initial margin and SMA

Initial margin is:

$$
IM_t = \sum_i m_i^{IM} V_{i,t}
$$

After cash flows and purchases, SMA is updated as:

$$
SMA_t = \max(SMA_t^*, NLV_t - IM_t)
$$

`SMA_t^*` is the prior SMA after the script applies deposits, withdrawals, and active-asset purchases. A path fails if:

$$
SMA_t < 0
$$

### Simulated breach probability

For a tested leverage level:

$$
\hat p_{ruin}(L) = \frac{1}{N}\sum_{j=1}^{N} 1\{\text{path } j \text{ fails}\}
$$

This is the share of Monte Carlo paths that breach either maintenance-margin excess liquidity or SMA.

### Reporting bound

The terminal report also computes a simple normal-approximation upper bound:

$$
\hat p_{95}^+ = \min\left(1, \hat p + 1.96\sqrt{\frac{\max(\hat p(1-\hat p), 10^{-12})}{N}}\right)
$$

This is used only for reporting whether the result is close to the risk budget. The optimizer itself uses `hat p_ruin <= p_max`.

### Leverage selection

The optimizer chooses:

$$
L^* = \max\left\{L: \hat p_{ruin}(L) \le p_{max}\right\}
$$

`p_max` is `MAX_MARGIN_CALL_PROBABILITY`. The search uses a fixed random seed and common random numbers, so each leverage level is tested on the same underlying market paths.

### Time-weighted return comparison

For the median surviving path, the script removes the effect of scheduled net cash flow:

$$
R_t = \frac{V_t - CF_t}{V_{t-1}}
$$

The annualized time-weighted return is:

$$
TWRR = \left(\prod_t R_t\right)^{1/Y} - 1
$$

This is used to compare the optimized run with the 1.00x benchmark.

## Outputs

The run produces terminal tables and charts:

- initial margin sheet;
- simulation horizon;
- execution directive;
- terminal NLV comparison against the 1.00x benchmark;
- NLV and leverage percentile bands through time;
- terminal NLV distribution comparison;
- terminal leverage diagnostics;
- breach probability curve across leverage levels.

Temporary memory-mapped files are created while storing path histories. The script removes them at the end of a normal run.

## Notes and limits

- Results depend on the current `yfinance` download, the current date, and the values in `config.py`.
- Broker rules are represented by the margin rates in `config.py`; the script does not validate them against live broker rules.
- Taxes, transaction costs, dividends as cash distributions, and forced liquidation mechanics are not modeled explicitly.
