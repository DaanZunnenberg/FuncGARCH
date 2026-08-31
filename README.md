# FunctionalScale

Research code for **functional GARCH** and **functional GAS-GARCH** models applied to intraday volatility surfaces. The models treat within-day return curves as functional observations, extending classical GARCH/GAS dynamics to function space ($L^2[0,1]$).

---

## Theory

### Setting

Let $\{r_t(u) : u \in [0,1]\}_{t=1}^T$ denote a sequence of intraday log-return curves across $T$ trading days, where $u$ indexes the normalized time within a day ($u = 0$ at market open, $u = 1$ at market close).

The conditional variance curve on day $t$ given the past filtration $\mathcal{F}_{t-1}$ is defined as the functional object $\sigma_t^2(\cdot) \in L^2[0,1]$.

---

### Functional GARCH(p,q)

Following Cerovecki et al. (2018), the model specifies:

$$y_t = \sigma_t \eta_t, \qquad (\eta_t) \text{ is an iid innovation process on } L^2[0,1] \text{ with } \mathbb{E}[\eta_t(u)] = 0, \ \mathbb{E}[\eta_t^2(u)] = 1$$

$$\sigma_t^2 = \delta + \sum_{i=1}^{q} \alpha_i(y_{t-i}^2) + \sum_{j=1}^{p} \beta_j(\sigma_{t-j}^2)$$

where multiplication is pointwise, the intercept curve $\delta \in H^+_*$, and $\alpha_i, \beta_j$ are positive kernel operators defined by:

$$[\alpha(x)](u) = \int_0^1 K_\alpha(u, v)\, x(v)\, dv, \qquad K_\alpha(u,v) \geq 0$$

#### Stationarity
A unique strictly stationary, non-anticipative solution exists if the top Lyapunov exponent of the companion operator sequence satisfies $\gamma < 0$. For a GARCH(1,1) process, this reduces to:

$$\mathbb{E}\log\left\|(\alpha\Upsilon_{t-1} + \beta)\cdots(\alpha\Upsilon_1 + \beta)\right\| < 0$$

where $\Upsilon_t$ is the pointwise multiplication operator by $\eta_t^2$.

#### Bernstein Parametrisation
To estimate the infinite-dimensional operators, we expand them using a basis of non-negative Bernstein polynomials $\varphi_1,\ldots,\varphi_M$:

$$\delta = \sum_{k=1}^M d_k\,\varphi_k, \qquad \alpha_i = \sum_{k,\ell=1}^M a_{k\ell}^{(i)}\,\varphi_k \otimes \varphi_\ell, \qquad \beta_j = \sum_{k,\ell=1}^M b_{k\ell}^{(j)}\,\varphi_k \otimes \varphi_\ell$$

where $(\varphi_k \otimes \varphi_\ell)(x) = \varphi_k\langle x, \varphi_\ell\rangle$. Non-negativity of the coefficients guarantees $\alpha_i, \beta_j \in K^+(H)$. 

#### QMLE Estimation
The parameters are estimated via Quasi-Maximum Likelihood:

$$\hat{\theta}_n = \arg\min_{\theta \in \Theta}\; \frac{1}{n}\sum_{t=1}^n \sum_{m=1}^M \left(\frac{\langle y_t^2,\, \varphi_m \rangle}{\langle \tilde{\sigma}_t^2,\, \varphi_m \rangle} + \log\langle \tilde{\sigma}_t^2,\, \varphi_m \rangle\right)$$

> **Note:** The current implementation (`loss_func` in `garch.py`) uses a Bernstein-projected MSE approximation rather than the formal QMLE criterion. The inner products are evaluated via sample means on a uniform grid of $N$ points, optimized using `scipy.optimize` (SLSQP).

---

### Functional GAS-GARCH

The Generalized Autoregressive Score (GAS) extension maps the infinite-dimensional update to a score-driven recursion on a low-dimensional cubic B-spline coefficient vector $b_t$. 

Let $\Phi(u) = (\phi_1(u), \ldots, \phi_M(u))^\top$ be the B-spline basis. The log-volatility curve is parameterized as:

$$\log \sigma_t(u) = \Phi(u)^\top b_t$$

The coefficients evolve via the GAS recursion:

$$b_t = \omega + B\,b_{t-1} + A\,s_{t-1}$$

#### Likelihood & Score
Evaluated on a discrete grid $u_1, \dots, u_N$, the return vector $r_t \in \mathbb{R}^N$ is modeled as a multivariate Student-$t$ distribution with $\nu$ degrees of freedom and an Ornstein-Uhlenbeck covariance kernel $\Lambda_\delta$ ($[\Lambda_\delta]_{ij} = \exp(-|u_i - u_j|/\delta)$):

$$r_t \mid \mathcal{F}_{t-1} \sim t_\nu\left(0,\ S_t \Lambda_\delta S_t\right), \quad S_t = \mathrm{diag}(\exp(\sigma_t(u_i)/2))$$

Let $\tilde{r}_t = S_t^{-1} r_t$ be the standardized returns, and $A_1 = 1 + \tilde{r}_t^\top \Lambda_\delta^{-1} \tilde{r}_t\,/\,\nu$. The analytic score $s_t = \nabla_{b_t} \log p(r_t | b_t)$ evaluates to:

$$s_t = -\frac{1}{2}\,\Phi\,\mathbf{1}_N + \frac{\nu + N}{2\nu A_1}\,\Phi\left(\tilde{r}_t \odot \Lambda_\delta^{-1}\tilde{r}_t\right)$$

where $\Phi$ is the $(M \times N)$ basis matrix and $\odot$ denotes the Hadamard product.

---

## Setup

```bash
git clone https://github.com/DaanZunnenberg/FunctionalScale.git
cd FunctionalScale
pip install -e .            # editable install, pulls in numpy/scipy/numba/pandas/matplotlib/tqdm
pip install -e ".[dev]"      # + pytest, jupyter (optional, for tests/notebooks)
```

Requires Python >= 3.9.

---

## Usage

### Functional GARCH

```python
import numpy as np
from funcgarch import fit, garch_filter

# mY: (N, T) matrix of intraday return curves — N grid points per day, T days
mY = np.load("returns.npy")
N, T = mY.shape
M = 4  # number of Bernstein basis functions

result = fit(mY, n_grid=N, M=M)          # QMLE-style estimation (scipy.optimize)
vtheta_hat = result.x

sigma2 = garch_filter(mY, n_grid=N, vtheta=vtheta_hat, M=M)  # (N, T) fitted variance surface
```

### Functional GAS

```python
from scipy.optimize import minimize
from funcgarch import gas_estimator

# vtheta = [nu, ou_scale, omega (M,), vec(B) (M*M,), vec(A) (M*M,)]
result = minimize(
    gas_estimator, x0=vtheta_init, args=(mY, N, M),
    method="SLSQP",
)
```

### Data pipeline (WRDS TAQ -> clean returns)

```bash
# 1. Pull raw TAQ data via SAS (see wrds/data_fetcher.sas)
# 2. Clean into an intraday return matrix
python -m scripts.taq_cleaner --parent-path /path/to/taq/exports
```

```python
from data.taq_cleaner import DataCleaner

cleaner = DataCleaner(ticker="SPY")
mY = cleaner.clean(parent_path="/path/to/taq/exports")  # -> (N, T) return matrix
```

---

## Data Flow

```
wrds/*.sas                    scripts/taq_cleaner.py           funcgarch/*.py
┌────────────────┐            ┌───────────────────┐           ┌─────────────────────┐
│ WRDS TAQ pull   │  raw CSV  │ DataCleaner.clean()│  mY (N,T) │ fit() / garch_filter │
│ (data_fetcher,  │ ────────► │  - align to grid   │ ────────► │ gas_estimator        │
│  taq_cleaner,   │           │  - compute returns │           │ bspline_garch_       │
│  nbbo/dynamic_  │           │  - reshape to      │           │  estimator           │
│  taq_minute,    │           │    (N, T) matrix   │           │  -> vtheta_hat,      │
│  export)        │           │                    │           │     sigma2 surface   │
└────────────────┘            └───────────────────┘           └─────────────────────┘
```

Dependency direction: `funcgarch` (core package) has no dependency on `data/` or `wrds/` — it only
consumes an `(N, T)` return matrix. The `data/taq_cleaner.py` pipeline depends on raw output from the
`wrds/*.sas` scripts. `examples/*.ipynb` depend on both `funcgarch` and pre-cleaned data.

---

## Repository Layout

---

_Last reviewed: 2026-08-31._
