"""Functional GARCH models — Bernstein-basis and B-spline-basis parametrisations.

Implements the complete estimation and filtering pipeline for the functional
GARCH(1,1) model.  The conditional variance curve σ²_t(·) is approximated in
a functional basis and evolved through an operator-valued GARCH recursion:

    σ²_t(u) = δ(u) + ∫α(u,s) r²_{t-1}(s) ds + ∫β(u,s) σ²_{t-1}(s) ds

Two parametrisations are provided:

``garch_estimator`` / ``garch_filter`` / ``fit``
    Bernstein polynomial basis (see ``basis.bernstein_basis``).

``bspline_garch_estimator``
    B-spline basis (see ``basis.cubic_bspline_basis``); no ``fit()`` wrapper.

Call chain
──────────
Estimation::

    fit()
      └─▶ garch_estimator()        # called once per scipy optimizer step
            ├─▶ _build_operators() # unpacks params -> delta_hat, alpha_hat, beta_hat
            └─▶ loss_func()        # Bernstein-projected MSE, accumulated per day

Post-estimation::

    garch_filter()                 # same recursion as garch_estimator,
      └─▶ _build_operators()       # but returns the full (n_grid, n_days) surface
"""

from __future__ import annotations

import warnings
import typing

import numpy as np
from numba import njit, jit
from scipy.optimize import minimize

from .basis import bernstein_basis
from .utils import ResultContainer

warnings.filterwarnings(action='ignore')

_call_count: list[int] = [0]


def _log_step(loss: float, params: np.ndarray, log_every: int = 500) -> None:
    """Print an optimizer progress line every `log_every` function evaluations."""
    _call_count[0] += 1
    if _call_count[0] % log_every == 0:
        vals = ' '.join(f'{100 * p:+.2f}' for p in params)
        print(f'step {_call_count[0]:>5} | loss: {loss:.6f} | params: {vals}')


_bspline_call_count: list[int] = [0]


def _log_bspline_step(loss: float, log_every: int = 1) -> None:
    """Print a convergence update for the B-spline GARCH estimator."""
    if _bspline_call_count[0] % log_every == 0:
        msg = f'Running GARCH estimator :: call #{_bspline_call_count[0]:<5} :: loss: {loss:.6f}'
        print(msg.ljust(len(msg) + 30, ' '), end='\r')
    _bspline_call_count[0] += 1


@njit
def delta(coefs: np.ndarray, u: float, n_basis: int, init: float = 0.0) -> float:
    r"""Level operator $\delta(u) = \sum_{k=1}^M c_k \varphi_k^M(u)$.

    Args:
        coefs: Level coefficient vector, length n_basis.
        u: Evaluation point(s) in [0, 1].  Pass a scalar or an array.
        n_basis: Number of Bernstein basis functions.
        init: Starting value for the accumulation.  Pass ``np.zeros(n_grid)``
              for vectorised evaluation over a grid of length n_grid.
    """
    acc = init
    for k, c in enumerate(coefs):
        acc = acc + c * bernstein_basis(u, n_basis, k + 1)
    return acc


@jit(nopython=True)
def kernel_operator(
    u: np.ndarray,
    coefs: np.ndarray,
    n_basis: int,
    init: np.ndarray,
) -> np.ndarray:
    r"""Kernel operator $\mathcal{K}(u,s) = \sum_{k,l} c_{kl}\,\varphi_k^M(u)\,\varphi_l^M(s)$.

    Evaluates the M²-term double sum via matrix products of Bernstein column
    vectors, returning an n_grid×n_grid matrix.

    Args:
        u: Grid vector of length n_grid; reshaped to a column internally.
        coefs: Flattened coefficient matrix, length n_basis².
        n_basis: Number of Bernstein basis functions per dimension.
        init: Zero-initialised (n_grid, n_grid) accumulator array (required by Numba).
    """
    acc = init
    col = u.reshape((len(u), 1))
    idx = 0
    for k in range(1, n_basis + 1):
        bk = bernstein_basis(col, n_basis, k)
        for j in range(1, n_basis + 1):
            acc = acc + coefs[idx] * bk @ bernstein_basis(col, n_basis, j).T
            idx += 1
    return acc


@jit(nopython=True)
def loss_func(
    returns: np.ndarray,
    variance: np.ndarray,
    n_basis: int,
    grid: np.ndarray,
) -> float:
    r"""Bernstein-projected MSE between squared returns and conditional variance.

    Computes the per-day loss

    .. math::

        L_t = \sum_{k=1}^{M}
              \frac{1}{N}\sum_{i=1}^{N}
              \bigl[(r_t(u_i)^2 - \sigma_t^2(u_i))\,\varphi_k^M(u_i)\bigr]^2

    Args:
        returns: Intraday return vector for one day, shape (n_grid,).
        variance: Conditional variance vector, shape (n_grid,).
        n_basis: Number of Bernstein basis functions.
        grid: Intraday evaluation grid, shape (n_grid,).
    """
    total = 0.0
    for k in range(1, n_basis + 1):
        w = bernstein_basis(grid, n_basis, k)
        total += np.mean(((returns ** 2 - variance) * w) ** 2)
    return total


def _build_operators(
    params: np.ndarray,
    n_basis: int,
    n_grid: int,
    delta_fn: typing.Callable,
    kernel_fn: typing.Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack parameter vector into pre-evaluated operator matrices.

    Returns:
        (grid, delta_hat, alpha_hat, beta_hat) where delta_hat is shape (n_grid,)
        and alpha_hat, beta_hat are (n_grid, n_grid).
    """
    grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)
    delta_coefs = params[:n_basis]
    alpha_coefs = params[n_basis: n_basis + n_basis ** 2]
    beta_coefs  = params[n_basis + n_basis ** 2:]
    delta_hat = delta_fn(delta_coefs, grid, n_basis=n_basis, init=np.zeros(n_grid))
    alpha_hat = kernel_fn(grid, alpha_coefs, n_basis=n_basis, init=np.zeros((n_grid, n_grid))).T
    beta_hat  = kernel_fn(grid, beta_coefs,  n_basis=n_basis, init=np.zeros((n_grid, n_grid))).T
    return grid, delta_hat, alpha_hat, beta_hat


def garch_filter(
    returns: np.ndarray,
    n_grid: int,
    params: np.ndarray,
    n_basis: int,
    initial_variance: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
) -> np.ndarray:
    """Extract the conditional variance surface from observed returns.

    Applies the fitted functional GARCH recursion forward through all days.

    Args:
        returns: Return matrix, shape (n_grid, n_days).
        n_grid: Number of intraday grid points.
        params: Parameter vector [delta_coefs (n_basis) | alpha_coefs (n_basis²) | beta_coefs (n_basis²)].
        n_basis: Number of Bernstein basis functions.
        initial_variance: Initial variance vector, shape (n_grid,).
        delta_fn: Level operator (injectable for alternative bases).
        kernel_fn: Kernel operator (injectable for alternative bases).

    Returns:
        Variance matrix of shape (n_grid, n_days).
    """
    n_grid_obs, n_days = returns.shape
    grid, delta_hat, alpha_hat, beta_hat = _build_operators(
        params, n_basis, n_grid, delta_fn, kernel_fn
    )
    variance = initial_variance * np.ones(n_grid)
    variance_surface = np.zeros((n_grid_obs, n_days))
    variance_surface[:, 0] = variance

    for t in range(1, n_days):
        variance = (
            delta_hat
            + (alpha_hat * returns[:, t - 1] ** 2) @ np.ones(n_grid_obs) / n_grid_obs
            + (beta_hat  * variance)               @ np.ones(n_grid_obs) / n_grid_obs
        )
        variance_surface[:, t] = variance
    return variance_surface


def garch_estimator(
    returns: np.ndarray,
    n_grid: int,
    params: np.ndarray,
    n_basis: int,
    initial_variance: np.ndarray,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
) -> float:
    """Compute the functional GARCH objective for a given parameter vector.

    Runs the GARCH recursion forward through all days and accumulates the
    Bernstein-projected MSE loss.  Designed to be passed to scipy.minimize.

    Args:
        returns: Return matrix, shape (n_grid, n_days).
        n_grid: Number of intraday grid points.
        params: Parameter vector [delta_coefs (n_basis) | alpha_coefs (n_basis²) | beta_coefs (n_basis²)].
        n_basis: Number of Bernstein basis functions.
        initial_variance: Initial variance vector, shape (n_grid,).
        delta_fn: Level operator.
        kernel_fn: Kernel operator.
        loss_fn: Per-day loss function (default: loss_func).
        print_convergence: If True, print progress every 500 evaluations.

    Returns:
        Scalar objective value.
    """
    n_grid_obs, n_days = returns.shape
    grid, delta_hat, alpha_hat, beta_hat = _build_operators(
        params, n_basis, n_grid, delta_fn, kernel_fn
    )
    variance = initial_variance * np.ones(n_grid)
    total_loss = 0.0

    for t in range(1, n_days):
        variance = (
            delta_hat
            + ((alpha_hat * returns[:, t - 1] ** 2) @ np.ones(n_grid_obs)
            +  (beta_hat  * variance)               @ np.ones(n_grid_obs)) / n_grid_obs
        )
        total_loss += loss_fn(returns[:, t], variance, n_basis, grid)

    if print_convergence:
        _log_step(total_loss, params)
    return total_loss


def fit(
    returns: np.ndarray,
    initial_variance: np.ndarray,
    n_grid: int,
    n_basis: int = 1,
    estimator_fn: typing.Callable = garch_estimator,
    delta_fn: typing.Callable = delta,
    kernel_fn: typing.Callable = kernel_operator,
    loss_fn: typing.Callable = loss_func,
    print_convergence: bool = False,
    options: dict | None = None,
    **kwargs,
) -> ResultContainer:
    """Estimate functional GARCH parameters by minimising the projected MSE loss.

    Wraps scipy.minimize; optimizer arguments (x0, bounds, method, …) are
    passed through **kwargs.

    Args:
        returns: Return matrix, shape (n_grid, n_days).
        initial_variance: Initial variance vector, shape (n_grid,).
        n_grid: Number of intraday grid points.
        n_basis: Number of Bernstein basis functions.
        estimator_fn: Full-sequence objective to minimise.
        delta_fn: Level operator (override to use a different basis).
        kernel_fn: Kernel operator (override to use a different basis).
        loss_fn: Per-day loss function.
        print_convergence: If True, log optimizer progress.
        options: Options dict forwarded to scipy.minimize (merged with defaults).
        **kwargs: Remaining arguments passed to scipy.minimize
                  (x0, bounds, method, constraints, …).

    Returns:
        ResultContainer wrapping the scipy OptimizeResult.

    Example::

        result = fit(
            returns, initial_variance=np.ones(n_grid), n_grid=n_grid, n_basis=2,
            x0=np.zeros(n_basis + 2 * n_basis**2),
            bounds=[(-.99, .99)] * (n_basis + 2 * n_basis**2),
            method='SLSQP',
        )
        params_hat = result.x
    """
    optimizer_options = {'disp': True}
    if options:
        optimizer_options.update(options)

    def _objective(params: np.ndarray) -> float:
        return estimator_fn(
            returns, n_grid, params,
            n_basis=n_basis, initial_variance=initial_variance,
            delta_fn=delta_fn, kernel_fn=kernel_fn, loss_fn=loss_fn,
            print_convergence=print_convergence,
        )

    try:
        opt = minimize(_objective, options=optimizer_options, **kwargs)
    finally:
        _call_count[0] = 0

    return ResultContainer(**{k: opt[k] for k in opt.__dir__()})


def bspline_garch_estimator(
    returns: np.ndarray,
    basis_mat: np.ndarray,
    params: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> tuple[float, np.ndarray]:
    """Functional GARCH estimator using B-spline basis projections.

    Implements the functional GARCH(1,1) recursion in the B-spline coefficient
    space:
        sigma²_t = B^T delta + (B^T A B) * y²_{t-1} + sigma²_{t-1} (B^T C B)

    Args:
        returns: Return matrix, shape (n_grid, n_days).
        basis_mat: B-spline basis matrix, shape (n_basis, n_grid).
        params: Parameter vector [delta_coefs (n_basis) | vec(alpha) (n_basis²) | vec(beta) (n_basis²)].
        p: AR order (currently only p=1 supported).
        q: MA order (currently only q=1 supported).

    Returns:
        Tuple of (MSE loss, fitted variance matrix of shape (n_grid, n_days)).
    """
    if max(p, q) > 1:
        raise NotImplementedError('Order (p,q) must be (1,1)')

    n_basis = basis_mat.shape[0]
    n_grid, n_days = returns.shape

    delta_coefs = np.array(params[:n_basis]).reshape(1, n_basis)
    alpha_coefs = params[n_basis: n_basis + n_basis ** 2].reshape(n_basis, n_basis)
    beta_coefs  = params[n_basis + n_basis ** 2:].reshape(n_basis, n_basis)

    variance         = np.ones(n_grid)
    variance_surface = np.zeros(returns.shape)
    variance_surface[:, 0] = variance

    delta_hat = delta_coefs @ basis_mat                      # (1, n_grid)
    alpha_hat = basis_mat.T @ (alpha_coefs @ basis_mat)     # (n_grid, n_grid)
    beta_hat  = basis_mat.T @ (beta_coefs  @ basis_mat)     # (n_grid, n_grid)

    loss = 0.0
    for t in range(1, n_days):
        variance = np.asarray(
            delta_hat
            + (alpha_hat * returns[:, t - 1] ** 2) @ np.ones(n_grid) / n_grid
            + (variance @ beta_hat) / n_grid
        ).ravel()
        variance_surface[:, t] = variance
        loss += float(np.sum(((returns[:, t] ** 2 - variance) * basis_mat) ** 2))

    _log_bspline_step(loss)
    return loss, variance_surface


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Functional GARCH demo')
    parser.add_argument('--data-path', default='../price_data_example.csv',
                         help='Path to CSV with a "date" index column and an "open" price column')
    parser.add_argument('--start-date', default='2023-12-14', help='Start date for the sample window')
    parser.add_argument('--end-date', default='2024-02-22', help='End date for the sample window')
    parser.add_argument('--resample', default='300S', help='Pandas resample rule for the price series')
    args = parser.parse_args()

    prices = (
        pd.read_csv(args.data_path, parse_dates=True, index_col='date')
        .open
        .loc[args.start_date:args.end_date]
        .resample(args.resample).last()
    )
    log_returns = 100 * np.log(prices[1:] / prices[:-1].values)
    n_grid, n_days = int(len(log_returns) / 70), 70
    returns = np.column_stack([
        log_returns.iloc[k * n_days: k * n_days + n_grid].values
        for k in range(n_days)
    ])

    n_basis = 4
    fit_result = fit(
        returns, initial_variance=np.ones(n_grid), n_grid=n_grid, n_basis=n_basis,
        x0=np.array([0.001] * n_basis + [np.random.uniform(-0.2, 1) for _ in range(2 * n_basis ** 2)]),
        bounds=[(-.99, .99)] * (n_basis + 2 * n_basis ** 2),
        method='SLSQP',
    )

    grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)
    fig, ax = plt.subplots()
    ax.plot(delta(fit_result.x[:n_basis], grid, n_basis=n_basis, init=np.zeros(n_grid)), label='level δ')
    ax.plot(returns[:, 50], color='black', lw=1, alpha=0.5, label='returns day 50')
    ax.plot(
        garch_filter(returns, n_grid=n_grid, params=fit_result.x, n_basis=n_basis, initial_variance=np.ones(n_grid))[:, 60],
        label='σ² day 60',
    )
    ax.legend()
    plt.show()
