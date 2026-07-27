"""Functional GAS model — score-driven B-spline parametrisation.

Implements the functional GAS (Generalised Autoregressive Score) model, which
parametrises the log-volatility curve with a B-spline basis Φ:

    log sigma_t(u) = Phi(u)^T b_t

The coefficient vector b_t is updated at each day by the scaled score of the
multivariate Student-t log-likelihood:

    b_t = ω + B b_{t-1} + A s_{t-1}

``gas_estimator`` returns the negative average log-likelihood and the fitted
log-volatility surface. It is called directly inside a ``scipy.optimize.minimize``
loop. See ``scripts/`` for worked demonstrations.

The B-spline GARCH baseline (no score updating) lives in ``garch.py`` as
``bspline_garch_estimator``, alongside the Bernstein-basis GARCH model.
"""

import warnings

import numpy as np
from scipy.special import gammaln

from .basis import ou_kernel

warnings.filterwarnings(action='ignore')

_call_count: list[int] = [0]


def _log_step(model_name: str, loss: float, log_every: int = 1) -> None:
    """Print a convergence update; increment the global call counter."""
    if _call_count[0] % log_every == 0:
        msg = f'Running {model_name} :: call #{_call_count[0]:<5} :: loss: {loss:.6f}'
        print(msg.ljust(len(msg) + 30, ' '), end='\r')
    _call_count[0] += 1


def gas_estimator(
    returns: np.ndarray,
    init_coefs: np.ndarray,
    basis_mat: np.ndarray,
    params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Functional GAS-GARCH log-likelihood and fitted log-volatility surface.

    Implements the score-driven update for the B-spline coefficient vector b_t:
        b_t = omega + persistence_mat @ b_{t-1} + score_gain_mat @ score_t

    where the score is derived from the multivariate Student-t log-likelihood
    with OU-structured covariance.

    Args:
        returns: Return matrix, shape (n_grid, n_days).
        init_coefs: Initial B-spline coefficient vector, shape (n_basis, 1).
        basis_mat: B-spline basis matrix, shape (n_basis, n_grid).
        params: Parameter vector:
            [nu, ou_scale, omega (n_basis), vec(B) (n_basis²), vec(A) (n_basis²)]

    Returns:
        Tuple of (negative average log-likelihood, log-volatility surface of
        shape (n_grid, n_days)).
    """
    n_basis, n_grid = basis_mat.shape
    n_days = returns.shape[1]

    nu       = params[0]
    ou_scale = params[1]
    omega          = params[2: n_basis + 2].reshape(n_basis, 1)
    persistence_mat = params[n_basis + 2: n_basis + 2 + n_basis ** 2].reshape(n_basis, n_basis)
    score_gain_mat  = params[-(n_basis ** 2):].reshape(n_basis, n_basis)

    cov_mat  = ou_kernel(np.linspace(0, 1, n_grid), delta=ou_scale)
    cov_inv  = np.linalg.inv(cov_mat)
    nu_scale = (nu + n_grid) / (2 * nu)

    coef_vec        = init_coefs.copy()
    returns_prev    = returns[:, 0].copy().reshape(n_grid, 1)
    log_vol_surface = np.zeros(returns.shape)
    log_lik         = 0.0

    for t in range(1, n_days):
        log_vol = basis_mat.T @ coef_vec          # (n_grid, 1)
        log_vol_surface[:, t] = log_vol[:, 0]

        std_dev   = np.exp(log_vol / 2)           # sigma_t at each grid point
        scale_inv = np.eye(n_grid) / std_dev      # diag(1/sigma_t)

        student_denom = float(np.sum(
            1 + (returns_prev.T @ (scale_inv @ (cov_inv @ (scale_inv @ returns_prev)))) / nu
        ))
        score_a = (returns_prev / std_dev).T * basis_mat   # (n_basis, n_grid)
        score_b = score_a @ (cov_inv @ (scale_inv @ returns_prev))

        if t > 5:
            log_lik += (
                -0.5 * float(np.sum(log_vol))
                - ((n_grid + nu) / 2) * np.log(student_denom)
            )

        score    = (
            -0.5 * basis_mat.sum(axis=1, keepdims=True)
            + (nu_scale / student_denom) * score_b
        )
        coef_vec     = omega + persistence_mat @ coef_vec + score_gain_mat @ score
        returns_prev = returns[:, t].reshape(n_grid, 1)

    log_lik += n_days * (
        gammaln((nu + n_grid) / 2)
        - gammaln(nu / 2)
        - (n_grid / 2) * np.log(np.pi * nu)
        - 0.5 * np.log(np.linalg.det(cov_mat))
    )
    _log_step('GAS estimator', -log_lik / n_days)
    return -log_lik / n_days, log_vol_surface
