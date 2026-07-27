"""funcgarch — Functional GARCH and GAS-GARCH models.

Public API
----------
Basis functions:
    bernstein_basis       Bernstein polynomial basis function (JIT-compiled).
    cubic_bspline_basis   B-spline basis matrix.
    ou_kernel             Ornstein-Uhlenbeck covariance kernel.

Functional GARCH:
    delta                    Level operator (Bernstein expansion of δ).
    kernel_operator          Kernel operator (double Bernstein sum for α/β).
    loss_func                Bernstein-projected MSE loss.
    garch_filter             Extract conditional variance surface from returns.
    garch_estimator          Compute loss for a given parameter vector.
    fit                      Estimate parameters via scipy.minimize.
    bspline_garch_estimator  Functional GARCH via B-spline coefficient recursion.

Functional GAS:
    gas_estimator            Score-driven estimator (Student-t likelihood).

Simulation:
    brownian              Scaled Brownian motion on [0, 1].
    simulate              Simulate a functional GARCH process.

Utilities:
    ResultContainer       Key-value container wrapping optimization results.
"""

from .basis import bernstein_basis, cubic_bspline_basis, ou_kernel
from .garch import (
    delta,
    kernel_operator,
    loss_func,
    garch_filter,
    garch_estimator,
    fit,
    bspline_garch_estimator,
)
from .gas import gas_estimator
from .simulate import brownian, simulate
from .utils import ResultContainer

__all__ = [
    'bernstein_basis',
    'cubic_bspline_basis',
    'ou_kernel',
    'delta',
    'kernel_operator',
    'loss_func',
    'garch_filter',
    'garch_estimator',
    'fit',
    'bspline_garch_estimator',
    'gas_estimator',
    'brownian',
    'simulate',
    'ResultContainer',
]
