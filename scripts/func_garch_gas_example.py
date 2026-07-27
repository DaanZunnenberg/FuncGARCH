import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funcgarch import cubic_bspline_basis, gas_estimator, bspline_garch_estimator

# ── Simulated data ────────────────────────────────────────────────────────────
dT = 1000
dn = 25

mSigma2 = np.zeros((dn, dT))
for id1 in range(dT):
    for id2 in range(dn):
        mSigma2[id2, id1] = (
            1
            + 10 * (id2 - (dn / 2 + dn / 4 * np.sin(id1 * 3 * np.pi / dT))) ** 2
            / (3 * dn / 4) ** 2
            + 3
            + 2 * np.sin(id1 * 2 * np.pi / dT)
        )
mY = np.sqrt(mSigma2) * np.random.standard_normal((dn, dT))

# ── Basis ─────────────────────────────────────────────────────────────────────
dK = 4
n_interior_knots = 3
vtau = np.linspace(0, 1, mY.shape[0])
n = len(vtau)
mBsplinesSparseMat = cubic_bspline_basis(vtau, dK - 2, n_interior_knots)
M = dK + 1

# ── GAS estimation ────────────────────────────────────────────────────────────
vb0 = np.ones((M, 1))
vtheta0_GAS = np.concatenate((
    [2.1, 0.001],
    np.ones(M),
    -0.05 * np.ones(M ** 2),
     0.01 * np.ones(M ** 2),
))
LB_GAS = np.concatenate(([1.05, 0.00001], -5 * np.ones(M), -2 * np.ones(M ** 2), -0.9 * np.ones(M ** 2)))
UB_GAS = np.concatenate(([500,  1],        15 * np.ones(M),  2 * np.ones(M ** 2),  0.9 * np.ones(M ** 2)))

fGAS_likelihood = lambda vtheta: gas_estimator(mY, vb0, mBsplinesSparseMat, vtheta)[0]
optim_GAS = minimize(
    fGAS_likelihood, vtheta0_GAS,
    bounds=list(zip(LB_GAS, UB_GAS)),
    method='SLSQP',
    options={'maxiter': 200},
)
vthetaHat_GAS = optim_GAS.x
print(optim_GAS)

mVolatilityHat_GAS = gas_estimator(mY, vb0, mBsplinesSparseMat, vthetaHat_GAS)[1]
mSigm_hat_GAS = np.exp(mVolatilityHat_GAS / 2)  # sigma (std dev)

# ── GARCH estimation ──────────────────────────────────────────────────────────
bnds_GARCH = [(0, 100)] * M + [(-.99, .99)] * (2 * M ** 2)
vtheta0_GARCH = np.array(
    [0.0] * M
    + (0.1 * np.eye(M).flatten()).tolist()
    + (0.8 * np.eye(M).flatten()).tolist()
)

fGARCH_likelihood = lambda vtheta: bspline_garch_estimator(mY, mBsplinesSparseMat, vtheta)[0]
optim_GARCH = minimize(
    fGARCH_likelihood, vtheta0_GARCH,
    bounds=bnds_GARCH,
    method='Nelder-Mead',
    options={'maxiter': 2000},
)
vthetaHat_GARCH = optim_GARCH.x
print(optim_GARCH)

mVolatilityHat_GARCH = bspline_garch_estimator(mY, mBsplinesSparseMat, vthetaHat_GARCH)[1]
mSigm_hat_GARCH = mVolatilityHat_GARCH ** 0.5  # sigma (std dev)

# ── 3-D comparison: real vs GAS vs GARCH ─────────────────────────────────────
z_real  = mSigma2[:, 4:]
z_GAS   = mSigm_hat_GAS[:, 4:]
z_GARCH = mSigm_hat_GARCH[:, 4:]

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=['Real variance', 'GAS volatility', 'GARCH volatility'],
)
fig.add_trace(go.Surface(z=z_real,  name='Real'),  row=1, col=1)
fig.add_trace(go.Surface(z=z_GAS,   name='GAS'),   row=1, col=2)
fig.add_trace(go.Surface(z=z_GARCH, name='GARCH'), row=1, col=3)
fig.show()
