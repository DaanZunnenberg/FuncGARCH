#!/usr/bin/env python3
"""Regenerate the GARCH-vs-GAS surfaces and save each as a standalone,
bare (no title/axis-labels/suptitle) image, split out of the combined
garch_vs_gas_vol_surface.png figure."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gas_vol_surface import N_GRID, N_DAYS, SEED, simulate_vol_surface, fit_gas
from gas_vol_surface_container import BG_950, LINE_SOFT, INK, _VOL_CMAP, WARMUP
from funcgarch.garch import fit, garch_filter

N_BASIS = 3

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'

OBS_EQ = r'$\displaystyle y_t(u) = \sigma_t(u)\eta_t(u)$'

GARCH_EQ = '\n'.join([
    OBS_EQ,
    r'$\displaystyle\sigma_t^2 = \delta + \sum_{i=1}^{q}\alpha_i\left(y_{t-i}^2\right)'
    r' + \sum_{j=1}^{p}\beta_j\left(\sigma_{t-j}^2\right)$',
])
GAS_EQ = '\n'.join([
    OBS_EQ,
    r'$\displaystyle\sigma_{t+1}^2 = \delta + B\sigma_t^2 + A\int \phi_K(s)\,y_t^2(s)\,ds$',
    r'$\displaystyle b_t = \omega + Bb_{t-1} + As_{t-1}$',
])

CACHE = pathlib.Path(__file__).resolve().parent / '_garch_vs_gas_cache.npz'


def get_sigmas():
    if CACHE.exists():
        d = np.load(CACHE)
        return d['sigma_garch'], d['sigma_gas']

    mY, sigma2_true = simulate_vol_surface(N_GRID, N_DAYS, seed=SEED)

    n_params = N_BASIS + 2 * N_BASIS ** 2
    initial_variance = np.full(N_GRID, mY.var())
    garch_result = fit(
        mY, initial_variance=initial_variance, n_grid=N_GRID, n_basis=N_BASIS,
        x0=np.zeros(n_params),
        bounds=[(-0.99, 0.99)] * n_params,
        method='SLSQP',
        print_convergence=True,
        options={'maxiter': 500, 'disp': True},
    )
    sigma2_garch = garch_filter(mY, N_GRID, garch_result.x, N_BASIS, initial_variance)
    sigma_garch = np.sqrt(np.clip(sigma2_garch, 1e-6, None))

    _, sigma_gas, _ = fit_gas(mY)

    np.savez(CACHE, sigma_garch=sigma_garch, sigma_gas=sigma_gas)
    return sigma_garch, sigma_gas


def plot_bare(sigma, out_path, vmin, vmax, equation, warmup=WARMUP, block=4):
    surf = sigma[:, warmup:]
    T_full = surf.shape[1]
    T_trim = (T_full // block) * block
    surf = surf[:, :T_trim].reshape(surf.shape[0], -1, block).mean(axis=2)

    n, T = surf.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(T) * block)

    fig = plt.figure(figsize=(13.5, 9), facecolor=BG_950)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d', facecolor=BG_950)
    ax.plot_surface(
        X, Y, surf.T, cmap=_VOL_CMAP, vmin=vmin, vmax=vmax,
        alpha=0.92, linewidth=0, antialiased=True,
        rcount=T, ccount=n,
    )
    # Fix the 3D box proportions to a 3:2 widescreen shape, independent of the
    # data's own extent, then pad the x-range (the day axis stays untouched)
    # so the surface only fills the left portion of that box — the grid
    # panes still span the full box, leaving genuine grid background on the
    # right for the equations to sit over.
    ax.set_box_aspect((2.1, 1.0, 0.72))
    ax.set_xlim(X.min(), X.min() + (X.max() - X.min()) * 1.7)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color'] = LINE_SOFT
        axis._axinfo['grid']['linewidth'] = 0.4
    ax.tick_params(colors=BG_950, labelsize=0, length=0)

    ax.text2D(
        0.96, 0.88, equation, transform=ax.transAxes,
        ha='right', va='top', fontsize=13, color=INK, linespacing=2.1,
    )

    fig.savefig(out_path, dpi=220, facecolor=BG_950)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    sigma_garch, sigma_gas = get_sigmas()
    vmin = min(sigma_garch[:, WARMUP:].min(), sigma_gas[:, WARMUP:].min())
    vmax = max(sigma_garch[:, WARMUP:].max(), sigma_gas[:, WARMUP:].max())
    plot_bare(sigma_garch, 'garch_surface_bare.png', vmin, vmax, GARCH_EQ)
    plot_bare(sigma_gas, 'gas_surface_bare.png', vmin, vmax, GAS_EQ)


if __name__ == '__main__':
    main()
