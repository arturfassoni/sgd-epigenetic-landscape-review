"""
fig2_fitness_imprint.py

Figure 2 of the review (Section 2.4): under uniform competition, the imprint
that fitness differences leave on the phenotypic composition during growth is
erased by the switching dynamics, at the rate given by its spectral gap.
Same computation as Figure 2 of the companion paper on the continuum result
(repository uniform-competition-continuum), without the additive-competition
comparison, drawn in the style of the other figures of the review.

Model (finite-volume discretisation on N = 400 cells of (0,1), zero flux):
    d_t u = d_x( D d_x u + P'(x) u ) + (1 - U) r(x) u,   U = int u dx,
i.e. v = -P', with P(x) = 0.03 cos(4 pi x) + 0.03 x, D = 0.02,
r(x) = 0.1 + 1.9 / (1 + exp(-(x - 1/2)/0.02)), and u(x,0) = 0.01 psi(x),
where psi ∝ exp(-P/D) is the stationary density of the switching dynamics.
The scheme uses the fluxes J = a (u_{i+1}/psi_{i+1} - u_i/psi_i)/h, a = D psi,
so that it conserves mass, preserves positivity and has psi as exact
stationary state. Time integration: BDF, relative tolerance 1e-9.
Panel (f): memory time 1/lambda_1 (lambda_1 = spectral gap of the discrete
switching generator) as a function of 1/D, for the same potential.

Run with: python3 fig2_fitness_imprint.py
Produces: fig2_fitness_imprint.pdf
Dependencies: numpy, scipy, matplotlib.
"""
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh
from scipy.special import expit
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ------------------------------------------------ style of the review ----
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.edgecolor'] = '0.4'
mpl.rcParams['axes.linewidth'] = 0.7
mpl.rcParams['xtick.color'] = '0.3'
mpl.rcParams['ytick.color'] = '0.3'
mpl.rcParams['text.color'] = '0.15'
mpl.rcParams['axes.labelcolor'] = '0.15'
mpl.rcParams['font.size'] = 10
panel_kw = dict(fontsize=12, fontweight='bold', va='bottom', ha='right')
title_kw = dict(loc='left', fontsize=10, color='0.25', pad=8)


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ------------------------------------------------------------ numerics ----
N = 400
edges = np.linspace(0.0, 1.0, N + 1)
x = 0.5 * (edges[1:] + edges[:-1])
h = np.diff(edges)
P = lambda z: 0.03 * np.cos(4 * np.pi * z) + 0.03 * z
r = 0.1 + 1.9 * expit((x - 0.5) / 0.02)


def generator(D):
    """Finite-volume switching generator and stationary density."""
    psi_c = np.exp(-(P(x) - P(x).min()) / D)
    a = D * np.exp(-(P(edges[1:-1]) - P(x).min()) / D)
    w = a / np.diff(x)
    rows, cols, vals = [], [], []
    for i in range(N - 1):
        for (ri, ci, s) in [(i, i + 1, 1), (i, i, -1), (i + 1, i + 1, -1), (i + 1, i, 1)]:
            rows.append(ri); cols.append(ci); vals.append(s * w[i] / h[ri] / psi_c[ci])
    L = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return L, psi_c / (h @ psi_c)


def spectral_gap(L, psi):
    s = np.sqrt(h / psi)
    B = (sp.diags(s) @ L @ sp.diags(1 / s)).toarray()
    return -np.sort(eigvalsh(0.5 * (B + B.T)))[-2]


def simulate(L, psi, T, t_eval):
    def rhs(t, u):
        return L @ u + (1 - h @ u) * r * u

    def jac(t, u):
        return L + sp.diags((1 - h @ u) * r)

    s = solve_ivp(rhs, [0, T], 0.01 * psi, t_eval=t_eval, method='BDF', jac=jac,
                  rtol=1e-9, atol=1e-13)
    assert s.success
    return s.y


D0 = 0.02
L, psi = generator(D0)
lam1 = spectral_gap(L, psi)
t = np.linspace(0, 100, 2001)
yu = simulate(L, psi, 100, t)
Uu = h @ yu
p = yu / Uu
dist_u = h @ np.abs(p - psi[:, None])
print(f"lambda_1 = {lam1:.4f}; max L1 distance {dist_u.max():.3f} at t = {t[dist_u.argmax()]:.1f}; "
      f"U(5) = {Uu[100]:.3f}; left-well share of psi = {h[x < 0.5] @ psi[x < 0.5]:.3f}")

Ds = np.geomspace(0.008, 0.1, 22)
lams = np.array([spectral_gap(*generator(Dv)) for Dv in Ds])
xs = np.linspace(0, 1, 20001); Ps = P(xs)
iR = np.argmin(np.where(xs > 0.5, Ps, np.inf))
iS = np.argmax(np.where((xs > 0.3) & (xs < 0.7), Ps, -np.inf))
dP = Ps[iS] - Ps[iR]
print(f"barrier seen from the shallow well: {dP:.4f}; memory time from "
      f"{1 / lams[-1]:.1f} (D = {Ds[-1]}) to {1 / lams[0]:.0f} (D = {Ds[0]})")

# -------------------------------------------------------------- figure ----
fig = plt.figure(figsize=(9.5, 10.2))
gs = GridSpec(3, 2, height_ratios=[0.85, 1.6, 1.05], hspace=0.5, wspace=0.32, figure=fig)

# (a) landscape
ax = fig.add_subplot(gs[0, 0])
ax.plot(xs, Ps, color='0.15', lw=1.3)
ax.set_xlim(0, 1)
ax.set_xlabel('phenotype $x$'); ax.set_ylabel('$P(x)$')
despine(ax)
ax.set_title('epigenetic landscape (tilted double well)', **title_kw)
ax.text(-0.18, 1.08, 'a', transform=ax.transAxes, **panel_kw)

# (b) proliferation rate
ax = fig.add_subplot(gs[0, 1])
ax.plot(x, r, color='0.35', lw=1.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 2.1)
ax.set_xlabel('phenotype $x$'); ax.set_ylabel('$r(x)$')
despine(ax)
ax.set_title('proliferation rate (favors the right well)', **title_kw)
ax.text(-0.18, 1.08, 'b', transform=ax.transAxes, **panel_kw)

# (c) kymograph of the composition u/U
T_show = 50.0
m = t <= T_show
ax = fig.add_subplot(gs[1, :])
im = ax.imshow(p[:, m].T, extent=[0, 1, 0, T_show], origin='lower', aspect='auto', cmap='Greys',
               norm=mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=p[:, m].max()))
ax.set_ylim(T_show, 0)
ax.set_xlabel('phenotype $x$'); ax.set_ylabel('time $t$')
ax.set_title('phenotypic composition $u/U$: pulled to the right well during growth, then back to $\\psi$', **title_kw)
ax.text(-0.09, 1.05, 'c', transform=ax.transAxes, **panel_kw)
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label('$u(x,t)/U(t)$', fontsize=10)
cbar.ax.tick_params(labelsize=8)

row3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.45)

# (d) total population
ax = fig.add_subplot(row3[0])
m20 = t <= 20
ax.plot(t[m20], Uu[m20], color='0.15', lw=1.3)
ax.axhline(1, color='0.6', lw=0.8, ls=(0, (4, 3)))
ax.text(19.5, 1.03, '$U^*$', color='0.5', fontsize=9, ha='right')
ax.set_xlim(0, 20); ax.set_ylim(0, 1.15)
ax.set_xlabel('time $t$'); ax.set_ylabel('$U(t)$')
despine(ax)
ax.set_title('total population', **title_kw)
ax.text(-0.25, 1.08, 'd', transform=ax.transAxes, **panel_kw)

# (e) distance of the composition to psi
ax = fig.add_subplot(row3[1])
ax.semilogy(t, dist_u, color='0.15', lw=1.3)
k40 = np.argmin(abs(t - 40)); tr = np.linspace(25, 100, 10)
ax.semilogy(tr, 4 * dist_u[k40] * np.exp(-lam1 * (tr - 40)), color='0.55', lw=1.0, ls=(0, (1, 1.5)))
ax.text(62, 10 * dist_u[k40] * np.exp(-lam1 * 22), r'$\propto e^{-\lambda_1 t}$', fontsize=9.5, color='0.45')
ax.set_xlim(0, 100); ax.set_ylim(1e-3, 4)
ax.set_xlabel('time $t$'); ax.set_ylabel(r'$\|u/U-\psi\|_{L^1}$')
despine(ax)
ax.set_title('imprint of fitness', **title_kw)
ax.text(-0.25, 1.08, 'e', transform=ax.transAxes, **panel_kw)

# (f) memory time versus 1/D
ax = fig.add_subplot(row3[2])
ax.semilogy(1 / Ds, 1 / lams, 'o-', color='0.15', ms=2.5, lw=1.0)
ref = (1 / lams[0]) * np.exp(dP * (1 / Ds - 1 / Ds[0]))
ax.semilogy(1 / Ds, ref, color='0.55', lw=1.0, ls=(0, (1, 1.5)))
ax.text(80, ref[np.argmin(abs(1 / Ds - 80))] * 3, r'$\propto e^{\Delta P/D}$', fontsize=9.5,
        color='0.45', ha='right')
ax.plot([1 / D0], [1 / lam1], 'o', mfc='white', mec='0.15', mew=1.2, ms=6, zorder=5)
ax.annotate('a–e', xy=(1 / D0, 1 / lam1), xytext=(1 / D0 + 30, 1 / lam1 / 5), fontsize=9,
            color='0.3', ha='center', arrowprops=dict(arrowstyle='-', color='0.4', lw=0.7))
ax.set_xlim(0, 128)
ax.set_xlabel('$1/D$'); ax.set_ylabel(r'memory time $1/\lambda_1$')
despine(ax)
ax.set_title('memory time (Kramers)', **title_kw)
ax.text(-0.25, 1.08, 'f', transform=ax.transAxes, **panel_kw)

plt.savefig('fig2_fitness_imprint.pdf', bbox_inches='tight')
print("Saved fig2_fitness_imprint.pdf")
