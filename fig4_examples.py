"""
fig4_examples.py

Assembles Figure 4 of the review: quantitative validation of the four
phenotypic-distribution examples of Section 2.8 (uniform, exponential,
Gaussian, heavy-tailed) plus a fifth, wide panel showing the general
Boltzmann-form distribution on a rugged multi-well landscape -- the same
landscape sketched schematically in Figure 1, panels e/f, and used in Figure 3.

Panel (e) also shows, in colour, one trajectory of a "more plastic" cell:
the same SDE with a larger, uniform diffusivity D_hi. For illustration it is a fast
cell (arrival near t=4, before any grey cell), drawn (every 0.1 time units) only until it first
reaches the bottom of the far well.
Median first-passage times for both noise levels are estimated from
independent ensembles and printed (they are quoted in the caption).

Layout: a 3-row grid. Rows 1-2 hold two side-by-side cases each (panels
a-d); row 3 holds one full-width case (panel e). Each case is itself a
2-row stack built by simulate_sde() + draw_case() from sde_core.py:
trajectories on top, histogram/stationary-density/potential below.

Run with: python3 fig4_examples.py   (sde_core.py must be in the same folder)
Produces: fig4_examples.pdf (and prints the first-passage statistics quoted in the text)

Dependencies: numpy, scipy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline
from sde_core import simulate_sde, draw_case

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

ACCENT = '#a3262a'   # single accent colour: the more plastic cell in panel (e)

fig = plt.figure(figsize=(11.5, 13.8))
outer = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.05], hspace=0.34, figure=fig)

def make_stack(subplot_spec, height_ratios=(2.0, 1.15)):
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec,
                                           height_ratios=height_ratios, hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    return ax_top, ax_bot

row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.30)
row3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])

axa = make_stack(row1[0])
axb = make_stack(row1[1])
axc = make_stack(row2[0])
axd = make_stack(row2[1])
axe = make_stack(row3[0], height_ratios=(1.7, 1.0))

# ============================================================
# Case (a): Flat landscape -> Uniform
# ============================================================
D1 = 1.0
L = 10.0
tgrid, traj = simulate_sde(
    drift_ito=lambda x: np.zeros_like(x),
    diffusivity=lambda x: np.full_like(x, D1),
    X0_sampler=lambda rng, N: np.full(N, 5.0),
    T=15.0, N=1000, dt=0.01, reflect_lo=0.0, reflect_hi=L, seed=1
)
draw_case(axa[0], axa[1], tgrid, traj,
          u_eq=lambda x: np.full_like(x, 1.0 / L),
          P=lambda x: np.zeros_like(x),
          xmin=0, xmax=L, panel_label='a')

# ============================================================
# Case (b): Linear potential -> Exponential
# ============================================================
a2, D2 = 1.0, 1.0
lam = a2 / D2
tgrid, traj = simulate_sde(
    drift_ito=lambda x: -a2 * np.ones_like(x),
    diffusivity=lambda x: np.full_like(x, D2),
    X0_sampler=lambda rng, N: np.full(N, 5.0),
    T=10.0, N=1000, dt=0.01, reflect_lo=0.0, seed=2
)
draw_case(axb[0], axb[1], tgrid, traj,
          u_eq=lambda x: lam * np.exp(-lam * x),
          P=lambda x: a2 * x,
          xmin=0, xmax=8, panel_label='b')

# ============================================================
# Case (c): Harmonic potential -> Gaussian
# ============================================================
theta3, mu3, D3 = 0.4, 3.0, 1.0
sigma2_3 = D3 / theta3
from scipy.stats import norm as normal_dist
tgrid, traj = simulate_sde(
    drift_ito=lambda x: -theta3 * (x - mu3),
    diffusivity=lambda x: np.full_like(x, D3),
    X0_sampler=lambda rng, N: np.full(N, -3.0),
    T=12.0, N=1000, dt=0.02, seed=3
)
draw_case(axc[0], axc[1], tgrid, traj,
          u_eq=lambda x: normal_dist.pdf(x, loc=mu3, scale=np.sqrt(sigma2_3)),
          P=lambda x: 0.5 * theta3 * (x - mu3) ** 2,
          xmin=-5, xmax=11, vline=mu3, vline_label=r'$\mu$', panel_label='c')

# ============================================================
# Case (d): State-dependent noise -> heavy-tailed
# ============================================================
D0, theta4 = 0.3, 0.65

def D4(x): return D0 * (1 + x**2)
def drift4(x): return -theta4 * x + 2 * D0 * x

p4 = theta4 / (2 * D0)
xx = np.linspace(-400, 400, 800001)
Z4 = np.trapezoid((1 + xx**2) ** (-p4), xx)

tgrid, traj = simulate_sde(
    drift_ito=drift4, diffusivity=D4,
    X0_sampler=lambda rng, N: np.full(N, -4.0),
    T=12.0, N=1000, dt=0.004, seed=4
)
draw_case(axd[0], axd[1], tgrid, traj,
          u_eq=lambda x: (1 + x**2) ** (-p4) / Z4,
          P=lambda x: 0.5 * theta4 * x**2,
          xmin=-15, xmax=15, vline=0.0, vline_label='$0$', panel_label='d')

# ============================================================
# Case (e): General rugged landscape, multiple minima -- Boltzmann form
# reusing the same landscape shape as Figures 1 and 3
# ============================================================
pts_x = np.array([-3.0, -2.4, -1.8, -1.2, -0.5, 0.3, 1.0, 1.6, 2.2, 2.8, 3.0])
pts_y = np.array([2.2, 1.3, 0.5, 1.4, 0.6, -0.3, 0.9, 1.6, 0.9, 1.8, 2.3])
Pspline = CubicSpline(pts_x, pts_y)
Pderiv = Pspline.derivative()
D5 = 0.35
T5 = 25.0

def P5(x):
    return Pspline(np.clip(x, pts_x[0], pts_x[-1]))

def drift5(x):
    xc = np.clip(x, pts_x[0], pts_x[-1])
    return -Pderiv(xc)

xs_wide = np.linspace(-3.0, 3.0, 4000)
Z5 = np.trapezoid(np.exp(-P5(xs_wide) / D5), xs_wide)

def ueq5(x):
    return np.exp(-P5(x) / D5) / Z5

tgrid, traj = simulate_sde(
    drift_ito=drift5, diffusivity=lambda x: np.full_like(x, D5),
    X0_sampler=lambda rng, N: np.full(N, -3.0),
    T=T5, N=1500, dt=0.01, seed=5
)
ax_pot_e = draw_case(axe[0], axe[1], tgrid, traj, u_eq=ueq5, P=P5,
                      xmin=-3.0, xmax=3.0, nbins=50, panel_label='e',
                      traj_alpha=0.02)

# --- more plastic cell: same SDE, larger uniform diffusivity ---
D_hi = 0.7
x_far = 2.0          # entrance of the far (right-hand) well


def first_passage_times(D, T, N, seed, dt=0.01):
    """Ensemble estimate of the first-passage time from x=-3 to x >= x_far (reflecting
    boundaries at x=-3, 3); NaN for cells that have not arrived by time T."""
    tg, tr = simulate_sde(drift_ito=drift5, diffusivity=lambda x: np.full_like(x, D),
                          X0_sampler=lambda rng, N: np.full(N, -3.0),
                          T=T, N=N, dt=dt, reflect_lo=-3.0, reflect_hi=3.0, seed=seed)
    reached = tr >= x_far
    idx = reached.argmax(axis=0)
    tau = np.where(reached.any(axis=0), tg[idx], np.nan)
    return tau, tg, tr


tau_lo, _, _ = first_passage_times(D5, T=2500.0, N=1000, seed=11, dt=0.02)
tau_hi, tg_hi, tr_hi = first_passage_times(D_hi, T=T5, N=400, seed=12)
tau_hi_long, _, _ = first_passage_times(D_hi, T=600.0, N=1000, seed=13, dt=0.02)
med_lo, med_hi = np.nanmedian(tau_lo), np.nanmedian(tau_hi_long)
print(f"median first-passage time to x>={x_far}: D={D5} -> {med_lo:.0f} "
      f"(not arrived by T: {np.isnan(tau_lo).mean():.3f});  D={D_hi} -> {med_hi:.0f} "
      f"(not arrived: {np.isnan(tau_hi_long).mean():.3f})")

# illustrative realisation: a fast cell of the D_hi batch, arriving at the far well before
# every cell of the grey population (whose earliest arrival is t ~ 5) yet slowly enough for
# its path through the wells to be visible (target arrival t_target); the trajectory is drawn
# only until it first reaches the bottom of the far well, so that it does not hide the
# grey population afterwards. Statistics (medians, quartiles) are printed for the caption.
q25, q75 = np.nanpercentile(tau_hi_long, [25, 75])
t_target = 4.0
j = np.nanargmin(np.abs(tau_hi - t_target))
x_well = 2.23                                  # bottom of the far well
k_end = np.argmax(tr_hi[:, j] >= x_well)
tau_lo_pop = first_passage_times(D5, T=T5, N=1500, seed=5)[0]   # grey population itself
pct = 100 * np.nanmean(tau_hi_long <= tau_hi[j])
print(f"D={D_hi}: quartiles of first-passage time {q25:.0f}-{q75:.0f}; chosen cell (percentile "
      f"{pct:.0f} of the ensemble) reaches x>={x_far} at t={tau_hi[j]:.1f}, the well bottom at "
      f"t={tg_hi[k_end]:.1f}")
print(f"grey population (D={D5}): {np.sum(~np.isnan(tau_lo_pop))} of 1500 cells reach x>={x_far} "
      f"by t={T5:.0f}; earliest at t={np.nanmin(tau_lo_pop):.1f}")
stride = 10                                    # display every 0.1 time units
xs_r = np.append(tr_hi[:k_end:stride, j], tr_hi[k_end, j])
ts_r = np.append(tg_hi[:k_end:stride], tg_hi[k_end])
axe[0].plot(xs_r, ts_r, color=ACCENT, lw=0.9, alpha=0.95, zorder=5)
axe[0].plot(tr_hi[0, j], 0, 'o', color=ACCENT, ms=4, zorder=6)
axe[0].plot(xs_r[-1], ts_r[-1], 'o', color=ACCENT, ms=4, zorder=6)
axe[0].text(0.99, 0.99, rf'population, $D={D5}$', transform=axe[0].transAxes, color='0.35',
            fontsize=9.5, ha='right', va='top')
axe[0].text(0.99, 0.915, rf'one cell, $D={D_hi}$', transform=axe[0].transAxes, color=ACCENT,
            fontsize=9.5, ha='right', va='top')

for ax_pair in (axa, axb, axc, axd, axe):
    ax_pair[1].set_ylabel('——  $u_{\\mathrm{eq}}(x)$', color='0.05', fontsize=9.5)

plt.savefig('fig4_examples.pdf', bbox_inches='tight', dpi=300)
print("Saved fig4_examples.pdf")
