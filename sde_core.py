"""
sde_core.py

Shared simulation and plotting core for the phenotype-distribution examples
(Figure 4 of the review). Two functions:

  simulate_sde(...)  Euler-Maruyama simulation of a 1D Ito SDE
                      dX_t = drift_ito(X_t) dt + sqrt(2 D(X_t)) dW_t,
                      with optional reflecting boundaries.

  draw_case(...)      Draws one "case" into a pair of matplotlib axes:
                       (top) individual trajectories, time axis inverted so
                       that the panel's bottom edge is the final time and
                       lines up with the histogram directly below;
                       (bottom) histogram of final positions + analytic
                       stationary density u_eq(x) (left axis, solid) +
                       potential P(x) (right axis, dashed).

Used by fig4_examples.py to assemble the five-panel figure. Trajectory
lines are drawn with `rasterized=True`: with ~1000-1500 semi-transparent
lines per panel, a fully-vector PDF balloons to tens of MB; rasterizing
just those artists (while keeping axes/text/curves as vector) keeps the
output figure at a few MB with no visible loss of quality at print size.

Dependencies: numpy, matplotlib.
"""

import numpy as np


def simulate_sde(drift_ito, diffusivity, X0_sampler, T, N=1000, dt=0.02,
                  reflect_lo=None, reflect_hi=None, seed=42):
    """Euler-Maruyama simulation of dX = drift_ito(X) dt + sqrt(2 D(X)) dW."""
    rng = np.random.default_rng(seed)
    nsteps = int(T / dt)
    X = X0_sampler(rng, N)
    traj = np.zeros((nsteps + 1, N))
    traj[0] = X
    for i in range(1, nsteps + 1):
        dW = rng.normal(0, np.sqrt(dt), size=N)
        X = X + drift_ito(X) * dt + np.sqrt(2 * diffusivity(X)) * dW
        if reflect_lo is not None:
            below = X < reflect_lo
            X[below] = 2 * reflect_lo - X[below]
        if reflect_hi is not None:
            above = X > reflect_hi
            X[above] = 2 * reflect_hi - X[above]
        traj[i] = X
    tgrid = np.linspace(0, T, nsteps + 1)
    return tgrid, traj


def draw_case(ax_traj, ax_hist, tgrid, traj, u_eq, P, xmin, xmax,
              nbins=40, vline=None, vline_label=None, panel_label=None,
              traj_alpha=0.035, show_ylabels=True):
    """Draw the trajectories + histogram/stationary/potential panels into given axes."""
    T = tgrid[-1]
    N = traj.shape[1]
    final_positions = traj[-1]

    for j in range(N):
        ax_traj.plot(traj[:, j], tgrid, color='0.15', alpha=traj_alpha, linewidth=0.5,
                     rasterized=True)
    ax_traj.set_ylim(T, 0)
    ax_traj.set_xlim(xmin, xmax)
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    if show_ylabels:
        ax_traj.set_ylabel('time $t$')
    if vline is not None:
        ax_traj.axvline(vline, color='0.5', linewidth=0.6, linestyle=(0, (4, 3)))
        if vline_label:
            ax_traj.text(vline, -0.35 * T / 12, vline_label, ha='center', va='bottom',
                         fontsize=9, color='0.35')
    if panel_label:
        ax_traj.text(-0.02, 1.06, panel_label, transform=ax_traj.transAxes,
                     fontsize=13, fontweight='bold', va='bottom', ha='right')

    counts, bins, patches = ax_hist.hist(
        final_positions, bins=nbins, range=(xmin, xmax), density=True,
        color='0.82', edgecolor='0.3', linewidth=0.5
    )
    xs = np.linspace(xmin, xmax, 400)
    stat_curve = u_eq(xs)
    ax_hist.plot(xs, stat_curve, color='0.05', linewidth=1.4)
    ymax_hist = max(np.nanmax(stat_curve), counts.max()) * 1.15
    ax_hist.set_ylim(0, ymax_hist)
    ax_hist.set_xlim(xmin, xmax)
    ax_hist.set_xlabel('phenotype $x$')
    ax_hist.spines['top'].set_visible(False)
    if show_ylabels:
        ax_hist.set_ylabel('——  $u_{\\mathrm{eq}}(x)$', color='0.05')
    ax_hist.tick_params(axis='y', colors='0.15')

    ax_pot = ax_hist.twinx()
    Pvals = P(xs)
    ax_pot.plot(xs, Pvals, color='0.55', linewidth=1.0, linestyle=(0, (5, 2)))
    ax_pot.tick_params(axis='y', colors='0.45')
    ax_pot.spines['top'].set_visible(False)
    ax_pot.spines['right'].set_color('0.55')
    if show_ylabels:
        ax_pot.set_ylabel('- - -  $P(x)$', color='0.45')

    return ax_pot
