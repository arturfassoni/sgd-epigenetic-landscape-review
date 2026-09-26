"""
fig3_pde.py

Figure 3 of the review (Section 2.6): the full nonlinear reaction-advection-
diffusion equation

    d_t u = r(x) u (1 - U(t)/K) - d_x(v u) + d_x(D d_x u),   U(t) = int u dx,

with v = -P' on the multi-well landscape, converges to the stationary density
of the switching dynamics, u_eq(x) = C exp(-P(x)/D), although the proliferation
rate r(x) = r0 - m x favors the left edge, where the population starts
(Theorem 2 of the review, proved in the companion paper).

Method of lines: the PDE is discretized on N = 150 nodes with zero-flux
boundaries, which gives a compartmental ODE system of the type covered by
Theorem 1, integrated with a stiff (BDF) solver. Initial condition: a narrow
Gaussian at the left edge, x0 = -3, with total mass U(0) = 0.05.

Landscape P(x): cubic spline through fixed control points, the same as in
Figure 1(e,f) and Figure 4(e).

Run with: python3 fig3_pde.py
Produces: fig3_pde.pdf (and prints the L1 distance to u_eq at t = 100)
Dependencies: numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------- style ----
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

# ---------------------------------------------------------- landscape P(x) --
# same control points as Figures 1(e,f) and 4(e)
pts_x = np.array([-3.0, -2.4, -1.8, -1.2, -0.5, 0.3, 1.0, 1.6, 2.2, 2.8, 3.0])
pts_y = np.array([2.2, 1.3, 0.5, 1.4, 0.6, -0.3, 0.9, 1.6, 0.9, 1.8, 2.3])
Pspline = CubicSpline(pts_x, pts_y)
Pderiv = Pspline.derivative()

D = 0.35            # constant diffusivity, matching Figure 4(e)
xmin, xmax = -3.0, 3.0

def P(x):
    return Pspline(x)

def v(x):
    return -Pderiv(x)

# ---------------------------------------------------- reaction kinetics ----
r0, m = 1.0, 0.15          # r(x) = r0 - m x : decreasing, always positive on [-3,3]
K = 1.0                    # carrying capacity
U0_target = 0.05           # initial total mass (well below K)

def r_of_x(x):
    return r0 - m * x

# ------------------------------------------------------------- grid -------
N = 150
x = np.linspace(xmin, xmax, N)
dx = x[1] - x[0]
x_face = 0.5 * (x[:-1] + x[1:])          # N-1 interior cell faces
v_face = v(x_face)
r_vec = r_of_x(x)

# ------------------------------------------------- initial condition ------
x0 = xmin
sigma0 = 0.25
u0 = np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2))
U0_raw = np.trapezoid(u0, x)
u0 *= U0_target / U0_raw   # normalize total initial mass

# ------------------------------------------------- method of lines --------
def rhs(t, u):
    U = np.trapezoid(u, x)
    g = 1.0 - U / K
    reaction = r_vec * u * g
    u_face = 0.5 * (u[:-1] + u[1:])            # centered advective flux
    du_face = (u[1:] - u[:-1]) / dx            # centered diffusive gradient
    J = v_face * u_face - D * du_face          # flux at N-1 interior faces
    J_full = np.concatenate(([0.0], J, [0.0]))  # zero-flux at both boundaries
    transport = -(J_full[1:] - J_full[:-1]) / dx
    return reaction + transport

T_final = 100.0
t_eval = np.linspace(0, T_final, 500)

sol = solve_ivp(rhs, [0, T_final], u0, t_eval=t_eval, method='BDF',
                 rtol=1e-8, atol=1e-10)
print("solver success:", sol.success, "| status:", sol.message)

U_t = np.trapezoid(sol.y, x, axis=0)

# ------------------------------------------------ theoretical prediction --
xs_fine = np.linspace(xmin, xmax, 800)
ueq_unnorm = np.exp(-P(xs_fine) / D)
Z = np.trapezoid(ueq_unnorm, xs_fine)
ueq = ueq_unnorm / Z   # normalized (probability) shape

u_final = sol.y[:, -1]
u_final_normalized = u_final / U_t[-1]

# =====================================================================
# figure
# =====================================================================
fig = plt.figure(figsize=(9.5, 9.5))
gs = GridSpec(3, 2, height_ratios=[0.85, 1.6, 1.05], width_ratios=[1, 1],
              hspace=0.5, wspace=0.32, figure=fig)

panel_kw = dict(fontsize=12, fontweight='bold', va='bottom', ha='right')
title_kw = dict(loc='left', fontsize=10, color='0.25', pad=8)

# --- (a) landscape, for reference ---
ax_land = fig.add_subplot(gs[0, 0])
ax_land.plot(xs_fine, P(xs_fine), color='0.15', lw=1.3)
ax_land.set_xlabel('phenotype $x$')
ax_land.set_ylabel('$P(x)$')
ax_land.set_xlim(xmin, xmax)
ax_land.spines['top'].set_visible(False)
ax_land.spines['right'].set_visible(False)
ax_land.set_title('epigenetic landscape (multi-well)', **title_kw)
ax_land.text(-0.18, 1.08, 'a', transform=ax_land.transAxes, **panel_kw)

# --- (b) r(x) ---
ax_r = fig.add_subplot(gs[0, 1])
ax_r.plot(x, r_vec, color='0.35', lw=1.3)
ax_r.set_xlabel('phenotype $x$')
ax_r.set_ylabel('$r(x)$')
ax_r.set_xlim(xmin, xmax)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)
ax_r.set_title('proliferation rate (highest at the left edge)', **title_kw)
ax_r.text(-0.18, 1.08, 'b', transform=ax_r.transAxes, **panel_kw)
ax_r.text(0.97, 0.90, r'$r(x)=r_0-mx$', transform=ax_r.transAxes,
          fontsize=9, ha='right', va='top')
import matplotlib.colors as mcolors

# --- (c) kymograph of u(x,t), extended to t in [0,50], time inverted ---
T_show = 50.0
show_mask = t_eval <= T_show
ax_kymo = fig.add_subplot(gs[1, :])
extent = [xmin, xmax, 0, T_show]
im = ax_kymo.imshow(sol.y[:, show_mask].T, extent=extent, origin='lower',
                     aspect='auto', cmap='Greys',
                     norm=mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=sol.y[:, show_mask].max()))
ax_kymo.set_ylim(T_show, 0)   # invert: t=0 at top, matching the SDE trajectory figures
ax_kymo.set_xlabel('phenotype $x$')
ax_kymo.set_ylabel('time $t$')
ax_kymo.set_title('population density $u(x,t)$, seeded at the left edge', **title_kw)
ax_kymo.text(-0.09, 1.05, 'c', transform=ax_kymo.transAxes, **panel_kw)
cbar = fig.colorbar(im, ax=ax_kymo, pad=0.02, fraction=0.035)
cbar.set_label('$u(x,t)$', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# --- (d) U(t) growth (full simulated range) ---
ax_U = fig.add_subplot(gs[2, 0])
ax_U.plot(t_eval, U_t, color='0.15', lw=1.3)
ax_U.axhline(K, color='0.6', lw=0.8, linestyle=(0, (4, 3)))
ax_U.text(T_final * 0.65, K * 1.03, '$K$', color='0.5', fontsize=9)
ax_U.set_xlabel('time $t$')
ax_U.set_ylabel(r'$U(t)=\int\,u(x,t)\,dx$')
ax_U.spines['top'].set_visible(False)
ax_U.spines['right'].set_visible(False)
ax_U.set_title('total population', **title_kw)
ax_U.text(-0.18, 1.08, 'd', transform=ax_U.transAxes, **panel_kw)

# --- (e) final profile vs. theoretical linear-PDE equilibrium ---
ax_final = fig.add_subplot(gs[2, 1])
ax_final.plot(xs_fine, ueq, color='0.05', lw=1.6, label='predicted s.s. PDE')
ax_final.plot(x, u_final_normalized, color='0.75', lw=1.4, linestyle=(0, (5, 2)),
              label=f'simulated, $t={T_final:.0f}$')
ax_final.set_xlabel('phenotype $x$')
ax_final.set_ylabel(r'$u(x,T)/U(T)$')
ax_final.legend(fontsize=8, frameon=False, loc='upper left')
ax_final.spines['top'].set_visible(False)
ax_final.spines['right'].set_visible(False)
ax_final.set_title('final profile vs. stationary density', **title_kw)
ax_final.text(-0.18, 1.08, 'e', transform=ax_final.transAxes, **panel_kw)

plt.savefig('fig3_pde.pdf', bbox_inches='tight')
print("Saved fig3_pde.pdf")

# quick numeric check of agreement
from scipy.interpolate import interp1d
ueq_on_x = interp1d(xs_fine, ueq)(x)
l1_error = np.trapezoid(np.abs(u_final_normalized - ueq_on_x), x)
print(f"L1 distance between simulated and predicted shape at t={T_final}: {l1_error:.4f}")
print(f"U(0) = {U_t[0]:.4f}, U(T) = {U_t[-1]:.4f}, K = {K}")
