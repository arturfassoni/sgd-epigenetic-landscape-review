"""
fig5_limit_cycle.py

Figure 5 of the review (Appendix A): under weighted competition the population
need not converge. Three states with cyclic switching 1 -> 2 -> 3 -> 1 at rate
0.2 and reverse switching at rate 0.001, proliferation rates r = (30, 0.01, 0.01),
competitive weights w = (0.001, 0.001, 1) and g(W) = 1 - W, W = w.u:

    u' = A u + g(W) diag(r) u.

The only positive equilibrium, u_eq = (500/501)(1, 1, 1), is unstable, and
trajectories approach an attracting limit cycle.

(a) Phase space (log10 u1, log10 u2, log10 u3): one trajectory starting near
    u_eq spirals out, one starting outside spirals in, both approach the cycle.
(b) Along the cycle, W oscillates across 1 (the zero of g), and the total
    population U = u1 + u2 + u3 oscillates between about 1.5 and 43.

Run with: python3 fig5_limit_cycle.py
Produces: fig5_limit_cycle.pdf (and prints the period and the range of U)
Dependencies: numpy, scipy, matplotlib.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

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
ACCENT = '#a3262a'
panel_kw = dict(fontsize=12, fontweight='bold', va='bottom', ha='right')

# ------------------------------------------------------------- model ----
k, eps = 0.2, 0.001
A = np.array([[-(k + eps), eps, k],
              [k, -(k + eps), eps],
              [eps, k, -(k + eps)]])          # columns sum to zero
r = np.array([30.0, 0.01, 0.01])
w = np.array([0.001, 0.001, 1.0])
u_eq = 500 / 501 * np.ones(3)


def rhs(t, u):
    return A @ u + (1 - w @ u) * r * u


def run(u0, T):
    s = solve_ivp(rhs, [0, T], u0, method='LSODA', rtol=1e-10, atol=1e-12,
                  dense_output=True)
    assert s.success
    return s.sol


# near the equilibrium (spirals out) and outside the cycle (spirals in)
sol_in = run(u_eq * np.array([1.02, 1.0, 0.99]), 250)
sol_out = run(np.array([0.3, 0.6, 2.2]), 250)
t_in = np.linspace(0, 250, 50001)
t_out = np.linspace(0, 60, 12001)

# the limit cycle itself, after the transient
sol_c = run(u_eq * 1.02, 700)
t_c = np.linspace(500, 700, 200001)
u_c = sol_c(t_c)
U_c, W_c = u_c.sum(0), w @ u_c
pk, _ = find_peaks(U_c)
period = np.diff(t_c[pk]).mean()
print(f"period of the cycle: {period:.2f}; U between {U_c.min():.2f} and {U_c.max():.1f}; "
      f"W between {W_c.min():.2f} and {W_c.max():.2f}")
m1 = (t_c >= t_c[pk[-2]]) & (t_c <= t_c[pk[-1]])        # one period

# ------------------------------------------------------------ figure ----
fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)

ax = fig.add_subplot(gs[0], projection='3d')
L = lambda sol, t: np.log10(sol(t))
li = L(sol_in, t_in)
lo = L(sol_out, t_out)
lc = np.log10(u_c[:, m1])
ax.plot(*li, color='0.4', lw=0.9, label=r'from near $u_{\rm eq}$ (spirals out)')
ax.plot(*lo, color='0.7', lw=0.9, label='from outside the cycle (spirals in)')
ax.plot(*lc, color='0.05', lw=2.6, label='attracting limit cycle')
ax.scatter(*np.log10(u_eq), color=ACCENT, marker='*', s=90, depthshade=False,
           label=r'unstable equilibrium $u_{\rm eq}$', zorder=10)
ax.set_xlabel(r'$\log_{10}u_1$')
ax.set_ylabel(r'$\log_{10}u_2$')
ax.set_zlabel(r'$\log_{10}u_3$')
ax.set_xticks([-1.5, 0, 1.5]); ax.set_yticks([-0.4, 0, 0.4, 0.8]); ax.set_zticks([0, 0.15, 0.3])
ax.tick_params(labelsize=8.5, pad=1)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor((1, 1, 1, 0))
    a.pane.set_edgecolor('0.85')
    a._axinfo['grid']['color'] = (0.9, 0.9, 0.9, 1)
ax.view_init(elev=22, azim=-60)
ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.12), frameon=False, fontsize=8.5)
ax.text2D(-0.02, 1.08, 'a', transform=ax.transAxes, **panel_kw)

ax = fig.add_subplot(gs[1])
sel = t_c <= t_c[0] + 62
ax.semilogy(t_c[sel] - t_c[0], U_c[sel], color='0.05', lw=1.4,
            label=r'$U=u_1+u_2+u_3$ (conserved by switching)')
ax.semilogy(t_c[sel] - t_c[0], W_c[sel], color='0.5', lw=1.4, ls=(0, (5, 2)),
            label=r'$W=w^{\top}u$ (enters competition)')
ax.axhline(1, color='0.6', lw=0.7, ls=(0, (1, 1.5)))
ax.text(61, 1.08, r'zero of $g$', color='0.5', fontsize=8.5, ha='right')
ax.set_xlim(0, 62)
ax.set_ylim(0.7, 200)
ax.set_xlabel(r'time $t-t_0$')
ax.set_ylabel('population (log scale)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc='upper right')
ax.text(-0.1, 1.03, 'b', transform=ax.transAxes, **panel_kw)

plt.savefig('fig5_limit_cycle.pdf', bbox_inches='tight')
print("Saved fig5_limit_cycle.pdf")
