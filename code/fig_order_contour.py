# Figure 2: sequencing share over horizon and withdrawal rate, with the
# expected-depletion boundary.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from verify_split import total_var_exact, level_var_closed
import figstyle
figstyle.apply()

W0 = 100.0
Ts = np.arange(5, 51)
ws = np.arange(2.0, 8.01, 0.25)
PANELS = [("Conservative ($\\mu=4\\%$, $\\sigma=8\\%$)", 0.04, 0.08),
          ("Aggressive ($\\mu=7\\%$, $\\sigma=18\\%$)", 0.07, 0.18)]

def rho(mu, sg, w, T):
    f = [-w] * int(T)
    tot = total_var_exact(mu, sg, f, W0); lev = level_var_closed(mu, sg, f, W0)
    return (tot - lev) / tot * 100 if tot else 0.0

def wstar(mu, T):                       # withdrawal at which E[W_T] = 0
    g = 1 + mu
    return W0 * (g - 1) / (1 - g ** (-T))

# saved with bbox_inches="tight", which trims the canvas to about figstyle.WIDTH
fig, axes = plt.subplots(1, 2, figsize=(7.36, 2.15), sharey=True)
TT, WW = np.meshgrid(Ts, ws)
levels = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
cf = None
for ax, (title, mu, sg) in zip(axes, PANELS):
    Z = np.array([[rho(mu, sg, w, T) for T in Ts] for w in ws])
    wl = np.array([wstar(mu, T) for T in Ts])
    wlc = np.clip(wl, ws[0], ws[-1])
    cf = ax.contourf(TT, WW, Z, levels=levels, cmap="Blues", extend="max")
    ax.plot(Ts, wlc, color="#14181d", lw=1.5, ls=(0, (5, 2)), zorder=3,
            path_effects=[pe.withStroke(linewidth=2.9, foreground="white")])
    ax.text(Ts[-1], min(wl[-1], ws[-1]) - 0.3, "plan depletes", fontsize=figstyle.FS_ANNOT,
            ha="right", va="top", style="italic", color=figstyle.INK, zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(title, fontsize=figstyle.FS_LABEL, pad=5)
    ax.set_xlabel("horizon $T$ (years)")
    ax.set_xlim(Ts[0], Ts[-1]); ax.set_ylim(ws[0], ws[-1])
axes[0].set_ylabel("withdrawal rate $w$ (%)")
cb = fig.colorbar(cf, ax=axes, fraction=0.045, pad=0.02, ticks=[0, 20, 40, 60, 80])
cb.set_label(r"sequencing share $\rho$ (% of variance)", fontsize=figstyle.FS_LABEL)
cb.ax.tick_params(width=0.6, length=2.8, labelsize=figstyle.FS_TICK)
fig.savefig("order_limit.pdf", bbox_inches="tight")
print("wrote order_limit.pdf")
