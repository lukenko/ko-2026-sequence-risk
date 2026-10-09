"""
Figure 1. Sequencing share of terminal-wealth variance against the cash-flow rate
(withdrawals negative, contributions positive) for horizons T = 10 to 50, aggressive
portfolio (mu = 7%, sigma = 18%). Exact closed forms.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_split import total_var_exact, level_var_closed
import figstyle
figstyle.apply()

W0, mu, sg = 100.0, 0.07, 0.18
rates = np.arange(-6.0, 12.01, 0.2)
horizons = [10, 20, 30, 40, 50]

def rho(rate, T):
    f = [rate]*T
    tot = total_var_exact(mu, sg, f, W0); lev = level_var_closed(mu, sg, f, W0)
    return (tot - lev) / tot * 100 if tot else 0.0

fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 3.0))
figstyle.ygrid(ax)
for T, (c, dash) in zip(horizons, figstyle.HORIZON5):
    y = np.array([rho(r, T) for r in rates])
    ax.plot(rates, y, color=c, linestyle=dash, lw=1.25, label=f"$T={T}$")

ax.axvline(0, color="0.55", lw=0.8, zorder=1)
ax.set_xlim(-6, 12.2)
ax.set_ylim(0, 19)
ax.set_xticks([-6, -3, 0, 3, 6, 9, 12])
ax.set_xlabel("cash-flow rate (% of initial wealth)")
ax.set_ylabel("sequencing share $\\rho$ (% of variance)")
ax.spines[["top", "right"]].set_visible(False)
ax.text(-3.5, 17.5, "withdrawal", fontsize=figstyle.FS_ANNOT, style="italic",
        color="0.30", ha="center")
ax.text(3.5, 17.5, "contribution", fontsize=figstyle.FS_ANNOT, style="italic",
        color="0.30", ha="center")
leg = ax.legend(loc="upper right", frameon=False, fontsize=figstyle.FS_LEGEND,
                handlelength=1.4, labelspacing=0.35, borderaxespad=0.4)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.96)
fig.savefig("phase_share.pdf")
print("wrote phase_share.pdf")
