"""
Figure 5. For each 30-year window of real 60/40 returns, the window's returns are held
fixed and reordered under a real withdrawal of 4 with W0 = 100. Plots the 10th to 90th
percentile band and median of terminal wealth across reorderings, and terminal wealth
under the historical order, each divided by that window's median.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from historical import NOM_S as NOM, NOM_B, INF
figstyle.apply()

Y0 = 1928
ws = 0.60  # 60/40 stock/bond, annually rebalanced
nom_blend = ws*np.array(NOM) + (1 - ws)*np.array(NOM_B)
real = (1.0 + nom_blend/100.0)/(1.0 + np.array(INF)/100.0) - 1.0
W0, T, w, K = 100.0, 30, 0.04, 8000
rng = np.random.default_rng(4)

starts = np.arange(len(real) - T + 1)
years = Y0 + starts
med = np.empty(len(starts)); p10 = np.empty(len(starts)); p90 = np.empty(len(starts))
realized = np.empty(len(starts))
for i, s in enumerate(starts):
    ret = real[s:s+T]
    idx = np.argsort(rng.random((K, T)), axis=1); u = 1.0 + ret[idx]
    W = np.full(K, W0)
    for t in range(T):
        W = np.maximum(W*u[:, t] - w*W0, 0.0)
    med[i], p10[i], p90[i] = np.median(W), np.percentile(W, 10), np.percentile(W, 90)
    Wr = W0
    for r in ret:
        Wr = max(Wr*(1+r) - w*W0, 0.0)
    realized[i] = Wr

# divide by each window's own median so that windows share a common scale
lo, hi, rr = p10/med, p90/med, realized/med
print(f"p10/med range [{lo.min():.2f},{lo.max():.2f}], p90/med range [{hi.min():.2f},{hi.max():.2f}]")
print(f"realized/med: min {rr.min():.2f} at {years[rr.argmin()]}, median {np.median(rr):.2f}, max {rr.max():.2f}")
print(f"realized below its own 10th pct in {(realized < p10).sum()} of {len(starts)} windows")

fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 3.0))
figstyle.ygrid(ax)
ax.fill_between(years, lo, hi, color=figstyle.BAND, alpha=0.55, lw=0, zorder=1,
                label="reordered, 10th--90th percentile")
ax.axhline(1.0, color=figstyle.MED, lw=1.3, zorder=2, label="reordered, median")
ax.plot(years, rr, color=figstyle.NAVY, lw=0, marker="o", ms=3.6,
        markeredgecolor="white", markeredgewidth=0.5, clip_on=False, zorder=5,
        label="realized (historical) order")
ax.set_xlim(years[0], years[-1])
ax.set_ylim(-0.06, 2.2)
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("retirement start year")
ax.set_ylabel("final wealth $\\div$ median reordering")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=1.8,
          borderaxespad=0.3, labelspacing=0.4, ncol=1)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.145, top=0.975)
fig.savefig("historical_reorder.pdf")
print("wrote historical_reorder.pdf")
