"""
Figure 4. Goal attainment as the target G varies, for three model portfolios. Same career
plan as Table 2 with a 15% savings rate. G is expressed as a multiple of final income,
100 (1.02)^39; the dashed line marks the Table 2 goal of 17.5 times final income. Left
panel: attainment probability. Right panel: sequencing share.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle
figstyle.apply()

W0, c, gamma, T = 100.0, 15.0, 0.02, 40
N, K = 40000, 200
income_T = 100.0 * (1 + gamma) ** (T - 1)
mults = np.round(np.arange(5.0, 25.01, 0.5), 2)
Gs = mults * income_T

def sweep(mu, sigma, seed=11):
    rng = np.random.default_rng(seed)
    f = np.array([c * (1 + gamma) ** k for k in range(T)])
    ph = np.zeros((N, len(Gs))); oc = np.zeros((N, len(Gs)))
    for i in range(N):
        r = rng.normal(mu, sigma, T)
        idx = np.argsort(rng.random((K, T)), axis=1)
        u = 1.0 + r[idx]
        W = np.full(K, W0)
        for t in range(T):
            W = W * u[:, t] + f[t]
        S = (W[:, None] >= Gs[None, :]).sum(axis=0).astype(float)
        ph[i] = S / K
        oc[i] = S * (K - S) / (K * (K - 1))
    P = ph.mean(0); order = oc.mean(0); tot = P * (1 - P)
    rho = np.where(tot > 0, order / tot, 0.0)
    return P, rho

ports = [
    ("Conservative $(4\\%,8\\%)$", 0.04, 0.08),
    ("Balanced $(5\\%,12\\%)$",   0.05, 0.12),
    ("Aggressive $(7\\%,18\\%)$", 0.07, 0.18),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.WIDTH, 2.5))
figstyle.ygrid(ax1); figstyle.ygrid(ax2)
handles = []
for (name, mu, sg), (col, dash) in zip(ports, figstyle.PORT3):
    P, rho = sweep(mu, sg)
    line, = ax1.plot(mults, P * 100, color=col, linestyle=dash, lw=1.4)
    ax2.plot(mults, rho * 100, color=col, linestyle=dash, lw=1.4)
    handles.append(line)

for ax, lab in ((ax1, "attainment probability (%)"),
                (ax2, "sequencing share $\\rho_{\\mathrm{att}}$ (%)")):
    ax.axvline(3800.0 / income_T, color="0.55", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(5, 25)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xlabel("target $G$ (multiple of final income)")
    ax.set_ylabel(lab)
    ax.spines[["top", "right"]].set_visible(False)
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 50)

fig.legend(handles, [p[0] for p in ports], loc="upper center", ncol=3,
           frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=2.2, columnspacing=1.8,
           bbox_to_anchor=(0.5, 1.01))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.155, top=0.86, wspace=0.27)
fig.savefig("target_share.pdf")
print("wrote target_share.pdf")
