"""
Figure 3. Shortfall against a floor L = ell * W0 for three model portfolios at a 5%
withdrawal, T = 30. A path falls short when min_t W_t <= L, so recording each ordering's
running minimum gives the shortfall probability and its sequencing share at every floor
in one pass. Left panel: shortfall probability. Right panel: sequencing share.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

W0 = 100.0
N, K = 20000, 200
ells = np.linspace(0.0, 0.6, 31)
Ls = ells * W0

def sweep(mu, sigma, w, T, seed=7):
    rng = np.random.default_rng(seed)
    f = np.full(T, -w)
    phat = np.zeros((N, len(Ls))); ocon = np.zeros((N, len(Ls)))
    for i in range(N):
        r = rng.normal(mu, sigma, T)
        idx = np.argsort(rng.random((K, T)), axis=1)
        u = 1.0 + r[idx]
        W = np.full(K, W0); runmin = np.full(K, W0)
        for t in range(T):
            Wraw = W * u[:, t] + f[t]
            runmin = np.minimum(runmin, Wraw)
            W = np.maximum(Wraw, 0.0)
        S = (runmin[:, None] <= Ls[None, :]).sum(axis=0).astype(float)
        phat[i] = S / K
        ocon[i] = S * (K - S) / (K * (K - 1))
    P = phat.mean(0); order = ocon.mean(0); tot = P * (1 - P)
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
for (name, mu, sg), (c, dash) in zip(ports, figstyle.PORT3):
    P, rho = sweep(mu, sg, 5.0, 30)
    line, = ax1.plot(ells, P * 100, color=c, linestyle=dash, lw=1.4)
    ax2.plot(ells, rho * 100, color=c, linestyle=dash, lw=1.4)
    handles.append(line)

for ax, lab in ((ax1, "shortfall probability (%)"),
                (ax2, "sequencing share $\\rho_{\\mathrm{dep}}$ (%)")):
    ax.set_xlim(0, 0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("floor $\\ell$ (fraction of initial wealth)")
    ax.set_ylabel(lab)
    ax.spines[["top", "right"]].set_visible(False)
ax1.set_ylim(0, 80)
ax2.set_ylim(0, 60)

fig.legend(handles, [p[0] for p in ports], loc="upper center", ncol=3,
           frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=2.2, columnspacing=1.8,
           bbox_to_anchor=(0.5, 1.01))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.155, top=0.86, wspace=0.27)
fig.savefig("floor_share.pdf")
print("wrote floor_share.pdf")
