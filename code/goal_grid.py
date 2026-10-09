"""
Table 2. Sequencing share of goal attainment, rho_att, for the four model portfolios
across savings rates.

Career plan: one year's income already saved (W0 = 100) and a contribution of s% of an
income that starts at 100 and grows at 2% a year in real terms, over T = 40 years, so
f_t = s (1.02)^(t-1). The goal G = 3800 is 25 times a draw of 70% of final income,
100 (1.02)^39 = 216, following the four-percent rule.

For each multiset, K orderings are sampled; the order variance is the unbiased U-statistic
S(K-S)/[K(K-1)], where S counts orderings that reach G, and
rho_att = mean over multisets of that variance / [P(1-P)].
"""
import numpy as np

W0, gamma, T, G = 100.0, 0.02, 40, 3800.0
N, K = 20000, 200

def cell(mu, sigma, s, seed=11):
    rng = np.random.default_rng(seed)
    f = np.array([s * (1 + gamma) ** k for k in range(T)])
    ph = np.empty(N); oc = np.empty(N)
    for i in range(N):
        r = rng.normal(mu, sigma, T)
        idx = np.argsort(rng.random((K, T)), axis=1)
        u = 1.0 + r[idx]
        W = np.full(K, W0)
        for t in range(T):
            W = W * u[:, t] + f[t]
        S = float((W >= G).sum())
        ph[i] = S / K
        oc[i] = S * (K - S) / (K * (K - 1))
    P = ph.mean(); order = oc.mean(); tot = P * (1 - P)
    rho = order / tot * 100 if tot > 0 else 0.0
    return rho, P * 100

ports = [("(4%,8%)", 0.04, 0.08), ("(5%,12%)", 0.05, 0.12),
         ("(6%,15%)", 0.06, 0.15), ("(7%,18%)", 0.07, 0.18)]
rates = [10.0, 15.0, 20.0, 25.0]

print(f"Table 2: rho_att % (attainment probability %), G = {G:.0f}, T = {T}, income growth {gamma:.0%}")
print(f"{'rate':>6} | " + " | ".join(f"{tag:>13}" for tag, _, _ in ports))
for s in rates:
    out = []
    for tag, mu, sg in ports:
        rho, P = cell(mu, sg, s)
        out.append(f"{rho:4.1f} ({P:4.1f})")
    print(f"{s:5.0f}% | " + " | ".join(f"{c:>13}" for c in out))
