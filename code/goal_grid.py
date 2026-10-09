"""
Table 2. Career plan: W0 = 100 (one year's income already saved), income starts at 100
and grows 2% a year, contribution = savings rate x income, T = 40. The goal is
25 x 70% of final income, about 3800.
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
