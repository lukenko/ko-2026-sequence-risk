# Numbers in Section 5.5: variance shares by portfolio and withdrawal rate at T = 30, and
# the CV of the exposure weights for withdrawals vs contributions.
import numpy as np
from verify_split import total_var_exact, level_var_closed, mean_path

W0, T = 100.0, 30
PORTFOLIOS = [("conservative", 0.04, 0.08), ("balanced", 0.05, 0.12),
              ("growth", 0.06, 0.15), ("aggressive", 0.07, 0.18)]


def seq_share(mu, sigma, f):
    tot = total_var_exact(mu, sigma, f, W0)
    lev = level_var_closed(mu, sigma, f, W0)
    return round(100*(tot - lev)/tot, 10) + 0.0   # avoid printing -0.0 when f = 0


def weights_cv(mu, f):
    g = 1 + mu
    B = mean_path(mu, f, W0)
    a = np.array([B[t-1]*g**(T-t) for t in range(1, T+1)])
    return a.std()/a.mean()


print(f"(1) Sequencing share of Var[W_T] (%), T = {T}, W0 = {W0:.0f}")
rates = (0, 3, 4, 5)
print(f"  {'portfolio':>12} " + " ".join(f"w={w}%".rjust(7) for w in rates))
for name, mu, sg in PORTFOLIOS:
    shares = [seq_share(mu, sg, [-float(w)]*T) for w in rates]
    print(f"  {name:>12} " + " ".join(f"{s:7.1f}" for s in shares))

print(f"\n(2) CV of exposure weights, aggressive portfolio, T = {T}")
mu, sg = 0.07, 0.18
for label, f in [("withdrawal 5", [-5.0]*T), ("contribution 5", [5.0]*T),
                 ("contribution 10", [10.0]*T), ("contribution 15", [15.0]*T)]:
    print(f"  {label:>16}  CV = {weights_cv(mu, f):5.2f}   share = {seq_share(mu, sg, f):5.1f}%")
