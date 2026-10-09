"""
Closed forms for the sequencing / return-level split (Section 5), plus a brute-force
check on a few 6-period plans: simulate multisets and enumerate all 720 orderings of each.
"""
import numpy as np
from itertools import permutations
from math import comb


def total_var_exact(mu, sigma, f, W0):
    T = len(f); g = 1 + mu; s2 = sigma**2
    EW = W0; VW = 0.0
    for t in range(T):
        VW = (g*g + s2)*VW + s2*EW*EW
        EW = g*EW + f[t]
    return VW


def mean_path(mu, f, W0):
    g = 1 + mu; EW = [W0]
    for ft in f:
        EW.append(g*EW[-1] + ft)
    return np.array(EW)


def cov_ebar(a, b, T, g, s2):
    # Proposition 1
    if a > b:
        a, b = b, a
    g2s2 = g*g + s2
    tot = 0.0
    for c in range(a+1):
        tot += comb(a, c)*comb(T-a, b-c)*(g2s2**c)*(g**(a+b-2*c))
    return tot/comb(T, b) - g**(a+b)


def level_var_closed(mu, sigma, f, W0):
    T = len(f); g = 1 + mu; s2 = sigma**2
    c = np.empty(T+1); c[0] = W0; c[1:] = f
    val = 0.0
    for k in range(T+1):
        for l in range(T+1):
            val += c[k]*c[l]*cov_ebar(T-k, T-l, T, g, s2)
    return val


def mc_split(mu, sigma, f, W0, n_mult=4000, seed=0):
    rng = np.random.default_rng(seed); T = len(f)
    seq_terms = []; level_means = []
    perms = list(permutations(range(T)))
    for _ in range(n_mult):
        u = 1.0 + rng.normal(mu, sigma, T)
        Ws = np.empty(len(perms))
        for i, p in enumerate(perms):
            up = u[list(p)]
            W = W0
            for t in range(T):
                W = W*up[t] + f[t]
            Ws[i] = W
        seq_terms.append(Ws.var())       # Var_pi(W_T | M)
        level_means.append(Ws.mean())    # E_pi(W_T | M)
    s = np.array(seq_terms); m = np.array(level_means); n = len(s)
    seq, se_seq = s.mean(), s.std(ddof=1)/np.sqrt(n)
    lev = m.var()
    se_lev = np.sqrt(max(np.mean((m - m.mean())**4) - lev**2, 0.0)/n)
    return seq, se_seq, lev, se_lev


def Epp(k, l, T, g, s2):
    m = max(k, l)
    return (g*g + s2)**(T - m) * g**abs(k - l)


def seq_direct(mu, sigma, f, W0):
    # sequencing term directly, instead of total minus level
    T = len(f); g = 1 + mu; s2 = sigma**2
    c = np.empty(T+1); c[0] = W0; c[1:] = f
    val = 0.0
    for k in range(T+1):
        for l in range(T+1):
            Eee = cov_ebar(T-k, T-l, T, g, s2) + g**((T-k)+(T-l))
            val += c[k]*c[l]*(Epp(k, l, T, g, s2) - Eee)
    return val


if __name__ == "__main__":
    plans = [
        ("withdrawal 4, sigma 11%  ", 0.05, 0.11, [-4.0]*6, 100.0),
        ("withdrawal 6, sigma 15%  ", 0.05, 0.15, [-6.0]*6, 100.0),
        ("contribution 10, sigma 12%", 0.06, 0.12, [10.0]*6, 100.0),
        ("no cash flows            ", 0.05, 0.12, [0.0]*6, 100.0),
    ]
    print("Split of Var[W_T], T = 6, checked against all 6! orderings of each multiset")
    for name, mu, sg, f, W0 in plans:
        tot = total_var_exact(mu, sg, f, W0)
        lev = level_var_closed(mu, sg, f, W0)
        seq = tot - lev
        mc_seq, se_seq, mc_lev, se_lev = mc_split(mu, sg, f, W0, n_mult=3000, seed=1)
        sdir = seq_direct(mu, sg, f, W0)
        rho = seq/tot if tot else 0.0
        print(f"\n  {name}")
        print(f"    total          {tot:12.4f}")
        print(f"    return level   {lev:12.4f}   simulated {mc_lev:12.4f} (s.e. {se_lev:8.4f})")
        print(f"    sequencing     {abs(seq):12.4f}   simulated {mc_seq:12.4f} (s.e. {se_seq:8.4f})")
        print(f"    sequencing, direct closed form {abs(sdir):12.4f}   |diff| {abs(sdir-seq):.2e}")
        print(f"    sequencing share rho = {abs(rho)*100:6.3f}%")
