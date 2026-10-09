"""
Section 9. Block bootstrap of the historical returns as an empirical check on the
variance share under serial dependence.

The split of unconstrained terminal wealth (withdrawal of 4 a year from W0 = 100 over
T = 30) is estimated two ways:
  (1) the closed form under independence, using the sample mean and variance;
  (2) a circular block bootstrap that builds 30-year paths from blocks of L consecutive
      years. For each path's multiset M, K uniform reorderings give E_pi[W_T | M] and
      Var_pi(W_T | M), and
          sequencing = mean over paths of Var_pi,  return level = variance of E_pi,
          rho = sequencing / (sequencing + return level).
With L = 1 the bootstrap is i.i.d. and should agree with (1); L > 1 keeps the serial
dependence of the data.
"""
import numpy as np
from math import comb
from historical import real_series

W0, T, draw = 100.0, 30, 4.0   # f_t = -draw each year (4% of W0)

# variance ratio of log returns over q years
def variance_ratio(r, q):
    x = np.log1p(r); n = len(x)
    v1 = x.var(ddof=1)
    xq = np.array([x[i:i+q].sum() for i in range(n-q+1)])
    return xq.var(ddof=1) / (q * v1)

# closed-form split under independence
def closed_form(mu, sig):
    g = 1.0 + mu; h = g*g + sig*sig
    f = np.full(T+1, -draw); c = f.copy(); c[0] = W0   # c_0=W0, c_k=f_k
    # mean path B_t = E[W_t]
    B = np.empty(T+1); B[0] = W0
    for t in range(1, T+1): B[t] = g*B[t-1] - draw
    # total variance, sum_t sig^2 B_{t-1}^2 h^(T-t)
    total = sum(sig*sig * B[t-1]**2 * h**(T-t) for t in range(1, T+1))
    # return level, sum_{k,l} c_k c_l Cov(ebar_{T-k}, ebar_{T-l})
    def cov(a, b):                                     # a<=b
        s = sum(comb(a, cc)*comb(T-a, b-cc) * h**cc * g**(a+b-2*cc) for cc in range(a+1))
        return s/comb(T, b) - g**(a+b)
    level = 0.0
    for k in range(T+1):
        for l in range(T+1):
            a, b = T-k, T-l
            if a > b: a, b = b, a
            level += c[k]*c[l]*cov(a, b)
    seq = total - level
    return total, level, seq, seq/total

# block bootstrap split
def boot_split(r, L, B=6000, K=240, seed=11):
    rng = np.random.default_rng(seed); n = len(r)
    means = np.empty(B); vars = np.empty(B)
    for b in range(B):
        # circular block bootstrap path of T returns
        path = np.empty(T); filled = 0
        while filled < T:
            start = rng.integers(0, n)
            take = min(L, T-filled)
            path[filled:filled+take] = r[(start + np.arange(take)) % n]
            filled += take
        # K uniform reorderings of this multiset
        idx = np.argsort(rng.random((K, T)), axis=1); u = 1.0 + path[idx]
        W = np.full(K, W0)
        for t in range(T): W = W*u[:, t] - draw
        means[b] = W.mean(); vars[b] = W.var()
    seq = vars.mean(); level = means.var(); total = seq + level
    return total, level, seq, seq/total

for ws, tag in [(1.00, "equity (100/0)"), (0.60, "60/40")]:
    r = real_series(ws); mu = r.mean(); sig = r.std(ddof=1)
    print(f"\n{tag}: real mean {mu*100:.1f}%, sd {sig*100:.1f}%, "
          f"VR(8) = {variance_ratio(r,8):.2f}, VR(10) = {variance_ratio(r,10):.2f}")
    tot, lev, seq, rho = closed_form(mu, sig)
    print(f"closed form (iid)   : rho = {rho*100:5.1f}%   (total {tot:.3g}, level {lev:.3g}, seq {seq:.3g})")
    for L in (1, 4, 8):
        tot, lev, seq, rho = boot_split(r, L)
        lab = "i.i.d. check" if L == 1 else "dependent"
        print(f"block bootstrap L={L:<2d}: rho = {rho*100:5.1f}%   (total {tot:.3g}, "
              f"level {lev:.3g}, seq {seq:.3g})   [{lab}]")
