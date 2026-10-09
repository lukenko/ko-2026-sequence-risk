"""
Section 8. Accumulation counterpart of Table 4: a level real contribution of 5 a year from
W0 = 100 over 30 years, with the same mean and volatility schedules. Reports the
probability of reaching a target set at expected terminal wealth under the falling
schedule (the variance-minimizing order for contributions, Proposition 4) and under its
time reverse. Linear model, no barrier.
"""
import numpy as np

T, W0, mu, f, N, SEED = 30, 100.0, 0.05, 5.0, 2_000_000, 5
g = 1 + mu
B = W0
for t in range(T):
    B = g * B + f
G = B                                   # target: expected terminal wealth
sig_up = np.linspace(0.08, 0.18, T)     # volatility rising from 8% to 18%
sig_dn = sig_up[::-1]                   # volatility falling from 18% to 8%


def attain(sig, seed=SEED):
    rng = np.random.default_rng(seed)
    W = np.full(N, W0)
    for t in range(T):
        W = W * (1 + rng.normal(mu, sig[t], N)) + f
    return (W >= G).mean()


p_up, p_dn = attain(sig_up), attain(sig_dn)
se = np.sqrt(0.25 / N)
print(f"T = {T}, W0 = {W0:.0f}, contribution {f:.0f}, mu = {mu:.0%}, target G = E[W_T] = {G:.1f}, N = {N:,}")
print(f"P(W_T >= G), volatility falling 18% to 8% = {100*p_dn:.2f}%")
print(f"P(W_T >= G), volatility rising 8% to 18%  = {100*p_up:.2f}%")
print(f"difference = {100*(p_dn-p_up):+.2f} percentage points (s.e. of each about {100*se:.2f})")
