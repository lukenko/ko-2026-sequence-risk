"""
Table 4. Depletion probability and terminal-wealth variance under three volatility
schedules with the same mean (5%) and average volatility (13%): rising from 8% to 18%,
constant at 13%, and falling from 18% to 8%. Thirty-year decumulation with a withdrawal of
5 from W0 = 100. Depletion uses the absorbing barrier at zero; the variance is exact, from
the per-period decomposition of the linear model.
"""
import numpy as np

def ruin_prob(mu, sig, f, W0, N=4_000_000, seed=0):
    rng = np.random.default_rng(seed); T = len(f)
    W = np.full(N, float(W0)); ruined = np.zeros(N, bool)
    for t in range(T):
        W = W*(1+rng.normal(mu[t], sig[t], N)) + f[t]
        W = np.maximum(W, 0.0); ruined |= (W <= 0)
    return ruined.mean()

def var_terminal(mu, sig, f, W0):
    """Var[W_T] = sum_t sigma_t^2 B_{t-1}^2 Gamma_t (linear model, no barrier)."""
    T = len(f); B = [float(W0)]
    for t in range(T - 1):
        B.append((1 + mu[t]) * B[-1] + f[t])
    return sum(sig[t]**2 * B[t]**2 * np.prod([(1 + mu[j])**2 + sig[j]**2 for j in range(t + 1, T)])
               for t in range(T))

if __name__ == "__main__":
    T = 30; mu = [0.05]*T; f = [-5.0]*T; W0 = 100.0
    lo, hi = 0.08, 0.18
    schedules = {
        "low volatility early, 8% to 18%":  list(np.linspace(lo, hi, T)),
        "constant, 13%":                    [0.13]*T,
        "high volatility early, 18% to 8%": list(np.linspace(hi, lo, T)),
    }
    print(f"Table 4: T = {T}, withdrawal 5, mu = 5%, W0 = 100")
    for nm, sg in schedules.items():
        print(f"  {nm:34s}  P(depletion) = {ruin_prob(mu, sg, f, W0, seed=3)*100:5.2f}%"
              f"   Var[W_T] = {var_terminal(mu, sg, f, W0):,.0f}")
