"""
Section 6.1. Depletion split under normal and Student-t (4 degrees of freedom) returns
with the same mean and variance, for the four model portfolios at a 5% withdrawal,
T = 30, W0 = 100.
"""
import numpy as np
W0 = 100.0
def draws(kind, mu, sigma, shape, rng):
    if kind == "normal":
        return rng.normal(mu, sigma, shape)
    return mu + (sigma / np.sqrt(2.0)) * rng.standard_t(4, shape)   # t(4) has variance 2
def ruin_count(returns, f, K, rng):
    T = len(returns); idx = np.argsort(rng.random((K, T)), axis=1)
    u = 1.0 + returns[idx]; W = np.full(K, W0); ruined = np.zeros(K, bool)
    for t in range(T):
        W = np.maximum(W * u[:, t] + f[t], 0.0); ruined |= (W <= 0.0)
    return int(ruined.sum())
def attribution(kind, mu, sigma, w, T, N, K, seed):
    rng = np.random.default_rng(seed); f = np.full(T, -(w/100.0)*W0)
    phat = np.empty(N); oc = np.empty(N)
    for i in range(N):
        r = draws(kind, mu, sigma, T, rng); S = ruin_count(r, f, K, rng)
        phat[i] = S/K; oc[i] = S*(K-S)/(K*(K-1))
    P = phat.mean(); order = oc.mean(); tot = P*(1-P); rho = order/tot*100 if tot>0 else 0.0
    B=1000; ps=np.empty(B); rs=np.empty(B)
    for b in range(B):
        s=rng.integers(0,N,N); Pb=phat[s].mean(); tb=Pb*(1-Pb)
        ps[b]=Pb; rs[b]=oc[s].mean()/tb*100 if tb>0 else 0.0
    return P*100, rho, np.std(ps)*100, np.std(rs)
if __name__ == "__main__":
    T, N, K = 30, 16000, 200
    ports = [("Conservative 4/8", 0.04, 0.08), ("Balanced 5/12", 0.05, 0.12),
             ("Growth 6/15", 0.06, 0.15), ("Aggressive 7/18", 0.07, 0.18)]
    print(f"Normal vs Student-t(4), w = 5%, T = {T}: depletion probability P and rho_dep, "
          f"with differences (t minus normal)")
    for name, mu, sigma in ports:
        Pn, rn, spn, srn = attribution("normal", mu, sigma, 5.0, T, N, K, seed=5)
        Pt, rt, spt, srt = attribution("t",      mu, sigma, 5.0, T, N, K, seed=5)
        print(f"  {name:16} normal P={Pn:5.2f} rho={rn:5.1f} | t P={Pt:5.2f} rho={rt:5.1f}"
              f" | dP={Pt-Pn:+5.2f} drho={rt-rn:+5.1f} (se_rho~{srn:.1f})")
