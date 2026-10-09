# Table 1: sequencing share of depletion by portfolio and withdrawal rate, T = 30.
import numpy as np
W0 = 100.0
def ruin_count(returns, f, K, rng):
    T = len(returns); idx = np.argsort(rng.random((K, T)), axis=1)
    u = 1.0 + returns[idx]; W = np.full(K, W0); ruined = np.zeros(K, bool)
    for t in range(T):
        W = np.maximum(W * u[:, t] + f[t], 0.0); ruined |= (W <= 0.0)
    return int(ruined.sum())
def attribution(mu, sigma, w, T, N, K, seed):
    rng = np.random.default_rng(seed); f = np.full(T, -(w/100.0)*W0)
    phat = np.empty(N); oc = np.empty(N)
    for i in range(N):
        r = rng.normal(mu, sigma, T); S = ruin_count(r, f, K, rng)
        phat[i] = S/K; oc[i] = S*(K-S)/(K*(K-1))
    P = phat.mean(); order = oc.mean(); tot = P*(1-P)
    rho = order/tot*100 if tot > 0 else 0.0
    B=800; rs=np.empty(B)
    for b in range(B):
        s=rng.integers(0,N,N); Pb=phat[s].mean(); tb=Pb*(1-Pb)
        rs[b]=oc[s].mean()/tb*100 if tb>0 else 0.0
    return P*100, rho, np.std(rs)
if __name__ == "__main__":
    T, N, K = 30, 14000, 200
    ports = [("(4,8)",0.04,0.08),("(5,12)",0.05,0.12),("(6,15)",0.06,0.15),("(7,18)",0.07,0.18)]
    ws = [3.0,4.0,5.0,6.0]
    print("rho_dep % (s.e.) [depletion probability %], T = 30")
    print("  w  |"+"".join(f"{p[0]:>20}" for p in ports))
    for w in ws:
        cells=[]
        for _,mu,sg in ports:
            P,rho,sr=attribution(mu,sg,w,T,N,K,seed=7)
            cells.append(f"{rho:4.1f}({sr:.1f})[{P:4.1f}]")
        print(f" {w:.0f}% |"+"".join(f"{c:>20}" for c in cells))
