"""
Table 5. Sequencing share of depletion under serial dependence, for the four model
portfolios at a withdrawal of 4 from W0 = 100 over T = 30.

Returns follow an AR(1), r_t = mu + phi (r_{t-1} - mu) + eps_t, with eps_t scaled so that
Var(r_t) = sigma^2. For each realized multiset, K orderings are drawn uniformly and weighted
by their AR(1) likelihood (self-normalized importance sampling); p_M is the weighted
depletion frequency and the order variance is p_M(1 - p_M). Shares are averaged over three
seeds, with the standard deviation across seeds in parentheses. The last columns give
Var[W_T] at phi = -0.2 and +0.2 relative to phi = 0, and the final two the median and
minimum effective sample size, (sum w)^2 / sum w^2, as a fraction of K.
"""
import numpy as np

def share_split(mu, sig, f, phi, W0=100.0, T=30, NM=2500, K=6000, seed=7):
    rng=np.random.default_rng(seed)
    se=sig*np.sqrt(1-phi**2) if phi!=0 else sig
    paths=np.empty((NM,T)); paths[:,0]=rng.normal(mu,sig,NM)
    for t in range(1,T): paths[:,t]=mu+phi*(paths[:,t-1]-mu)+rng.normal(0,se,NM)
    M=np.sort(paths,axis=1); pM=np.empty(NM); ess=np.empty(NM)
    for i in range(NM):
        m=M[i]; idx=np.argsort(rng.random((K,T)),axis=1); x=m[idx]
        W=np.full(K,W0); dead=np.zeros(K,bool)
        for t in range(T): W=W*(1+x[:,t])+f; dead|=(W<=0); W=np.maximum(W,0)
        R=dead.astype(float)
        if phi!=0:
            innov=x[:,1:]-mu-phi*(x[:,:-1]-mu)
            lw=-0.5*((x[:,0]-mu)**2/sig**2)-0.5*np.sum(innov**2,axis=1)/se**2
            lw-=lw.max(); w=np.exp(lw)
        else: w=np.ones(K)
        pM[i]=np.dot(w,R)/w.sum(); ess[i]=w.sum()**2/np.dot(w,w)
    seq=np.mean(pM*(1-pM)); lev=np.var(pM)
    return pM.mean()*100, seq/(seq+lev)*100, np.median(ess)/K, ess.min()/K

def var_ratio(mu,sig,f,phi,W0=100.0,T=30,N=1000000,seed=3):
    rng=np.random.default_rng(seed); se=sig*np.sqrt(1-phi**2) if phi!=0 else sig
    r=np.empty((N,T)); r[:,0]=rng.normal(mu,sig,N)
    for t in range(1,T): r[:,t]=mu+phi*(r[:,t-1]-mu)+rng.normal(0,se,N)
    W=np.full(N,W0)
    for t in range(T): W=W*(1+r[:,t])+f
    return np.var(W)

ports=[('conservative',0.04,0.08),('balanced',0.05,0.12),('growth',0.06,0.15),('aggressive',0.07,0.18)]
f=-4.0
phis=(-0.2,0.0,0.2)
print("Table 5: withdrawal 4, T = 30, W0 = 100")
print("share = sequencing share of depletion, mean (s.d.) over 3 seeds; [dep] = depletion probability")
print(f"{'portfolio':>12} | " + " | ".join(f"{'phi=%+.1f share [dep]' % p:>22}" for p in phis)
      + f" | {'V-0.2/0':>7} {'V+0.2/0':>7} | {'ESS/K med':>9} {'min':>6}")
for nm,mu,sig in ports:
    rows={p:[] for p in phis}; deps={p:[] for p in phis}; ess_med=[]; ess_min=[]
    for sd in (101,202,303):
        for phi in phis:
            d,s,em,emin=share_split(mu,sig,f,phi,seed=sd)
            rows[phi].append(s); deps[phi].append(d)
            if phi!=0.0: ess_med.append(em); ess_min.append(emin)
    v0=var_ratio(mu,sig,f,0.0); vm=var_ratio(mu,sig,f,-0.2); vp=var_ratio(mu,sig,f,+0.2)
    cells=[f"{np.mean(rows[p]):5.1f} ({np.std(rows[p]):.1f}) [{np.mean(deps[p]):4.1f}]" for p in phis]
    print(f"{nm:>12} | " + " | ".join(f"{c:>22}" for c in cells)
          + f" | {vm/v0:6.2f}x {vp/v0:6.2f}x | {np.median(ess_med):9.3f} {np.min(ess_min):6.3f}")
