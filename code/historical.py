"""
Section 7. Sequence risk in U.S. returns, 1928-2025: summary statistics of the 60/40
portfolio, the 1966 cohort, and Table 3.

A 30-year window's returns are held fixed and reordered, so only their order varies. A
portfolio's nominal return is the fixed-weight average of the stock and bond returns
(annual rebalancing), and its real return is (1 + nominal)/(1 + inflation) - 1.

Data: S&P 500 and ten-year U.S. Treasury total returns (Damodaran, NYU Stern, "Historical
Returns on Stocks, Bonds and Bills", January 2026) and annual-average U.S. CPI inflation
(BLS), all in percent. The same series are in data/us_annual_returns_1928_2025.csv.
"""
import numpy as np

# S&P 500 total return (%)
NOM_S = np.array([43.81,-8.30,-25.12,-43.84,-8.64,49.98,-1.19,46.74,31.94,-35.34,29.28,-1.10,
-10.67,-12.77,19.17,25.06,19.03,35.82,-8.43,5.20,5.70,18.30,30.81,23.68,18.15,-1.21,
52.56,32.60,7.44,-10.46,43.72,12.06,0.34,26.64,-8.81,22.61,16.42,12.40,-9.97,23.80,
10.81,-8.24,3.56,14.22,18.76,-14.31,-25.90,37.00,23.83,-6.98,6.51,18.52,31.74,-4.70,
20.42,22.34,6.15,31.24,18.49,5.81,16.54,31.48,-3.06,30.23,7.49,9.97,1.33,37.20,22.68,
33.10,28.34,20.89,-9.03,-11.85,-21.97,28.36,10.74,4.83,15.61,5.48,-36.55,25.94,14.82,
2.10,15.89,32.15,13.52,1.38,11.77,21.61,-4.23,31.21,18.02,28.47,-18.04,26.06,24.88,17.78])
# ten-year U.S. Treasury total return (%)
NOM_B = np.array([0.84,4.20,4.54,-2.56,8.79,1.86,7.96,4.47,5.02,1.38,4.21,4.41,5.40,-2.02,2.29,
2.49,2.58,3.80,3.13,0.92,1.95,4.66,0.43,-0.30,2.27,4.14,3.29,-1.34,-2.26,6.80,-2.10,-2.65,
11.64,2.06,5.69,1.68,3.73,0.72,2.91,-1.58,3.27,-5.01,16.75,9.79,2.82,3.66,1.99,3.61,15.98,
1.29,-0.78,0.67,-2.99,8.20,32.81,3.20,13.73,25.71,24.28,-4.96,8.22,17.69,6.24,15.00,9.36,
14.21,-8.04,23.48,1.43,9.94,14.92,-8.25,16.66,5.57,15.12,0.38,4.49,2.87,1.96,10.21,20.10,
-11.12,8.46,16.04,2.97,-9.10,10.75,1.28,0.69,2.80,-0.02,9.64,11.33,-4.42,-17.83,3.88,-1.64,7.80])
# CPI inflation (%)
INF = np.array([-1.7,0.0,-2.3,-9.0,-9.9,-5.1,3.1,2.2,1.5,3.6,-2.1,-1.4,0.7,5.0,10.9,6.1,1.7,2.3,
8.3,14.4,8.1,-1.2,1.3,7.9,1.9,0.8,0.7,-0.4,1.5,3.3,2.8,0.7,1.7,1.0,1.0,1.3,1.3,1.6,2.9,
3.1,4.2,5.5,5.7,4.4,3.2,6.2,11.0,9.1,5.8,6.5,7.6,11.3,13.5,10.3,6.2,3.2,4.3,3.6,1.9,3.6,
4.1,4.8,5.4,4.2,3.0,3.0,2.6,2.8,3.0,2.3,1.6,2.2,3.4,2.8,1.6,2.3,2.7,3.4,3.2,2.8,3.8,-0.4,
1.6,3.2,2.1,1.5,1.6,0.1,1.3,2.1,2.4,1.8,1.2,4.7,8.0,4.1,2.9,2.6])

Y0, W0, T = 1928, 100.0, 30
assert len(NOM_S) == len(NOM_B) == len(INF) == 98

def real_series(ws):
    nom = ws*NOM_S + (1-ws)*NOM_B                 # annually rebalanced blend, nominal %
    return (1.0 + nom/100.0)/(1.0 + INF/100.0) - 1.0

def realized(ret, w):
    W = W0
    for r in ret:
        W = W*(1+r) - w*W0
        if W <= 0: return 0.0, True
    return W, False

def reorder(ret, w, K, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((K, T)), axis=1); u = 1.0 + ret[idx]
    W = np.full(K, W0); dead = np.zeros(K, bool)
    for t in range(T):
        W = W*u[:, t] - w*W0; dead |= (W <= 0.0)
    return np.maximum(W, 0.0), dead.mean()

if __name__ == "__main__":
    real = real_series(0.60)
    print(f"60/40 real return {Y0}-{Y0+len(real)-1}: mean {real.mean()*100:.1f}%, "
          f"sd {real.std(ddof=1)*100:.1f}%")
    s = 1966 - Y0; ret = real[s:s+T]
    Wr, dr = realized(ret, 0.04)
    Wc, pm = reorder(ret, 0.04, K=60000)
    print(f"1966-1995, 60/40, 4% rule: realized order -> final {Wr:.1f} "
          f"({'depleted' if dr else 'survived'}); reordered -> median {np.median(Wc):.0f}, "
          f"deplete {pm*100:.0f}% of orderings")

    # Table 3: for each allocation, count historical depletions and how many survive
    # under a majority of reorderings of the same returns
    print("\nTable 3: windows depleted, and of those, reversed under a majority of reorderings")
    print(f"{'alloc':>6} {'sd%':>5} | {'4% dep':>6} {'4% rev':>6} | {'5% dep':>6} {'5% rev':>6}")
    for ws, tag in [(1.00,"100/0"),(0.80,"80/20"),(0.60,"60/40"),(0.40,"40/60"),(0.20,"20/80")]:
        rr = real_series(ws); nwin = len(rr)-T+1
        cells = []
        for w in (0.04, 0.05):
            ndep = nrev = 0
            for st in range(nwin):
                win = rr[st:st+T]
                _, d = realized(win, w)
                if d:
                    _, p = reorder(win, w, K=40000, seed=5)
                    ndep += 1; nrev += (p < 0.5)
            cells += [ndep, nrev]
        print(f"{tag:>6} {rr.std(ddof=1)*100:5.1f} | {cells[0]:6d} {cells[1]:6d} | {cells[2]:6d} {cells[3]:6d}")
