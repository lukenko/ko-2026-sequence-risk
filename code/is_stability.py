"""
Checks that the Table 5 importance-sampling estimates are stable: same multisets with
K = 6000 vs 24000 orderings, and with the lowest effective-sample-size multisets dropped.
The weights only depend on standardized returns, so seeds 202 and 303 are run for one
portfolio only.
"""
import numpy as np


def run(mu, sig, f, phi, K, W0=100.0, T=30, NM=2500, seed=101, chunk=6000):
    rng = np.random.default_rng(seed)
    se = sig * np.sqrt(1 - phi**2)
    paths = np.empty((NM, T)); paths[:, 0] = rng.normal(mu, sig, NM)
    for t in range(1, T):
        paths[:, t] = mu + phi * (paths[:, t-1] - mu) + rng.normal(0, se, NM)
    M = np.sort(paths, axis=1)
    pM = np.empty(NM); ess = np.empty(NM)
    for i in range(NM):
        m = M[i]; lws = []; Rs = []
        for start in range(0, K, chunk):            # in chunks to limit memory
            k = min(chunk, K - start)
            idx = np.argsort(rng.random((k, T)), axis=1); x = m[idx]
            W = np.full(k, W0); dead = np.zeros(k, bool)
            for t in range(T):
                W = W * (1 + x[:, t]) + f; dead |= (W <= 0); W = np.maximum(W, 0)
            innov = x[:, 1:] - mu - phi * (x[:, :-1] - mu)
            lws.append(-0.5 * (x[:, 0] - mu)**2 / sig**2 - 0.5 * np.sum(innov**2, axis=1) / se**2)
            Rs.append(dead.astype(float))
        lw = np.concatenate(lws); R = np.concatenate(Rs)
        lw -= lw.max(); w = np.exp(lw)
        pM[i] = np.dot(w, R) / w.sum(); ess[i] = w.sum()**2 / np.dot(w, w) / K
    return pM, ess


def share(pM, keep=None):
    p = pM if keep is None else pM[keep]
    seq = np.mean(p * (1 - p)); lev = np.var(p)
    return 100 * seq / (seq + lev)


def report(label, mu, sig, phi, seed, f=-4.0):
    p6, e6 = run(mu, sig, f, phi, 6000, seed=seed)
    p24, _ = run(mu, sig, f, phi, 24000, seed=seed)
    k1 = e6 >= 0.01; k5 = e6 >= 0.05
    print(f"{label:>12} {seed:>5} {phi:+5.1f} | {share(p6):7.1f} {share(p24):8.1f} | "
          f"{e6.min():7.4f} | {(~k1).sum():4d} {share(p6, k1):7.1f} | "
          f"{(~k5).sum():4d} {share(p6, k5):7.1f}", flush=True)


if __name__ == "__main__":
    print("Sequencing share of depletion (%), withdrawal 4, T = 30, 2500 multisets")
    print(f"{'portfolio':>12} {'seed':>5} {'phi':>5} | {'K=6000':>7} {'K=24000':>8} | "
          f"{'min ESS':>7} | {'n<1%':>4} {'drop':>7} | {'n<5%':>4} {'drop':>7}")
    portfolios = [("conservative", 0.04, 0.08), ("balanced", 0.05, 0.12),
                  ("growth", 0.06, 0.15), ("aggressive", 0.07, 0.18)]
    for name, mu, sig in portfolios:
        for phi in (-0.2, 0.2):
            report(name, mu, sig, phi, 101)
    for seed in (202, 303):
        for phi in (-0.2, 0.2):
            report("balanced", 0.05, 0.12, phi, seed)
