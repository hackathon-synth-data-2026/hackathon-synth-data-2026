"""Statistic matching: swap blocks of values between synthetic rows so that the
rank-correlation and mutual-information matrices move towards real_train's.

A block is a group of columns that must stay together for clinical consistency:
the five summaries of a vital sign (all invasive or all non-invasive blood pressures
as one block, so diastolic <= systolic is kept), the first/last pair of a lab, or a
single column. Swapping a block between two rows keeps every column's values and
missing cells, so marginals, missingness rates and the repair rules are untouched.

Objective (what the evaluator's fidelity score rewards):
    corr_error / corr_baseline + mi_error / mi_baseline
where corr_error is the mean |Spearman_real - rank_corr_syn| over feature pairs, and
mi_error the same for normalised mutual information on the evaluator's own 20-quantile
bins (missing = its own bin). Rank correlation uses each column's global ranks on the
pairwise-complete rows, a close proxy for pandas' Spearman. MI is tracked exactly.

Moves are greedy: a random block, two random rows, kept if the objective drops.
"""
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata

SUFFIXES = ("first", "last", "lowest", "highest", "median")
INVASIVE = {"DiasABP", "SysABP", "MAP"}
NONINVASIVE = {"NIDiasABP", "NISysABP", "NIMAP"}


def make_units(cols):
    groups = {}
    for c in cols:
        head, _, tail = c.rpartition("_")
        base = head if tail in SUFFIXES else c
        base = "invasiveBP" if base in INVASIVE else "noninvasiveBP" if base in NONINVASIVE else base
        groups.setdefault(base, []).append(c)
    return list(groups.values())


def _codes(df):
    """Bin codes exactly as the evaluator's mi_matrix computes them."""
    from hackathon_synth_data_2026.fidelity import discretize
    return discretize(df).apply(lambda c: pd.factorize(c)[0]).to_numpy().astype(np.int64)


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -(p * np.log(p)).sum()


@njit(cache=True)
def _f(c):
    return c * np.log(c) if c > 0 else 0.0


@njit(cache=True)
def _corr(n, si, sj, sii, sjj, sij):
    v1 = n * sii - si * si
    v2 = n * sjj - sj * sj
    if n >= 3 and v1 > 1e-9 and v2 > 1e-9:
        return (n * sij - si * sj) / np.sqrt(v1 * v2)
    return 0.0


@njit(cache=True)
def _run(Rk, Ob, B, uflat, uptr, unit_of, N, SX, SXX, SXY, C, S, H, Tc, Tm, validc, Ec, Em, idx,
         n_iter, wc, wm, seed, reseed):
    # Seeded once; later chunks continue the same random stream, so running the swaps in chunks
    # (to report progress) gives exactly the same result as one long run.
    if reseed:
        np.random.seed(seed)
    n, p = Rk.shape
    nu = len(uptr) - 1
    logn = np.log(n)
    maxu = 0
    for u in range(nu):
        maxu = max(maxu, uptr[u + 1] - uptr[u])
    st = np.empty((maxu, p, 6))
    nEc = np.empty((maxu, p))
    nEm = np.empty((maxu, p))
    nS = np.empty((maxu, p))
    chg = np.zeros((maxu, p), dtype=np.int8)
    accepted = 0
    total = 0.0
    for it in range(n_iter):
        u = np.random.randint(nu)
        a = np.random.randint(n)
        b = np.random.randint(n)
        if a == b:
            continue
        s0, s1 = uptr[u], uptr[u + 1]
        same = True
        for t in range(s0, s1):
            i = uflat[t]
            if Ob[a, i] != Ob[b, i] or (Ob[a, i] and Rk[a, i] != Rk[b, i]):
                same = False
                break
        if same:
            continue
        dc = 0.0
        dm = 0.0
        for t in range(s0, s1):
            i = uflat[t]
            ti = t - s0
            oai, obi = Ob[a, i], Ob[b, i]
            xai, xbi = Rk[a, i], Rk[b, i]
            bai, bbi = B[a, i], B[b, i]
            for j in range(p):
                chg[ti, j] = 0
                if unit_of[j] == u:
                    continue
                oaj, obj = Ob[a, j], Ob[b, j]
                xaj, xbj = Rk[a, j], Rk[b, j]
                dn = 0.0; dsi = 0.0; dsj = 0.0; dsii = 0.0; dsjj = 0.0; dsij = 0.0
                if oai and oaj:
                    dn -= 1; dsi -= xai; dsj -= xaj; dsii -= xai * xai; dsjj -= xaj * xaj; dsij -= xai * xaj
                if obi and obj:
                    dn -= 1; dsi -= xbi; dsj -= xbj; dsii -= xbi * xbi; dsjj -= xbj * xbj; dsij -= xbi * xbj
                if obi and oaj:
                    dn += 1; dsi += xbi; dsj += xaj; dsii += xbi * xbi; dsjj += xaj * xaj; dsij += xbi * xaj
                if oai and obj:
                    dn += 1; dsi += xai; dsj += xbj; dsii += xai * xai; dsjj += xbj * xbj; dsij += xai * xbj
                n2 = N[i, j] + dn; si2 = SX[i, j] + dsi; sj2 = SX[j, i] + dsj
                sii2 = SXX[i, j] + dsii; sjj2 = SXX[j, i] + dsjj; sij2 = SXY[i, j] + dsij
                st[ti, j, 0] = n2; st[ti, j, 1] = si2; st[ti, j, 2] = sj2
                st[ti, j, 3] = sii2; st[ti, j, 4] = sjj2; st[ti, j, 5] = sij2
                chg[ti, j] = 1
                if validc[i, j]:
                    e = abs(_corr(n2, si2, sj2, sii2, sjj2, sij2) - Tc[i, j])
                    nEc[ti, j] = e
                    dc += e - Ec[i, j]
                else:
                    nEc[ti, j] = 0.0
                baj, bbj = B[a, j], B[b, j]
                if bai != bbi and baj != bbj:
                    c1 = C[i, j, bai, baj]; c2 = C[i, j, bbi, bbj]
                    c3 = C[i, j, bbi, baj]; c4 = C[i, j, bai, bbj]
                    dS = (_f(c1 - 1) - _f(c1) + _f(c2 - 1) - _f(c2)
                          + _f(c3 + 1) - _f(c3) + _f(c4 + 1) - _f(c4))
                    Sn = S[i, j] + dS
                    den = (H[i] + H[j]) / 2
                    M = (H[i] + H[j] - (logn - Sn / n)) / den if den > 0 else 0.0
                    e = abs(M - Tm[i, j])
                    nEm[ti, j] = e
                    nS[ti, j] = Sn
                    dm += e - Em[i, j]
                    chg[ti, j] = 2
        delta = wc * dc + wm * dm
        if delta < 0:
            accepted += 1
            total += delta
            for t in range(s0, s1):
                i = uflat[t]
                ti = t - s0
                bai, bbi = B[a, i], B[b, i]
                for j in range(p):
                    if chg[ti, j] == 0:
                        continue
                    N[i, j] = st[ti, j, 0]; N[j, i] = st[ti, j, 0]
                    SX[i, j] = st[ti, j, 1]; SX[j, i] = st[ti, j, 2]
                    SXX[i, j] = st[ti, j, 3]; SXX[j, i] = st[ti, j, 4]
                    SXY[i, j] = st[ti, j, 5]; SXY[j, i] = st[ti, j, 5]
                    Ec[i, j] = nEc[ti, j]; Ec[j, i] = nEc[ti, j]
                    if chg[ti, j] == 2:
                        baj, bbj = B[a, j], B[b, j]
                        C[i, j, bai, baj] -= 1; C[i, j, bbi, bbj] -= 1
                        C[i, j, bbi, baj] += 1; C[i, j, bai, bbj] += 1
                        C[j, i, baj, bai] -= 1; C[j, i, bbj, bbi] -= 1
                        C[j, i, baj, bbi] += 1; C[j, i, bbj, bai] += 1
                        S[i, j] = nS[ti, j]; S[j, i] = nS[ti, j]
                        Em[i, j] = nEm[ti, j]; Em[j, i] = nEm[ti, j]
            for t in range(s0, s1):
                i = uflat[t]
                Rk[a, i], Rk[b, i] = Rk[b, i], Rk[a, i]
                Ob[a, i], Ob[b, i] = Ob[b, i], Ob[a, i]
                B[a, i], B[b, i] = B[b, i], B[a, i]
            idx[u, a], idx[u, b] = idx[u, b], idx[u, a]
    return accepted, total


def match_statistics(syn, real, cols, n_iter=2_000_000, seed=0, w_corr=1.0, w_mi=1.0, verbose=False,
                     units=None, corr_cols=None, progress=None):
    """Return syn with blocks of `cols` swapped between rows to match real's correlations and MI.

    units: list of column groups swapped together (default: the ICU suffix groups).
    corr_cols: columns whose rank correlations count (default: all); leave out unordered categories.
    progress: optional callback taking the finished share (0-1); the result does not depend on it.
    """
    from hackathon_synth_data_2026.fidelity import matrix_diff, mi_matrix

    units = units or make_units(cols)
    cols = [c for u in units for c in u]
    p, n = len(cols), len(syn)
    pos = {c: k for k, c in enumerate(cols)}
    uflat = np.array([pos[c] for u in units for c in u], dtype=np.int64)
    uptr = np.cumsum([0] + [len(u) for u in units]).astype(np.int64)
    unit_of = np.empty(p, dtype=np.int64)
    for k, u in enumerate(units):
        unit_of[[pos[c] for c in u]] = k

    X0 = syn[cols].to_numpy(dtype=float)
    Ob = ~np.isnan(X0)
    Rk = np.zeros_like(X0)
    for j in range(p):
        Rk[Ob[:, j], j] = rankdata(X0[Ob[:, j], j])
    B = _codes(syn[cols])
    K = int(B.max()) + 1

    O = Ob.astype(float)
    N, SX, SXX, SXY = O.T @ O, Rk.T @ O, (Rk ** 2).T @ O, Rk.T @ Rk
    C = np.zeros((p, p, K, K), dtype=np.int64)
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                C[i, j] = np.bincount(B[:, i] * K + B[:, j], minlength=K * K).reshape(K, K)
                S[i, j] = sum(c * np.log(c) for c in C[i, j].ravel() if c > 0)
    H = np.array([_entropy(np.bincount(B[:, j])) for j in range(p)])

    real_f = real[cols]
    Tc = real_f.corr("spearman").to_numpy()
    Tm_df = mi_matrix(real_f)
    Tm = Tm_df.to_numpy()
    validc = ~np.isnan(Tc) & ~np.eye(p, dtype=bool)
    if corr_cols is not None:
        keep = np.isin(cols, list(corr_cols))
        validc &= keep[:, None] & keep[None, :]
    Tc = np.nan_to_num(Tc)

    def current_errors():
        Ec = np.zeros((p, p))
        num = N * SXY - SX * SX.T
        v = N * SXX - SX ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / np.sqrt(v * v.T)
        r = np.where((N >= 3) & (v > 1e-9) & (v.T > 1e-9), r, 0.0)
        Ec[validc] = np.abs(r - Tc)[validc]
        den = (H[:, None] + H[None, :]) / 2
        Hij = np.log(n) - S / n
        M = np.where(den > 0, (H[:, None] + H[None, :] - Hij) / np.where(den > 0, den, 1), 0.0)
        Em = np.abs(M - Tm)
        np.fill_diagonal(Em, 0)
        return Ec, Em

    Ec, Em = current_errors()
    # Baselines as in the evaluator: real with every column shuffled independently.
    rng = np.random.default_rng(0)
    base = real_f.apply(lambda c: pd.Series(rng.permutation(c.values), index=c.index))
    cc = cols if corr_cols is None else [c for c in cols if c in set(corr_cols)]
    bc = matrix_diff(real_f[cc].corr("spearman"), base[cc].corr("spearman")) if len(cc) > 1 else 0.0
    bm = matrix_diff(Tm_df, mi_matrix(base))
    P, Pc = p * (p - 1), max(len(cc) * (len(cc) - 1), 1)
    wc = w_corr / (Pc * bc) if bc > 0 else 0.0
    wm = w_mi / (P * bm) if bm > 0 else 0.0
    obj0 = wc * Ec.sum() + wm * Em.sum()

    idx = np.tile(np.arange(n), (len(units), 1))
    chunks = 20 if progress else 1
    acc, gain = 0, 0.0
    for k in range(chunks):
        m = n_iter // chunks + (1 if k < n_iter % chunks else 0)
        a, g = _run(Rk, Ob, B, uflat, uptr, unit_of, N, SX, SXX, SXY, C, S, H, Tc, Tm, validc,
                    Ec, Em, idx, m, wc, wm, seed, k == 0)
        acc, gain = acc + a, gain + g
        if progress:
            progress((k + 1) / chunks)
    if verbose:
        print(f"statmatch: {acc} of {n_iter} swaps kept; objective {obj0:.4f} -> {obj0 + gain:.4f}")

    out = syn.copy()
    for k, u in enumerate(units):
        jj = [pos[c] for c in u]
        out[u] = X0[idx[k]][:, jj]
    return out
