"""
Minimal similarity evaluation: real.csv vs synthetic.csv

Usage:
    python evaluate.py real.csv synthetic.csv
    python evaluate.py real.csv synthetic.csv --id patient_id --time charttime
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats

BINS = 20
EPS = 1e-10
warnings.filterwarnings("ignore")  # e.g. autocorr of constant sequences


def histograms(r, s):
    """Shared bins -> two probability vectors (works for numeric and categorical)."""
    if pd.api.types.is_numeric_dtype(r):
        edges = np.histogram_bin_edges(pd.concat([r, s]), bins=BINS)
        p = np.histogram(r, edges)[0].astype(float)
        q = np.histogram(s, edges)[0].astype(float)
    else:
        cats = sorted(set(r) | set(s), key=str)
        p = r.value_counts().reindex(cats, fill_value=0).values.astype(float)
        q = s.value_counts().reindex(cats, fill_value=0).values.astype(float)
    return p / p.sum(), q / q.sum()


def column_metrics(r, s):
    r, s = r.dropna(), s.dropna()
    p, q = histograms(r, s)
    out = {
        "TVD": 0.5 * np.abs(p - q).sum(),
        "KL": stats.entropy(p + EPS, q + EPS),
        "HistOverlap": np.minimum(p, q).sum(),  # 1 = identical
    }
    if pd.api.types.is_numeric_dtype(r):
        out["KS"] = stats.ks_2samp(r, s).statistic
        out["Wasserstein"] = stats.wasserstein_distance(r, s) / (r.std() + EPS)  # in units of real std
    else:
        counts = np.vstack([p * len(r), q * len(s)])
        counts = counts[:, counts.sum(0) > 0]
        out["Chi2_p"] = stats.chi2_contingency(counts)[1]  # high p = no detectable difference
    return out


def discretize(df):
    return df.apply(lambda c: pd.qcut(c, BINS, labels=False, duplicates="drop")
                    if pd.api.types.is_numeric_dtype(c) else c.astype(str)).fillna(-1)


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -(p * np.log(p)).sum()


def mi_matrix(df):
    """Pairwise normalized mutual information (arithmetic normalisation), fast via bincount."""
    d = discretize(df).apply(lambda c: pd.factorize(c)[0])
    X, cols = d.values, d.columns
    k = X.max(axis=0) + 1
    H = np.array([_entropy(np.bincount(X[:, i])) for i in range(X.shape[1])])
    M = np.eye(len(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            hij = _entropy(np.bincount(X[:, i] * k[j] + X[:, j]))
            denom = (H[i] + H[j]) / 2
            M[i, j] = M[j, i] = (H[i] + H[j] - hij) / denom if denom > 0 else 0.0
    return pd.DataFrame(M, index=cols, columns=cols)


def cramers_v(a, b):
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table)[0]
    return np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))


def matrix_diff(a, b):
    """Mean absolute difference of the off-diagonal entries (NaNs ignored)."""
    mask = ~np.eye(len(a), dtype=bool)
    return np.nanmean(np.abs(a.values - b.values)[mask])


def fidelity_score(real, syn, cols, num, cat, mi_real, mi_syn, weights=None, seed=0):
    """
    One fidelity number in [0, 1] (1 = indistinguishable from real, 0 = nothing preserved).
      marginals    1 - mean KS (numeric) / 1 - TVD (categorical)
      correlation  Spearman error, relative to a "no dependencies" baseline
      dependency   normalized-MI error, relative to the same baseline
      missingness  1 - mean |missing rate real - missing rate synthetic|
    Baseline = real data with every column shuffled independently: perfect marginals, zero
    dependencies. Matching it scores 0 on correlation/dependency, matching real data scores 1.
    """
    w = weights or {"marginals": 0.4, "correlation": 0.25, "dependency": 0.15, "missingness": 0.2}
    rng = np.random.default_rng(seed)
    base = real[cols].apply(lambda c: pd.Series(rng.permutation(c.values), index=c.index))

    def relative(err, err_base):
        return float(np.clip(1 - err / err_base, 0, 1)) if err_base > 0 else 1.0

    marg = [1 - stats.ks_2samp(real[c].dropna(), syn[c].dropna()).statistic for c in num
            if real[c].notna().any() and syn[c].notna().any()]
    marg += [1 - 0.5 * np.abs(np.subtract(*histograms(real[c].dropna(), syn[c].dropna()))).sum() for c in cat]

    comp = {"marginals": float(np.mean(marg))}
    if len(num) > 1:
        cr = real[num].corr("spearman")
        comp["correlation"] = relative(matrix_diff(cr, syn[num].corr("spearman")),
                                       matrix_diff(cr, base[num].corr("spearman")))
    else:
        comp["correlation"] = 1.0
    comp["dependency"] = relative(matrix_diff(mi_real, mi_syn), matrix_diff(mi_real, mi_matrix(base)))
    comp["missingness"] = float(1 - np.mean(np.abs(real[cols].isna().mean() - syn[cols].isna().mean())))
    score = sum(w[k] * comp[k] for k in comp) / sum(w.values())
    return score, comp


def structural(real, syn, id_col, time_col, num_cols):
    """Time series: sequence lengths, step-to-step changes, lag-1 autocorrelation."""
    out = {}
    lens_r, lens_s = real.groupby(id_col).size(), syn.groupby(id_col).size()
    out["SeqLength_KS"] = stats.ks_2samp(lens_r, lens_s).statistic
    for c in num_cols:
        dr = real.sort_values(time_col).groupby(id_col)[c].diff().dropna()
        ds = syn.sort_values(time_col).groupby(id_col)[c].diff().dropna()
        ar = real.sort_values(time_col).groupby(id_col)[c].apply(lambda x: x.autocorr()).mean()
        as_ = syn.sort_values(time_col).groupby(id_col)[c].apply(lambda x: x.autocorr()).mean()
        out[f"{c}: delta_KS"] = stats.ks_2samp(dr, ds).statistic if len(dr) and len(ds) else np.nan
        out[f"{c}: autocorr real/syn"] = f"{ar:.2f} / {as_:.2f}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real")
    ap.add_argument("synthetic")
    ap.add_argument("--id", help="patient/sequence id column (enables structural metrics)")
    ap.add_argument("--time", help="time column for ordering sequences")
    args = ap.parse_args()

    real, syn = pd.read_csv(args.real), pd.read_csv(args.synthetic)
    skip = [c for c in (args.id, args.time) if c]
    cols = [c for c in real.columns if c in syn.columns and c not in skip]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
    cat = [c for c in cols if c not in num]

    # 1 + 3: statistical & histogram similarity per column
    per_col = pd.DataFrame({c: column_metrics(real[c], syn[c]) for c in cols}).T
    print("\n=== Per-column similarity ===")
    print(per_col.round(3).to_string())

    # 5: correlation preservation
    print("\n=== Correlation preservation (mean abs diff, 0 = perfect) ===")
    if len(num) > 1:
        for m in ("pearson", "spearman"):
            print(f"{m:10s}: {matrix_diff(real[num].corr(m), syn[num].corr(m)):.3f}")
    if len(cat) > 1:
        cv = lambda df: pd.DataFrame([[cramers_v(df[a], df[b]) for b in cat] for a in cat], index=cat, columns=cat)
        print(f"{'cramers_v':10s}: {matrix_diff(cv(real), cv(syn)):.3f}")

    # 4: mutual information
    print("\n=== Mutual information (mean abs diff of normalized MI, 0 = perfect) ===")
    mi_real, mi_syn = mi_matrix(real[cols]), mi_matrix(syn[cols])
    print(f"{matrix_diff(mi_real, mi_syn):.3f}")

    # 2: structural similarity
    if args.id and args.time:
        print("\n=== Structural similarity (time series) ===")
        for k, v in structural(real, syn, args.id, args.time, num).items():
            print(f"{k:30s}: {v if isinstance(v, str) else round(v, 3)}")

    # one summary number for the leaderboard
    score, comp = fidelity_score(real, syn, cols, num, cat, mi_real, mi_syn)
    print("\n=== Fidelity score (0 = useless, 1 = indistinguishable) ===")
    for k, v in comp.items():
        print(f"  {k:12s}: {v:.3f}")
    print(f"  {'FIDELITY':12s}: {score:.3f}")


if __name__ == "__main__":
    main()