"""Synthetic ICU table generator, v3 of our CART + repair pipeline.

Steps, in order (each one was kept only if it raised the score; see synthenauts/README.md):

  1. Sequential CART synthesis (synthpop style) with explicit missingness, columns in
     mortality-importance order, leaf size 10. Same idea as cart_synth.py, vectorised.
  2. Demographic jitter: small noise on Age/Height/Weight so exact real combinations of
     these quasi-identifiers are not reused (privacy).
  3. Correlation restoration: CART shrinks rank correlations by ~13%. Values are reordered
     within each column (Iman-Conover, minimal-change map) so the Spearman matrix moves
     back towards real_train's. Each column keeps its values and its missing cells.
  4. Marginal matching: each column is mapped rank-preserving onto real_train's values.
  5. Clinical consistency repair (same rules as repair.py): 0% impossible patients.
  6. Statistic matching (statmatch.py): 2M greedy block swaps between rows to cut the
     remaining Spearman and mutual-information error. Blocks keep the repair rules intact.
  7. Learned-rule repair (v4): every rule that holds in all real_train rows and is unlikely by
     chance is enforced, using anysynth.py's rule learner: 95 order rules (MAP <= SysABP, ...) and
     43 category rules (a patient is in at most one of CCU, CSRU and SICU; no ventilation in
     the last 8 hours caps MechVentDuration; ...).
  8. Mortality relabelling: labels are redrawn from a risk model fitted on real_train only,
     at temperature 0.3 (less label noise than real data), with the real death rate.

Only real_train is used. real_test is never read.

Usage (from the repo root):
    uv run python synthenauts/synth.py                       # best settings -> results/submission.parquet
    uv run python synthenauts/synth.py --seed 3 --out results/sub_s3.parquet
    uv run python synthenauts/synth.py --help                # switch individual steps on/off
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
ID, TARGET = "recordid", "mortality"
VITALS = ["DiasABP", "GCS", "Glucose", "HR", "MAP", "NIDiasABP", "NIMAP",
          "NISysABP", "RespRate", "SaO2", "SysABP", "Temp"]
SUFFIX = ["first", "last", "lowest", "highest", "median"]
PRESSURE_PAIRS = [("DiasABP", "SysABP"), ("NIDiasABP", "NISysABP")]
DEMO_NUMERIC = ["Age", "Height", "Weight"]  # with Gender, the evaluator's 'demographics' privacy group

# XGBoost settings for the label model, chosen by 5-fold CV average precision on real_train.
LABEL_MODEL = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                   colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0,
                   eval_metric="logloss", random_state=0, n_jobs=1, tree_method="hist")  # 1 thread: bit-reproducible


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_float(df):
    return pd.DataFrame({c: df[c].to_numpy(dtype=float, na_value=np.nan) for c in df.columns}, index=df.index)


def leaf_sample(model, Xr, yr, Xs, rng):
    """For each synthetic row, draw one real value from the tree leaf it falls into."""
    lr, ls = model.apply(Xr), model.apply(Xs)
    order = np.argsort(lr, kind="stable")
    leaves, start, count = np.unique(lr[order], return_index=True, return_counts=True)
    pos = np.searchsorted(leaves, ls)
    idx = start[pos] + (rng.random(len(ls)) * count[pos]).astype(int)
    return yr[order][idx]


def importance_order(real, path=None):
    """Columns sorted by mortality importance (from train_importance.py output), rest appended."""
    cols = [c for c in real.columns if c not in (ID, TARGET)]
    path = Path(path) if path else ROOT / "synthenauts/train_imp.json"
    if not path.exists():
        return cols
    imp = json.loads(path.read_text())["utility"]["trtr"]["permutation_importances"]
    ranked = [d["feature"] for d in sorted(imp, key=lambda d: -d["permutation_importance_mean"])]
    return [c for c in ranked if c in cols] + [c for c in cols if c not in ranked]


# ---------------------------------------------------------------------------
# 1. sequential CART
# ---------------------------------------------------------------------------

def cart_synthesize(real, n, order, rng, min_leaf=10, leaf_top=None, top_k=0, label_first=True, seed=0,
                    cat_cols=(), progress=None):
    """Generate n rows column by column. real must be all-float (see to_float).

    cat_cols: integer-coded categorical columns, always fitted with a classification tree.
    progress: optional callback taking the finished share (0-1).
    """
    syn = pd.DataFrame(index=range(n))
    done = []
    if label_first:
        k = int(round(n * real[TARGET].mean()))
        syn[TARGET] = rng.permutation(np.r_[np.ones(k), np.zeros(n - k)])
        done = [TARGET]
    for i, c in enumerate(order):
        leaf = leaf_top if (leaf_top and i < top_k) else min_leaf
        y = real[c].to_numpy()
        miss = np.isnan(y)
        if not done:  # first column: sample its marginal
            syn[c] = rng.choice(y, n)
            done.append(c)
            continue
        Xr, Xs = real[done].to_numpy(), syn[done].to_numpy()
        out = np.full(n, np.nan)
        if miss.any() and not miss.all():
            m = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=seed).fit(Xr, miss)
            smiss = leaf_sample(m, Xr, miss, Xs, rng).astype(bool)
        else:
            smiss = np.full(n, miss.all())
        obs = ~miss
        if obs.any() and (~smiss).any():
            yo = y[obs]
            Tree = DecisionTreeClassifier if (c in cat_cols or len(np.unique(yo)) <= 2) else DecisionTreeRegressor
            m = Tree(min_samples_leaf=leaf, random_state=seed).fit(Xr[obs], yo)
            out[~smiss] = leaf_sample(m, Xr[obs], yo, Xs[~smiss], rng)
        syn[c] = out
        done.append(c)
        if progress:
            progress((i + 1) / len(order))
    return syn


# ---------------------------------------------------------------------------
# 2. mortality relabelling
# ---------------------------------------------------------------------------

def fit_label_model(real_f, params=None):
    from xgboost import XGBClassifier
    X = real_f.drop(columns=[c for c in (ID, TARGET) if c in real_f])
    return XGBClassifier(**(params or LABEL_MODEL)).fit(X, real_f[TARGET].astype(int)), list(X.columns)


def relabel(syn, model, cols, prevalence, rng, temperature=1.0):
    """Exactly round(n*prevalence) positives: top-k of logit(p) + temperature * logistic noise.

    temperature 1 behaves like Bernoulli(p) draws, 0 is a deterministic top-k by risk.
    """
    p = np.clip(model.predict_proba(syn[cols])[:, 1], 1e-6, 1 - 1e-6)
    u = rng.random(len(p))
    z = np.log(p / (1 - p)) + temperature * np.log(u / (1 - u))
    k = int(round(len(p) * prevalence))
    y = np.zeros(len(p))
    y[np.argsort(-z)[:k]] = 1
    return y


# ---------------------------------------------------------------------------
# 3. clinical repair (same rules as repair.py)
# ---------------------------------------------------------------------------

def repair(d):
    d = d.copy()
    for s in SUFFIX:
        d[f"GCS_{s}"] = d[f"GCS_{s}"].clip(3, 15)

    def blocks():
        for v in VITALS:
            block = d[[f"{v}_{s}" for s in SUFFIX]]
            lo, hi = block.min(axis=1), block.max(axis=1)
            d[f"{v}_lowest"] = lo.where(d[f"{v}_lowest"].notna())
            d[f"{v}_highest"] = hi.where(d[f"{v}_highest"].notna())
            for s in ["first", "last", "median"]:
                c = f"{v}_{s}"
                d[c] = d[c].clip(lower=d[f"{v}_lowest"], upper=d[f"{v}_highest"]).where(d[c].notna())

    def pressures():
        for dia, sys_ in PRESSURE_PAIRS:
            for s in SUFFIX:
                a, b = f"{dia}_{s}", f"{sys_}_{s}"
                swap = (d[a] > d[b]).fillna(False)
                d.loc[swap, [a, b]] = d.loc[swap, [b, a]].values

    for _ in range(3):
        blocks()
        pressures()
    blocks()
    return d


# ---------------------------------------------------------------------------
# 4. marginal matching
# ---------------------------------------------------------------------------

def match_marginals(syn, real, cols, rng):
    """Rank-preserving map of each column's observed synthetic values onto real quantiles."""
    syn = syn.copy()
    for c in cols:
        r = np.sort(real[c].dropna().to_numpy())
        s = syn[c].to_numpy()
        obs = np.flatnonzero(~np.isnan(s))
        if len(obs) == 0 or len(r) == 0:
            continue
        jitter = rng.random(len(obs)) * 1e-9  # random tie-breaking
        ranks = np.argsort(np.argsort(s[obs] + jitter * (np.abs(s[obs]) + 1)))
        q = (ranks + 0.5) / len(obs)
        new = s.copy()
        new[obs] = r[np.minimum((q * len(r)).astype(int), len(r) - 1)]
        syn[c] = new
    return syn


# ---------------------------------------------------------------------------
# 5. correlation restoration (Iman-Conover with a minimal-change map)
# ---------------------------------------------------------------------------

def _psd(m, eps=1e-6):
    m = np.nan_to_num((m + m.T) / 2)
    np.fill_diagonal(m, 1.0)
    w, v = np.linalg.eigh(m)
    m = (v * np.maximum(w, eps)) @ v.T
    d = np.sqrt(np.diag(m))
    return m / d[:, None] / d[None, :]


def _sqrtm(m):
    w, v = np.linalg.eigh((m + m.T) / 2)
    return (v * np.sqrt(np.maximum(w, 0))) @ v.T


def restore_correlations(syn, real, cols, rng, strength=1.0, iters=2, fixed=(), progress=None):
    """Reorder observed values within each column so the Spearman matrix moves towards real's.

    CART synthesis shrinks correlations (slope ~0.87 on this data). Each column keeps exactly
    its own values and its missing cells, so marginals and missingness are unchanged. The
    reordering uses the optimal-transport (Monge) map between the two Gaussian copulas,
    which is the linear map that moves rows the least.
    """
    from scipy.stats import norm
    syn = syn.copy()
    to_pearson = lambda s: 2 * np.sin(np.pi * s / 6)
    target = to_pearson(real[cols].corr("spearman").to_numpy())
    for it in range(iters):
        X = syn[cols].to_numpy()
        mask = ~np.isnan(X)
        n = len(X)
        Z = np.zeros_like(X)
        for j in range(len(cols)):
            o = mask[:, j]
            r = np.argsort(np.argsort(X[o, j] + rng.random(o.sum()) * 1e-9 * (np.abs(X[o, j]) + 1)))
            Z[o, j] = norm.ppf((r + 0.5) / o.sum())
        # Work on the covariance of the zero-filled scores; a pairwise correlation gap d_ij
        # shows up there scaled by the share of rows where both columns are observed.
        C = Z.T @ Z / n
        F = mask.T.astype(float) @ mask / n
        S = to_pearson(syn[cols].corr("spearman").to_numpy())
        T = C + strength * F * np.nan_to_num(target - S)
        lam = 0.05
        dC = np.diag(np.diag(C))
        C, T = (1 - lam) * C + lam * dC, (1 - lam) * T + lam * dC
        w, v = np.linalg.eigh((T + T.T) / 2)
        T = (v * np.maximum(w, 1e-3 * w.max())) @ v.T
        Ch = _sqrtm(C)
        Chi = np.linalg.inv(Ch)
        A = Chi @ _sqrtm(Ch @ T @ Ch) @ Chi
        Zn = Z @ A
        for j, c in enumerate(cols):
            if c in fixed:  # keep the strongest mortality predictors exactly as CART made them
                continue
            o = mask[:, j]
            vals = np.sort(X[o, j])
            new = X[:, j].copy()
            new[np.flatnonzero(o)[np.argsort(Zn[o, j])]] = vals
            syn[c] = new
        if progress:
            progress((it + 1) / iters)
    return syn


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def finalize(syn, real_raw):
    """Restore the 121-column schema and dtypes, add fresh identifiers."""
    out = pd.DataFrame(index=syn.index)
    for c in real_raw.columns:
        if c == ID:
            continue
        dt = real_raw[c].dtype
        if str(dt) == "boolean" or pd.api.types.is_bool_dtype(dt):
            out[c] = syn[c].map({1.0: True, 0.0: False}).astype("boolean")
        elif c == TARGET:
            out[c] = syn[c].astype("int64")
        else:
            out[c] = syn[c].astype(float)
    out[ID] = np.arange(10_000_000, 10_000_000 + len(out))
    return out[list(real_raw.columns)]


def generate(real_raw, seed=0, n=None, min_leaf=10, leaf_top=None, top_k=0, label_first=True,
             relabel_temp=None, label_params=None, do_repair=True, match=False, order=None,
             importance_path=None, label_model=None, corr_strength=0.0, corr_iters=2, corr_fixed_top=0, demo_noise=0.0, stat_iters=0, stat_verbose=False,
             learned_rules=False):
    rng = np.random.default_rng(seed)
    real = to_float(real_raw.drop(columns=[ID]))
    n = n or len(real)
    order = order or importance_order(real_raw, importance_path)
    feats = [c for c in real.columns if c != TARGET]
    linked, eq_links = set(), []
    if learned_rules:  # rules that hold in every real row (see step 7)
        import anysynth as AS
        schema = AS.detect_schema(real_raw)
        rules_order, rules_equal = AS.find_rules(real, [c for c in feats if schema[c]["kind"] == "numeric"])
        crules = AS.find_category_rules(real, schema, feats)
        # Columns tied by an "if A then B" rule (CCU, CSRU, SICU) are drawn consistently by CART.
        # The later steps keep them together instead of moving each one on its own.
        eq_links = [(r[0], r[3]) for r in crules if r[2] == "eq"]
        linked = {c for pair in eq_links for c in pair}
    syn = cart_synthesize(real, n, order, rng, min_leaf, leaf_top, top_k, label_first, seed)
    if demo_noise > 0:  # jitter demographics so exact real (Age, Height, Weight) combinations are not reused
        for c in DEMO_NUMERIC:
            q75, q25 = np.nanpercentile(real[c], [75, 25])
            syn[c] = syn[c] + rng.normal(0, demo_noise * (q75 - q25), n)
    if corr_strength > 0:
        syn = restore_correlations(syn, real, feats, rng, corr_strength, corr_iters,
                                   set(order[:corr_fixed_top]) | linked)
    if match:  # before repair, so the repaired table stays clinically consistent
        syn = match_marginals(syn, real, [c for c in feats if c not in linked], rng)
    if do_repair:
        syn = repair(syn)
    if stat_iters > 0:  # block swaps between rows: keeps marginals and the repair rules
        from statmatch import make_units, match_statistics
        units = make_units(feats)
        for a, b in eq_links:  # tied columns are swapped as one block
            ua, ub = (next(u for u in units if c in u) for c in (a, b))
            if ua is not ub:
                units.remove(ub)
                ua.extend(ub)
        syn = match_statistics(syn, real, feats, stat_iters, seed, verbose=stat_verbose, units=units)
    if learned_rules:  # after every step that moves values, so nothing breaks the rules again
        for _ in range(5):  # fixing one rule type can touch the other
            syn = AS.repair_category_rules(syn, crules, rng)
            syn = AS.repair_rules(syn, rules_order, rules_equal)
            if not AS.category_violations(syn, crules) and not AS.rule_violations(syn, rules_order):
                break
    if not label_first or relabel_temp is not None:  # last, so labels follow the final features
        model, cols = label_model or fit_label_model(real, label_params)
        syn[TARGET] = relabel(syn, model, cols, real[TARGET].mean(), rng,
                              1.0 if relabel_temp is None else relabel_temp)
    return finalize(syn, real_raw)


BEST = dict(min_leaf=10, demo_noise=0.25, corr_strength=1.0, corr_iters=10, match=True,
            do_repair=True, relabel_temp=0.3, stat_iters=2_000_000, learned_rules=True)
# v3 (our v3 pick was seed 10) is BEST with learned_rules=False.


def ensure_importance(train_path, path=ROOT / "synthenauts/train_imp.json"):
    """Column order comes from train_importance.py; compute it once if it is missing."""
    if not Path(path).exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, str(ROOT / "synthenauts/train_importance.py"), str(train_path), str(path)],
                       check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "data/real_train.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results/submission.parquet"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-leaf", type=int, default=BEST["min_leaf"])
    ap.add_argument("--demo-noise", type=float, default=BEST["demo_noise"],
                    help="jitter on Age/Height/Weight in IQR units (0 = off)")
    ap.add_argument("--corr-iters", type=int, default=BEST["corr_iters"],
                    help="correlation-restoration passes (0 = off)")
    ap.add_argument("--relabel-temp", type=float, default=BEST["relabel_temp"],
                    help="mortality label temperature; negative keeps the CART labels")
    ap.add_argument("--stat-iters", type=int, default=BEST["stat_iters"],
                    help="statistic-matching swap attempts (0 = off)")
    ap.add_argument("--no-match", action="store_true", help="skip marginal matching")
    ap.add_argument("--no-repair", action="store_true", help="skip the clinical repair")
    ap.add_argument("--no-learned-rules", action="store_true",
                    help="skip the learned-rule repair (gives the v3 submission's pipeline)")
    args = ap.parse_args()
    ensure_importance(args.train)
    real = pd.read_parquet(args.train)
    syn = generate(real, seed=args.seed, min_leaf=args.min_leaf, demo_noise=args.demo_noise,
                   corr_strength=1.0 if args.corr_iters > 0 else 0.0, corr_iters=args.corr_iters,
                   match=not args.no_match, do_repair=not args.no_repair,
                   relabel_temp=None if args.relabel_temp < 0 else args.relabel_temp,
                   stat_iters=args.stat_iters, learned_rules=not args.no_learned_rules)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    syn.to_parquet(args.out, index=False)
    print("wrote", args.out, syn.shape)


if __name__ == "__main__":
    main()
