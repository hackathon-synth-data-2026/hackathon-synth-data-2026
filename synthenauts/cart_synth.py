"""Usage: python cart_synth.py real_train.parquet out.parquet [min_leaf=20] [seed=0] [evaluation.json]

Sequential CART synthesis (synthpop-style) baseline for the UKA synthetic ICU challenge.

Each column is generated in turn from a tree fitted on the real data using the target and
all earlier columns; values are drawn from the real values in the matching leaf (with
small noise for numerics). Missingness is modelled explicitly per column.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

real_path, out_path = sys.argv[1], sys.argv[2]
min_leaf = int(sys.argv[3]) if len(sys.argv) > 3 else 20
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)

real = pd.read_parquet(real_path)
ident, target = "recordid", "mortality"
cols = [c for c in real.columns if c not in (ident, target)]
# optional 5th arg: path to an evaluation.json whose TRTR permutation importances set the column order
order = sys.argv[5] if len(sys.argv) > 5 else ""
if order:
    import json
    imp = json.load(open(order))["utility"]["trtr"]["permutation_importances"]
    ranked = [d["feature"] for d in sorted(imp, key=lambda d: -d["permutation_importance_mean"])]
    cols = [c for c in ranked if c in cols] + [c for c in cols if c not in ranked]
n = len(real)

syn = pd.DataFrame(index=range(n))
syn[target] = rng.choice(real[target].values, n, replace=True)
done = [target]


def leaf_sample(model, Xr, yr, Xs):
    leaves_r = model.apply(Xr)
    leaves_s = model.apply(Xs)
    pools = pd.Series(yr).groupby(leaves_r).apply(np.array).to_dict()
    return np.array([rng.choice(pools[l]) for l in leaves_s])


for c in cols:
    Xr, Xs = real[done].astype(float).values, syn[done].astype(float).values
    y = real[c]
    miss = y.isna().values
    out = np.full(n, np.nan)
    # 1) missingness
    if miss.any() and not miss.all():
        m = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=seed).fit(Xr, miss)
        smiss = leaf_sample(m, Xr, miss, Xs).astype(bool)
    else:
        smiss = np.full(n, miss.all())
    # 2) observed values
    obs = ~miss
    if obs.sum() > 0 and (~smiss).sum() > 0:
        yo = y.values[obs].astype(float)
        if len(np.unique(yo)) <= 2:
            m = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=seed).fit(Xr[obs], yo)
        else:
            m = DecisionTreeRegressor(min_samples_leaf=min_leaf, random_state=seed).fit(Xr[obs], yo)
        vals = leaf_sample(m, Xr[obs], yo, Xs[~smiss])
        out[~smiss] = vals
    syn[c] = out
    done.append(c)

# restore dtypes
for c in cols:
    if pd.api.types.is_bool_dtype(real[c]) or str(real[c].dtype) == "boolean":
        syn[c] = syn[c].map({1.0: True, 0.0: False}).astype("boolean")
    else:
        syn[c] = syn[c].astype(real[c].dtype) if real[c].notna().all() else syn[c]
syn[target] = syn[target].astype(real[target].dtype)
syn[ident] = np.arange(10_000_000, 10_000_000 + n)
syn = syn[real.columns]
syn.to_parquet(out_path, index=False)
print("wrote", out_path, syn.shape)
