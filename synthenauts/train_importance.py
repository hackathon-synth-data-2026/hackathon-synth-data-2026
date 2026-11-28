"""Permutation importances from real_train only (5-fold CV), saved in the evaluation.json layout cart_synth.py reads.
Usage: python train_importance.py real_train.parquet out.json"""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
t = pd.read_parquet(sys.argv[1])
X = t.drop(columns=["recordid", "mortality"]).astype(float); y = t["mortality"].values
imp = np.zeros(X.shape[1])
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, tree_method="hist").fit(X.iloc[tr], y[tr])
    imp += permutation_importance(m, X.iloc[va], y[va], scoring="average_precision", n_repeats=3, random_state=0).importances_mean
out = [{"feature": c, "permutation_importance_mean": float(v)} for c, v in zip(X.columns, imp / 5)]
json.dump({"utility": {"trtr": {"permutation_importances": out}}}, open(sys.argv[2], "w"))
print(sorted(out, key=lambda d: -d["permutation_importance_mean"])[:8])
