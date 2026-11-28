"""Pick the mortality risk model used for relabelling, by 5-fold CV average precision on real_train only.
Usage: uv run python synthenauts/tune_label_model.py"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

t = pd.read_parquet("data/real_train.parquet")
X = t.drop(columns=["recordid", "mortality"]).astype(float); y = t["mortality"].values
base = dict(eval_metric="logloss", random_state=0, n_jobs=8, tree_method="hist")
cands = {
    "official(100,d6,lr.1)": dict(n_estimators=100, max_depth=6, learning_rate=0.1),
    "d3 lr.03 n400 sub.8 col.5": dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0),
    "d2 lr.03 n600 sub.8 col.5": dict(n_estimators=600, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0),
    "d4 lr.02 n600 sub.7 col.4": dict(n_estimators=600, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.4, min_child_weight=10, reg_lambda=10.0),
    "d3 lr.02 n800 sub.7 col.3": dict(n_estimators=800, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.3, min_child_weight=5, reg_lambda=5.0),
    "d5 lr.01 n1000 sub.7 col.3": dict(n_estimators=1000, max_depth=5, learning_rate=0.01, subsample=0.7, colsample_bytree=0.3, min_child_weight=10, reg_lambda=10.0),
}
for name, p in cands.items():
    aps = []
    for rep in range(3):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            m = XGBClassifier(**base, **p).fit(X.iloc[tr], y[tr])
            aps.append(average_precision_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    print(f"{name:32s} CV AP {np.mean(aps):.4f} ± {np.std(aps)/np.sqrt(len(aps)):.4f}", flush=True)
