"""Fast local scorer that reuses the official evaluator code.

It computes the same utility, fidelity, privacy and final points as
`python -m hackathon_synth_data_2026.main`, but skips the plots, the 2000-sample
bootstrap and the permutation importances, which do not affect the score.

Usage (from the repo root):
    uv run python synthenauts/score.py results/submission.parquet [more.parquet ...]
    uv run python synthenauts/score.py syn.parquet --no-privacy        # quicker, utility + fidelity only
    uv run python synthenauts/score.py syn.parquet --train X --test Y   # score against another real split
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from hackathon_synth_data_2026 import fidelity as _fid
from hackathon_synth_data_2026 import privacy as _priv
from hackathon_synth_data_2026 import utility as _util
from hackathon_synth_data_2026.validation import normalize_submission

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "src/hackathon_synth_data_2026/eval_config.json").read_text())
SPEC = CONFIG["data_specs"]
TRAIN = ROOT / "data/real_train.parquet"
TEST = ROOT / "data/real_test.parquet"

# Permutation importances are reported by the evaluator but never scored; stub them out.
_util.permutation_importance = lambda model, X, y, **kw: types.SimpleNamespace(
    importances_mean=np.zeros(X.shape[1]), importances_std=np.zeros(X.shape[1]))

PARAMS = dict(CONFIG["downstream_model_kwargs"]["hyperparameters"])
PARAMS.setdefault("random_state", 42)
PARAMS.setdefault("enable_categorical", True)
PARAMS.setdefault("tree_method", "hist")
THRESHOLD = CONFIG["downstream_model_kwargs"].get("threshold", 0.5)

VITALS = ["DiasABP", "GCS", "Glucose", "HR", "MAP", "NIDiasABP", "NIMAP",
          "NISysABP", "RespRate", "SaO2", "SysABP", "Temp"]
SUFFIX = ["first", "last", "lowest", "highest", "median"]

_trtr_cache = {}


def _load(path_or_df):
    return path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else _util.load_table(path_or_df)


def utility_ap(syn, train=TRAIN, test=TEST):
    """Return (ap_tstr, ap_trtr, auc_tstr) exactly as the official utility evaluator computes them."""
    key = (str(train), str(test))
    real_train, real_test = (_util.load_real(p, SPEC) for p in (train, test))
    if key not in _trtr_cache:
        _trtr_cache[key] = _util.evaluate_arm(real_train, real_test, SPEC, PARAMS, THRESHOLD)[0]["pr_auc"]
    s, report = _util.validate_table(_load(syn), SPEC)
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    res = _util.evaluate_arm(s, real_test, SPEC, PARAMS, THRESHOLD)[0]
    return res["pr_auc"], _trtr_cache[key], res["auc"]


def fidelity_parts(syn, train=TRAIN):
    real = normalize_submission(_util.load_table(train), SPEC)
    s = normalize_submission(_load(syn), SPEC)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _fid.evaluate_fidelity(real, s, target_col=SPEC["target_var"], id_col=SPEC["identifier_var"],
                                     fidelity_kwargs=CONFIG["fidelity_kwargs"])
    return out["score"], out["components"]


def privacy_parts(syn_path, train=TRAIN, test=TEST):
    cats = ["In-hospital_death" if c == SPEC["target_var"] else c for c in SPEC.get("categorical_vars", [])]
    res = _priv.evaluate_privacy(train=train, control=test, synthetic=syn_path, budget=500,
                                 privacy_options={"categorical_columns": cats, "block_rows": 500}, data_specs=SPEC)
    s = res["summary"]
    if not s["technically_complete"]:
        raise RuntimeError(f"privacy incomplete: {s['reasons'][:3]}")
    return s["privacy_score"], s["components"]


def clinical_violation_rate(df):
    bad = pd.Series(False, index=df.index)
    for v in VITALS:
        lo, hi = df[f"{v}_lowest"], df[f"{v}_highest"]
        for s in ["first", "last", "median"]:
            c = df[f"{v}_{s}"]
            bad |= (c < lo) | (c > hi)
        bad |= lo > hi
    for dia, sys_ in [("DiasABP", "SysABP"), ("NIDiasABP", "NISysABP")]:
        for s in SUFFIX:
            bad |= df[f"{dia}_{s}"] > df[f"{sys_}_{s}"]
    for s in SUFFIX:
        g = df[f"GCS_{s}"]
        bad |= (g < 3) | (g > 15)
    return float(bad.mean())


def score(syn, train=TRAIN, test=TEST, with_privacy=True):
    """Score a synthetic table (path or DataFrame). Returns a flat dict."""
    t0 = time.time()
    df = _load(syn)
    ap_tstr, ap_trtr, auc = utility_ap(df, train, test)
    util = min(1.0, ap_tstr / ap_trtr)
    fid, comps = fidelity_parts(df, train)
    out = dict(ap_tstr=ap_tstr, ap_trtr=ap_trtr, auc_tstr=auc, utility=util, fidelity=fid,
               **{f"fid_{k}": v for k, v in comps.items()}, clinical_violations=clinical_violation_rate(df))
    if with_privacy:
        if isinstance(syn, pd.DataFrame):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "syn.parquet"
                df.to_parquet(path, index=False)
                priv, pcomps = privacy_parts(path, train, test)
        else:
            priv, pcomps = privacy_parts(Path(syn), train, test)
        out.update(privacy=priv, **{f"priv_{k}": v for k, v in pcomps.items()})
        out["points"] = 100 * (util + fid + priv) / 3
    out["seconds"] = time.time() - t0
    return out


def _score_one(args):
    f, train, test, with_privacy = args
    return score(f, Path(train), Path(test), with_privacy=with_privacy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--no-privacy", action="store_true")
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--test", default=str(TEST))
    ap.add_argument("--jobs", type=int, default=1, help="score several files in parallel")
    args = ap.parse_args()
    rows = []
    run = lambda f: score(f, Path(args.train), Path(args.test), with_privacy=not args.no_privacy)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_score_one, [(f, args.train, args.test, not args.no_privacy) for f in args.files]))
    else:
        results = map(run, args.files)
    for f, r in zip(args.files, results):
        r["file"] = Path(f).name
        rows.append(r)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    if len(rows) > 1:
        df = pd.DataFrame(rows).set_index("file")
        cols = [c for c in ["points", "utility", "fidelity", "privacy", "ap_tstr", "fid_correlation",
                            "fid_dependency", "priv_membership", "priv_near_copy"] if c in df]
        print(df[cols].round(4).to_string())
        print("mean:\n" + df[cols].mean().round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
