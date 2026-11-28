"""Experiment harness for the generator settings in synth.py.

Two protocols:
  cv      Utility only, by 5-fold CV inside real_train. real_test is never used, so this
          is what we use to choose utility-related settings (label model, temperature, leaves).
  official  Full official score (utility, fidelity, privacy) on real_train/real_test for
          several seeds. Used for reporting and for fidelity/privacy settings, which
          do not involve the test labels.

Usage:
    uv run python synthenauts/experiments.py cv        --configs synthenauts/configs.json --names a b --seeds 0 1 2
    uv run python synthenauts/experiments.py official  --configs synthenauts/configs.json --names a b --seeds 0 1 2
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402
import synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/.cv_folds"


def folds(k=5, seed=0):
    """Write k stratified train/validation splits of real_train to disk (once)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    real = pd.read_parquet(S.TRAIN)
    paths = []
    for i, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(real, real[synth.TARGET])):
        pa, pb = CACHE / f"fold{i}_train.parquet", CACHE / f"fold{i}_val.parquet"
        if not pa.exists():
            real.iloc[a].to_parquet(pa, index=False)
            real.iloc[b].to_parquet(pb, index=False)
        paths.append((pa, pb))
    return paths


def _cv_job(job):
    name, cfg, seed, i, pa, pb, with_fid = job
    syn = synth.generate(pd.read_parquet(pa), seed=seed, **cfg)
    ap_tstr, ap_trtr, _ = S.utility_ap(syn, pa, pb)
    out = dict(name=name, seed=seed, fold=i, ap_tstr=ap_tstr, ap_trtr=ap_trtr, ratio=ap_tstr / ap_trtr)
    if with_fid:
        out["fidelity"], comps = S.fidelity_parts(syn, pa)
        out.update({f"fid_{k}": v for k, v in comps.items()})
    return out


def _official_job(job):
    name, cfg, seed, out_dir = job
    syn = synth.generate(pd.read_parquet(S.TRAIN), seed=seed, **cfg)
    path = Path(out_dir) / f"{name}_s{seed}.parquet"
    syn.to_parquet(path, index=False)
    r = S.score(path)
    return dict(name=name, seed=seed, **r)


def run(kind, configs, seeds, jobs=6, with_fid=False, out_dir=None):
    if kind == "cv":
        tasks = [(n, c, s, i, pa, pb, with_fid) for n, c in configs.items() for s in seeds
                 for i, (pa, pb) in enumerate(folds())]
        fn = _cv_job
    else:
        out_dir = Path(out_dir or ROOT / "results/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(n, c, s, str(out_dir)) for n, c in configs.items() for s in seeds]
        fn = _official_job
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(fn, tasks))
    df = pd.DataFrame(rows)
    cols = [c for c in ["ratio", "ap_tstr", "points", "utility", "fidelity", "privacy", "fid_marginals",
                        "fid_correlation", "fid_dependency", "fid_missingness", "priv_near_copy",
                        "priv_membership", "clinical_violations"] if c in df]
    summary = df.groupby("name", sort=False)[cols].agg(["mean", "std"]) if kind != "cv" else \
        df.groupby("name", sort=False)[cols].mean()
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["cv", "official"])
    ap.add_argument("--configs", default=str(ROOT / "synthenauts/configs.json"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--fid", action="store_true", help="cv: also compute fidelity per fold")
    ap.add_argument("--save", help="csv path for the per-run rows")
    args = ap.parse_args()
    configs = json.loads(Path(args.configs).read_text())
    if args.names:
        configs = {n: configs[n] for n in args.names}
    df, summary = run(args.kind, configs, args.seeds, args.jobs, args.fid)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())
    if args.save:
        df.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()
