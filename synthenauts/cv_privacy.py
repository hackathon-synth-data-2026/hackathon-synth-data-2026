"""Check a privacy setting on internal folds: fit on 80% of real_train, held-out 20% as the control set.
Usage: uv run python synthenauts/cv_privacy.py"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as E, score as S, synth


def job(a):
    noise, i, pa, pb = a
    syn = synth.generate(pd.read_parquet(pa), seed=0, **{**synth.BEST, "demo_noise": noise})
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.parquet"; syn.to_parquet(p, index=False)
        priv, comps = S.privacy_parts(p, pa, pb)
    return dict(noise=noise, fold=i, privacy=priv, **comps)


if __name__ == "__main__":
    tasks = [(z, i, pa, pb) for z in [0.0, 0.25, 0.4] for i, (pa, pb) in enumerate(E.folds())]
    with ProcessPoolExecutor(8) as pool:
        d = pd.DataFrame(pool.map(job, tasks))
    print(d.pivot_table(index="fold", columns="noise", values="privacy").round(4).to_string())
    print(d.groupby("noise")[["privacy", "membership", "near_copy"]].mean().round(4).to_string())
