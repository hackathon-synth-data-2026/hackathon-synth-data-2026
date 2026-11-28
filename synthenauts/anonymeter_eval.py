"""Privacy attacks with Anonymeter (singling out, linkability, inference) on a synthetic table.

Anonymeter (https://pypi.org/project/anonymeter/) simulates three attackers who only see the
synthetic table and try to learn something about real_train patients. Each attack is repeated on
real_test patients as a control (patients the generator never saw), and the risk is the part of
the attack's success that the control cannot explain. 0 = no leak, 1 = complete leak.

Anonymeter pins numpy<1.27 and pyarrow<22, while the project needs numpy>=2 and pyarrow>=24, so
it cannot live in the project environment. Run it in its own throw-away environment (the project
lockfile is not touched):

    uv run --no-project --python 3.12 --with anonymeter --with pandas --with pyarrow \\
        python synthenauts/anonymeter_eval.py synthenauts/submissions/synthenauts_submission.parquet

Missing values: Anonymeter's distance is undefined when only one of two values is missing, so the
attacker only uses columns observed in at least --min-observed of real_train (default 95%) and
only patients complete on those columns. The row counts are reported. This is a different (and
weaker-knowledge) attacker than the official privacy metric, so the numbers are not comparable.
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import pandas as pd
from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ID, TARGET = "recordid", "mortality"
# Linkability: the attacker holds two separate views of a patient (e.g. two databases) and tries
# to link them through the synthetic table. View A: admission data and scores; view B: first labs.
LINK_A = ["Age", "Gender", "SAPS-I", "SOFA", "CCU", "CSRU", "SICU"]
LINK_B = ["HR_first", "GCS_first", "Temp_first", "Glucose_first", "Creatinine_first", "BUN_first",
          "HCT_first", "Platelets_first", "Na_first", "K_first"]


def prepare(df, columns):
    """Complete cases on ``columns``; flags and the target become categorical strings."""
    out = df.loc[df[columns].notna().all(axis=1), columns].reset_index(drop=True)
    for c in columns:
        if str(out[c].dtype) == "boolean" or c == TARGET:
            out[c] = out[c].astype(int).astype(str)
        else:
            out[c] = out[c].astype(float)
    return out


def summarise(name, evaluator, seconds, **extra):
    r = evaluator.results()
    risk = evaluator.risk()
    return dict(
        attack=name, **extra,
        risk=round(float(risk.value), 4), risk_ci95=[round(float(risk.ci[0]), 4), round(float(risk.ci[1]), 4)],
        attack_success=round(float(r.attack_rate.value), 4), control_success=round(float(r.control_rate.value), 4),
        baseline_success=round(float(r.baseline_rate.value), 4),
        n_success=int(r.n_success), n_control=int(r.n_control), n_baseline=int(r.n_baseline),
        n_attacks=int(r.n_attacks_ori), seconds=round(seconds, 1),
    )


def run(args):
    train = pd.read_parquet(args.train)
    control = pd.read_parquet(args.control)
    synth = pd.read_parquet(args.synth)
    missing = [c for c in train.columns if c not in synth.columns]
    if missing:
        raise SystemExit(f"{args.synth} lacks columns {missing[:5]}")

    observed = train.notna().mean()
    features = [c for c in train.columns if c not in (ID, TARGET) and observed[c] >= args.min_observed]
    attack_cols = features + [TARGET]
    ori, ctl, syn = (prepare(df, attack_cols) for df in (train, control, synth))
    print(f"attacker columns: {len(features)} features (+ target); rows used: "
          f"train {len(ori)}/{len(train)}, control {len(ctl)}/{len(control)}, synthetic {len(syn)}/{len(synth)}")

    results, n = [], min(args.n_attacks, len(ctl), len(ori))
    if n < args.n_attacks:
        print(f"n_attacks capped at {n}: the control set has only {len(ctl)} complete patients")
    features_only = lambda df: df[features]

    t = time.time()
    so = SinglingOutEvaluator(ori=features_only(ori), syn=features_only(syn), control=features_only(ctl),
                              n_attacks=n, n_cols=args.n_cols, seed=args.seed)
    so.evaluate(mode="multivariate")
    results.append(summarise("singling_out", so, time.time() - t, n_cols=args.n_cols))

    groups = ([c for c in LINK_A if c in features], [c for c in LINK_B if c in features])
    t = time.time()
    link = LinkabilityEvaluator(ori=features_only(ori), syn=features_only(syn), control=features_only(ctl),
                                aux_cols=groups, n_attacks=n, n_neighbors=1)
    link.evaluate(n_jobs=args.n_jobs)
    results.append(summarise("linkability", link, time.time() - t, aux_a=len(groups[0]), aux_b=len(groups[1])))

    for secret in args.secrets:
        t = time.time()
        aux = [c for c in features if c != secret]
        data = (ori, syn, ctl)
        inf = InferenceEvaluator(ori=data[0], syn=data[1], control=data[2], aux_cols=aux, secret=secret,
                                 n_attacks=n)
        inf.evaluate(n_jobs=args.n_jobs)
        results.append(summarise(f"inference[{secret}]", inf, time.time() - t, aux_columns=len(aux)))

    worst = max(r["risk_ci95"][1] for r in results)
    return dict(
        synthetic=str(args.synth), attacker_features=len(features),
        rows=dict(train=len(ori), control=len(ctl), synthetic=len(syn)),
        results=results,
        worst_case_risk=worst,  # highest upper 95% bound over the attacks
        anonymeter_privacy_score=round(1 - worst, 4),  # our summary (1 - worst case), not an Anonymeter output
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synth", help="synthetic table (Parquet) with the real_train schema")
    ap.add_argument("--train", default=str(ROOT / "data/real_train.parquet"))
    ap.add_argument("--control", default=str(ROOT / "data/real_test.parquet"),
                    help="real patients the generator never saw (the official metric also uses real_test)")
    ap.add_argument("--n-attacks", type=int, default=500)
    ap.add_argument("--n-cols", type=int, default=3, help="columns per singling-out query")
    ap.add_argument("--min-observed", type=float, default=0.95)
    ap.add_argument("--secrets", nargs="+", default=[TARGET, "Gender"], help="columns the inference attacker guesses")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="JSON output (default: results/anonymeter_<file>.json)")
    args = ap.parse_args()
    logging.getLogger("anonymeter").setLevel(logging.ERROR)

    report = run(args)
    out = Path(args.out) if args.out else ROOT / "results" / f"anonymeter_{Path(args.synth).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'attack':22s} {'risk':>7s}  {'95% CI':>16s}  {'attack':>7s} {'control':>8s} {'baseline':>9s}")
    for r in report["results"]:
        lo, hi = r["risk_ci95"]
        print(f"{r['attack']:22s} {r['risk']:7.3f}  [{lo:6.3f}, {hi:6.3f}]  {r['attack_success']:7.3f} "
              f"{r['control_success']:8.3f} {r['baseline_success']:9.3f}")
    print(f"\nworst-case risk (highest upper bound): {report['worst_case_risk']:.3f}"
          f"  ->  anonymeter privacy score {report['anonymeter_privacy_score']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
