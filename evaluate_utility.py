"""
evaluate_utility.py — TSTR / TRTR utility evaluation of a synthetic dataset.

Runnable entry point. Takes the path to a participant-supplied synthetic dataset
and the experiment config, validates the submission against data_specs, then:
  * prep X/y from the flat modeling table using the spec
  * run the two comparisons on the SAME real test rows:
      - TRTR: train on real_train,  test on real_test   (reference ceiling)
      - TSTR: train on synth_train, test on real_test   (the claim under test)
  * draw ROC / PR / calibration figures
  * persist JSON-light metrics to <run_dir>/metrics/utility.json, pickling the
    fitted models to <run_dir>/artifacts
  * return {"trtr": ..., "tstr": ...}

The TRTR-vs-TSTR gap (especially on PR-AUC) is the utility headline.

Usage
-----
    uv run evaluate_utility.py <path_to_experiment_config.json> --synth <synthetic.csv>
    uv run evaluate_utility.py <path_to_experiment_config.json> --synth <synthetic.csv> --validate-only
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, brier_score_loss,
                             confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.calibration import calibration_curve
import pickle

# import load
# import provenance
# import utility_metrics as um
# from io_utils import ensure_run_dirs, write_json, strip_and_pickle_model
# from log_experiment import RunLogger

# --------------------------------------------------------------------------- #
# JSON safety: numpy scalars/arrays and pandas objects aren't serializable      #
# as-is. Every stage writes JSON, so this lives once, here.                    #
# --------------------------------------------------------------------------- #
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict(orient="records")
        if isinstance(o, pd.Series):
            return o.to_dict()
        return super().default(o)


def write_json(path, obj):
    """Dump obj to path as JSON, tolerating numpy/pandas types."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


# --------------------------------------------------------------------------- #
# Run folder layout — identical for every stage so the sweep aggregator can    #
# rely on a uniform structure.                                                 #
# --------------------------------------------------------------------------- #
def ensure_run_dirs(run_dir):
    """Create (if needed) and return the standard run subfolders.

    Returns (run_dir, figures, metrics, artifacts) as Path objects.
    """
    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    metrics = run_dir / "metrics"
    artifacts = run_dir / "artifacts"
    for d in (figures, metrics, artifacts):
        d.mkdir(parents=True, exist_ok=True)
    return run_dir, figures, metrics, artifacts


# --------------------------------------------------------------------------- #
# X/y preparation                                                              #
# --------------------------------------------------------------------------- #
def _prep_xy(df, target_col, target_mapping, drop_cols, categorical_cols):
    """Split a flat table into (X, y). One-hot encodes categoricals; column
    alignment between train and test is handled by the caller."""
    y = df[target_col].map(target_mapping).astype(int)
    feature_cols = [c for c in df.columns
                    if c not in set(drop_cols) | {target_col}]
    X = df[feature_cols].copy()
    cats = [c for c in categorical_cols if c in X.columns]
    if cats:
        for cat in cats:
            X[cat] = X[cat].astype('category')
        # X = pd.get_dummies(X, columns=cats, dtype=int)
    return X, y


def _align_columns(X_train, X_test):
    """Ensure test has exactly the training columns, in the same order
    (one-hot categories present in one split but not the other -> 0)."""
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


# --------------------------------------------------------------------------- #
# Figures (domain-specific — stay local to this stage)                         #
# --------------------------------------------------------------------------- #
def _save_utility_figures(trtr, tstr, save_dir):
    # ROC — TRTR vs TSTR on the same axes
    fig, ax = plt.subplots(figsize=(6, 6))
    for res, name in ((trtr, "TRTR"), (tstr, "TSTR")):
        fpr, tpr, _ = roc_curve(res["_y_true"], res["_y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — TRTR vs TSTR"); ax.legend()
    fig.tight_layout(); fig.savefig(save_dir / "roc_trtr_vs_tstr.png", bbox_inches="tight")
    plt.close(fig)

    # PR — the more informative curve under class imbalance
    fig, ax = plt.subplots(figsize=(6, 6))
    for res, name in ((trtr, "TRTR"), (tstr, "TSTR")):
        prec, rec, _ = precision_recall_curve(res["_y_true"], res["_y_prob"])
        ax.plot(rec, prec, label=f"{name} (PR-AUC={res['pr_auc']:.3f})")
    ax.axhline(trtr["class_prevalence_test"], color="k", ls="--", alpha=0.4,
               label=f"prevalence={trtr['class_prevalence_test']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — TRTR vs TSTR"); ax.legend()
    fig.tight_layout(); fig.savefig(save_dir / "pr_trtr_vs_tstr.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true, prob_tstr, prob_trtr, save_dir,
                     n_bins=10, strategy="quantile",
                     filename="calibration_tstr_vs_trtr"):
    """
    Parameters
    ----------
    y_true      : true labels of the REAL test set (0/1)
    prob_tstr   : predicted P(y=1) from the synthetic-trained model, on real test
    prob_trtr   : predicted P(y=1) from the real-trained model, on real test
    save_dir    : figures directory
    n_bins      : number of calibration bins
    strategy    : 'quantile' (equal-count bins; better for small / imbalanced data)
                  or 'uniform' (equal-width bins)
    filename    : stem for the saved figure (no extension)

    Returns
    -------
    dict of Brier scores (lower = better calibrated) for recording in utility.json
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    curves = {
        "TRTR": (np.asarray(prob_trtr), "tab:blue"),
        "TSTR": (np.asarray(prob_tstr), "tab:orange"),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")

    briers = {}
    for name, (prob, color) in curves.items():
        brier = brier_score_loss(y_true, prob)
        briers[f"brier_{name.lower()}"] = float(brier)
        # frac_pos = observed fraction positive per bin (y); mean_pred = mean predicted per bin (x)
        frac_pos, mean_pred = calibration_curve(
            y_true, prob, n_bins=n_bins, strategy=strategy)
        ax.plot(mean_pred, frac_pos, marker="o", color=color,
                label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed fraction positive (per bin)")
    ax.set_title("Calibration — TSTR vs TRTR (on real test set)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(save_dir / f"{filename}.{ext}", bbox_inches="tight")
    plt.close(fig)

    # difference is the headline: positive => TSTR worse-calibrated than TRTR
    briers["brier_diff_tstr_minus_trtr"] = (
        briers["brier_tstr"] - briers["brier_trtr"])
    return briers


# --------------------------------------------------------------------------- #
# Comparison of AUC between TRTR and TSTR                                      #
# --------------------------------------------------------------------------- #
def bootstrap_metric_difference_ci(y_true, y_prob_trtr, y_prob_tstr, n_bootstrap):
    auc_diffs = []
    ap_diffs = []
    n = n_bootstrap
    rng = np.random.RandomState(42)
    test_size = len(y_true)
    for _ in range(n):
        idx = rng.randint(0, test_size, size=test_size)
        if len(np.unique(y_true[idx])) < 2:      # skip degenerate resamples (only 1 class)
            continue
        auc_diffs.append(roc_auc_score(y_true[idx], y_prob_trtr[idx])
                    - roc_auc_score(y_true[idx], y_prob_tstr[idx]))
        ap_diffs.append(average_precision_score(y_true[idx], y_prob_trtr[idx])
                    - average_precision_score(y_true[idx], y_prob_tstr[idx]))

    def ci(d): return np.percentile(d, [2.5, 97.5])

    return {"auc_diff_ci": ci(auc_diffs), "pr_auc_diff_ci": ci(ap_diffs)}

"""
utility_metrics.py — downstream-model training/evaluation computation.

Model-agnostic: given prepared (X_train, y_train, X_test, y_test) and a model
class + kwargs, fit and score. Knows nothing about splits, files, or the spec —
that wiring lives in evaluate_utility.py. Used for both TRTR and TSTR by calling
it twice with different training data and the SAME real test set.
"""

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score)
from sklearn.inspection import permutation_importance
import pandas as pd


def evaluate_downstream_utility(X_train, y_train, X_test, y_test,
                                model_class=None,
                                model_kwargs=None,
                                threshold=0.5):
    """
    Fit a downstream classifier and evaluate on the (real) test set.

    AUC and PR-AUC are threshold-free and are the headline numbers — especially
    PR-AUC, given the transfusion label is a minority class. precision/recall/f1
    depend on `threshold` (default 0.5, recorded in the output) and should be
    read as secondary. Designed for fixed-hyperparameter baseline comparison.
    """
    # 1. Defaults (XGBoost imported lazily so the module loads without it)
    if model_class is None:
        from xgboost import XGBClassifier
        model_class = XGBClassifier
    if model_kwargs is None:
        model_kwargs = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'enable_categorical': True,
        }

    # 2. Train
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)

    # 3. Score — get probabilities robustly, then threshold consistently
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):     # e.g. SVM without probability
        s = model.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        y_prob = model.predict(X_test).astype(float)
    y_pred = (y_prob >= threshold).astype(int)     # honor the recorded threshold

    # 4. Permutation importance
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                  random_state=model_kwargs['random_state'], scoring='roc_auc')
    feat_names = getattr(model, 'feature_names_in_', X_test.columns)
    feats_df = pd.DataFrame({
        'permutation_importance_mean': perm_imp.importances_mean,
        'permutation_importance_std': perm_imp.importances_std
    }, index=feat_names).sort_values('permutation_importance_mean', ascending=False)

    # 5. Structured results
    n_pos = int(y_test.sum())
    return {
        "n_positive": n_pos,
        "n_negative": int(len(y_test) - n_pos),
        "class_prevalence_test": float(y_test.mean()),
        "class_prevalence_train": float(y_train.mean()),
        "auc": float(roc_auc_score(y_test, y_prob)),           # headline (ranking)
        "pr_auc": float(average_precision_score(y_test, y_prob)),  # headline (imbalance)
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "threshold": float(threshold),
        "model_class": model_class.__name__,
        "model_kwargs": model_kwargs,
        "fitted_model": model,                     # adapter strips this to artifacts/
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "permutation_importances_df": feats_df,
        # kept for the ROC/PR figures the adapter draws:
        "_y_true": y_test,
        "_y_prob": y_prob,
    }

# --------------------------------------------------------------------------- #
# Main evaluation logic                                                        #
# --------------------------------------------------------------------------- #
def evaluate_utility(synth_train_df,
                     real_train_df,
                     real_test_df,
                     run_dir,
                     *,
                     target_col,
                     target_mapping,
                     drop_cols=["patientid"],
                     categorical_cols=None,
                     model_class=None,
                     model_kwargs=None,
                     n_bootstrap=100,
                     threshold=0.5):
    """
    Parameters
    ----------
    synth_train_df : synthetic data (participant submission) -> TSTR train
    real_train_df  : real TRAIN split -> TRTR train
    real_test_df   : held-out real split -> the common test set for BOTH
    run_dir        : per-run output folder
    target_col, drop_cols, categorical_cols : from the shared data_spec

    Returns
    -------
    {"trtr": {...}, "tstr": {...}}  (also written to metrics/utility.json)
    """
    assert isinstance(drop_cols, list), 'drop_cols must be a list'
    assert isinstance(categorical_cols, list), 'categorical_cols must be a list'

    run_dir, figs, metrics_dir, artifacts = ensure_run_dirs(run_dir)

    # test set is prepped ONCE and shared, so TRTR and TSTR are compared fairly
    X_test, y_test = _prep_xy(real_test_df, target_col, target_mapping, drop_cols, categorical_cols)

    def _run(train_df):
        X_tr, y_tr = _prep_xy(train_df, target_col, target_mapping, drop_cols, categorical_cols)
        X_tr, X_te = _align_columns(X_tr, X_test)     # test conforms to train's columns
        return evaluate_downstream_utility(
            X_tr, y_tr, X_te, y_test,
            model_class=model_class, model_kwargs=model_kwargs, threshold=threshold)

    trtr = _run(real_train_df)     # reference: train on real
    tstr = _run(synth_train_df)    # the claim: train on synthetic

    _save_utility_figures(trtr, tstr, figs)

    for res, name in ((trtr, "trtr"), (tstr, "tstr")):
        # confusion matrices as csv (cheap, useful for the thresholded read)
        y_pred = (res["_y_prob"] >= threshold).astype(int)
        cm = confusion_matrix(res["_y_true"], y_pred)
        pd.DataFrame(cm, index=["true_neg", "true_pos"],
                     columns=["pred_neg", "pred_pos"]).to_csv(
        metrics_dir / f"confusion_{name}.csv")

    briers = plot_calibration(trtr['_y_true'], tstr["_y_prob"], trtr["_y_prob"], save_dir=figs)

    metric_cis = bootstrap_metric_difference_ci(trtr['_y_true'], trtr['_y_prob'], tstr['_y_prob'], n_bootstrap)

    # strip heavy models + non-serializable arrays before JSON
    for res, name in ((trtr, "trtr"), (tstr, "tstr")):
        strip_and_pickle_model(res, "fitted_model", artifacts, f"{name}_model.pkl")
        res.pop("_y_true", None)
        res.pop("_y_prob", None)

        perm_df = res.pop("permutation_importances_df", None)
        if perm_df is not None:
            res["permutation_importances"] = (
                perm_df.reset_index()
                    .rename(columns={"index": "feature"})
                    .to_dict(orient="records"))
            # also persist as csv for eyeballing
            perm_df.to_csv(metrics_dir / f"downstream_model_permutation_importance_{name}.csv")

    result = {
        "trtr": trtr,
        "tstr": tstr,
        'auc_ci_low': metric_cis['auc_diff_ci'][0],
        'auc_ci_high': metric_cis['auc_diff_ci'][1],
        'pr_auc_diff_ci_low': metric_cis['pr_auc_diff_ci'][0],
        'pr_auc_diff_ci_high': metric_cis['pr_auc_diff_ci'][1],
        # the headline: how much utility is lost by training on synthetic
        "auc_gap_trtr_minus_tstr": float(trtr["auc"] - tstr["auc"]),
        "pr_auc_gap_trtr_minus_tstr": float(trtr["pr_auc"] - tstr["pr_auc"]),
    }
    result.update(briers)
    write_json(metrics_dir / "utility.json", result)
    _write_summary(metrics_dir / "utility_summary.txt", result)
    return result


def _write_summary(path, r):
    t, s = r["trtr"], r["tstr"]

    # CIs are on the DIFFERENCE (gap); crossing zero = within noise
    auc_low, auc_high = r["auc_ci_low"], r["auc_ci_high"]
    pr_low, pr_high = r["pr_auc_diff_ci_low"], r["pr_auc_diff_ci_high"]

    def verdict(lo, hi):
        return ("within noise — gap NOT distinguishable from 0"
                if lo <= 0 <= hi else
                "gap is distinguishable from 0")

    lines = [
        "UTILITY SUMMARY (TRTR = train real, TSTR = train synthetic; same test set)",
        "=" * 60,
        f"class prevalence (test): {t['class_prevalence_test']:.3f}   "
        f"(n_pos={t['n_positive']}, n_neg={t['n_negative']})",
        "",
        f"{'':12}{'AUC':>10}{'PR-AUC':>10}{'F1':>10}{'recall':>10}",
        f"{'TRTR':12}{t['auc']:>10.3f}{t['pr_auc']:>10.3f}{t['f1']:>10.3f}{t['recall']:>10.3f}",
        f"{'TSTR':12}{s['auc']:>10.3f}{s['pr_auc']:>10.3f}{s['f1']:>10.3f}{s['recall']:>10.3f}",
        "",
        f"AUC gap    (TRTR - TSTR): {r['auc_gap_trtr_minus_tstr']:+.3f}   "
        f"95% CI [{auc_low:+.3f}, {auc_high:+.3f}]",
        f"    -> {verdict(auc_low, auc_high)}",
        f"PR-AUC gap (TRTR - TSTR): {r['pr_auc_gap_trtr_minus_tstr']:+.3f}   "
        f"95% CI [{pr_low:+.3f}, {pr_high:+.3f}]   ",
        f"    -> {verdict(pr_low, pr_high)}",
        "",
        f"threshold for F1/recall: {s['threshold']}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Input loading                                                                #
# --------------------------------------------------------------------------- #
import warnings
def enforce_dtypes(df, data_specs):
    df = df.copy()
    try:
        for col in data_specs['boolean_vars']:
            if col in df.columns:
                df[col] = df[col].astype('boolean')
    except KeyError:
        pass
    for col in data_specs['categorical_vars']:
        if col in df.columns:
            df[col] = df[col].astype('object')
    for col in data_specs['numerical_vars']:
        if col in df.columns:
            # Keep int/float as-is; only convert if needed
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col])
    return df

def load_dataset(data_path, data_specs):
    df = pd.read_parquet(data_path)

    # Read feature definitions
    input_vars = list(data_specs["input_vars"])
    categorical_vars = set(data_specs.get("categorical_vars", []))
    numerical_vars = set(data_specs.get("numerical_vars", []))
    boolean_vars = set(data_specs.get("boolean_vars", []))

    identifier_var = data_specs["identifier_var"]
    target_var = data_specs["target_var"]
    split_var = data_specs.get("split_var")

    required_cols = [*input_vars, target_var]
    # Identifier and split are metadata, not model inputs.
    # They are retained if present.
    optional_cols = [
        col for col in [identifier_var, split_var]
        if col is not None
    ]
    keep_cols = list(dict.fromkeys(required_cols + optional_cols))

    feature_types = {
    "categorical_vars": categorical_vars - {target_var},
    "numerical_vars": numerical_vars,
    "boolean_vars": boolean_vars,
    }
    declared = set().union(*feature_types.values())
    present = set(df.columns)
    missing = declared - present
    extra = present - declared
    unused_features = declared - set(input_vars)

    if unused_features:
        raise ValueError(
            "Features declared in type lists but absent from "
            f"'input_vars': {sorted(unused_features)}"
        )
    if missing:
        raise ValueError(f"Spec declares columns absent from data: {sorted(missing)}")
    if extra:
        warnings.warn(f"Data has columns not in spec: {sorted(extra)}")
        # Report and remove unselected columns
        dropped_cols = sorted(set(df.columns) - set(keep_cols))
        if dropped_cols:
            print(
                f"[INFO] Dropping {len(dropped_cols)} columns "
                "not selected in data_specs['input_vars']:"
            )
            print("       " + ", ".join(dropped_cols))

        df = df[[col for col in keep_cols if col in df.columns]].copy()
        print(
            f"[INFO] Loaded {len(df)} rows with "
            f"{len(input_vars)} selected input features."
        )

    if df[data_specs['identifier_var']].duplicated().any():
        raise ValueError(f"'{data_specs['identifer_var']}' has duplicates — not one row per patient?")
    if df[data_specs['identifier_var']].isna().any():
        raise ValueError(
            f"Identifier '{data_specs['identifier_var']}' contains missing values."
        )

    for col in data_specs['numerical_vars']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"'{col}' declared numerical but dtype is {df[col].dtype}")

    present_target = set(df[data_specs['target_var']].unique())
    declared_target = set(data_specs['target_mapping'].keys())
    missing_target = declared_target - present_target
    extra_target = present_target - declared_target
    if missing_target:
        raise ValueError(f"Spec declares target groups absent from data: {sorted(missing_target)}")
    if extra_target:
        raise ValueError(f"Unexpected target labels in data: these groups are not in spec: {sorted(extra_target)}")
    if df[data_specs['target_var']].isna().any():
        raise ValueError(
            f"Target '{data_specs['target_var']}' contains missing values."
        )

    df = enforce_dtypes(df, data_specs)
    df = df[sorted(df.columns)]

    return df


def load_train_and_test(df_path, data_specs):
    df_train = load_dataset(os.path.join(df_path, 'real_train.parquet'), data_specs)
    df_test = load_dataset(os.path.join(df_path, 'real_test.parquet'), data_specs)

    return df_train, df_test

def load_table(path):
    p = str(path)
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    if p.endswith((".tsv", ".txt")):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)                      # .csv / .csv.gz


def load_real_splits(experiment_config, data_specs):
    """Real (train, test). Prefer explicit paths, else the shared trainTestSplit dir."""
    rt = experiment_config.get('real_train_path')
    rte = experiment_config.get('real_test_path')
    if rt and rte:
        return load_table(rt), load_table(rte)
    return load_train_and_test(experiment_config['real_data_path'], data_specs)


# --------------------------------------------------------------------------- #
# The pattern both fidelity and utility repeat: a results dict contains a       #
# heavy fitted model that must NOT go into JSON. Pop it, pickle it to           #
# artifacts/, and replace it with the pickle's path. Enforced in one place so   #
# the "no heavy objects in JSON" rule can't drift between stages.               #
# --------------------------------------------------------------------------- #
def strip_and_pickle_model(result, key, artifacts_dir, filename):
    """Pop result[key] (a fitted model), pickle it to artifacts_dir/filename,
    and set result[key + '_path'] to that location. No-op if key is absent.

    Mutates and returns `result` for convenience.
    """
    model = result.pop(key, None)
    if model is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / filename
        with open(path, "wb") as f:
            pickle.dump(model, f)
        result[f"{key}_path"] = str(path)
    return result

# --------------------------------------------------------------------------- #
# Format validation against data_specs                                         #
# --------------------------------------------------------------------------- #
def validate_synthetic_submission(synth_df, data_specs):
    """Gate a submission against data_specs and return (sanitized_df, report).

    Two reasons this must run before evaluate_utility():
      * _align_columns() reindexes the real TEST set onto the SYNTHETIC column
        set, so a missing column would be silently dropped rather than raising.
      * _prep_xy() builds X from every column present, so extra columns would
        become features — and because each arm aligns to its own training
        columns, TRTR and TSTR would then be evaluated on different feature
        sets. The returned frame is restricted to the spec's column set so both
        arms see identical features.
    """
    target_var = data_specs['target_var']
    target_mapping = data_specs['target_mapping']
    id_var = data_specs.get('identifier_var')
    num_vars = list(data_specs.get('numerical_vars', []))
    bool_vars = list(data_specs.get('boolean_vars', []))
    required = list(dict.fromkeys(
        data_specs.get('input_vars') or (num_vars + bool_vars)))

    errors, warnings = [], []

    missing = [c for c in required if c not in synth_df.columns]
    if missing:
        errors.append(f"missing required feature columns: {missing}")
    if target_var not in synth_df.columns:
        errors.append(f"missing target column: '{target_var}'")
    if id_var and id_var not in synth_df.columns:
        warnings.append(f"missing identifier column: '{id_var}' (ignored by the scorer)")

    keep = ([id_var] if id_var else []) + required + [target_var]
    extra = [c for c in synth_df.columns
             if c not in set(keep)]
    if extra:
        warnings.append(f"{len(extra)} extra column(s) ignored: {extra[:10]}")

    if len(synth_df) == 0:
        errors.append("synthetic table has 0 rows")

    # target must map to 0/1 with both classes present
    if target_var in synth_df.columns:
        mapped = synth_df[target_var].map(target_mapping)
        if mapped.isna().any():
            bad = sorted(set(synth_df.loc[mapped.isna(), target_var].astype(str).unique()))
            errors.append(f"unmappable target values {bad}; "
                          f"allowed per target_mapping: {sorted(map(str, target_mapping))}")
        else:
            n_pos, n_neg = int(mapped.sum()), int(len(mapped) - mapped.sum())
            if n_pos == 0 or n_neg == 0:
                errors.append(f"degenerate synthetic target: n_pos={n_pos}, "
                              "n_neg={n_neg}; both classes are required to fit the model")

    # declared numerical columns must be numerically parseable
    for c in num_vars:
        if c in synth_df.columns and synth_df[c].dtype == object:
            coerced = pd.to_numeric(synth_df[c], errors="coerce")
            if bool((synth_df[c].notna() & coerced.isna()).any()):
                errors.append(f"column '{c}' is declared numerical but holds "
                              "non-numeric values")

    # boolean columns: representation drift (0/1 vs True/False) is the common failure
    for c in bool_vars:
        if c in synth_df.columns:
            vals = set(synth_df[c].dropna().unique().tolist())
            if not vals <= {0, 1, True, False, 0.0, 1.0}:
                errors.append(f"boolean column '{c}' has unexpected values: "
                              f"{sorted(map(str, vals))[:8]}")

    # informational: SDV regenerates the primary key; collisions are a leakage
    # question, not a utility one.
    if id_var and id_var in synth_df.columns and synth_df[id_var].duplicated().any():
        warnings.append(f"duplicate values in '{id_var}'")

    report = {
        "schema_check": "PASS" if not errors else "FAIL",
        "n_rows": int(len(synth_df)),
        "required_cols": required,
        "missing_cols": missing,
        "ignored_extra_cols": extra,
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        return None, report
    return synth_df.loc[:, [c for c in keep if c in synth_df.columns]], report


# --------------------------------------------------------------------------- #
# Downstream model from config                                                 #
# --------------------------------------------------------------------------- #
def _resolve_model_class(name):
    """Map the config's model_class string to a class; None -> pipeline default."""
    if name is None:
        return None
    if name == "XGBClassifier":
        from xgboost import XGBClassifier; return XGBClassifier
    if name == "HistGradientBoostingClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier
    if name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier; return RandomForestClassifier
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression; return LogisticRegression
    raise ValueError(f"Unknown model_class '{name}'; extend _resolve_model_class().")


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_utility_eval(experiment_config, synthetic_data_path, logger=None):
    data_specs = experiment_config['data_specs']
    out_path = experiment_config['evaluation_output_path']
    os.makedirs(out_path, exist_ok=True)
    write_json(os.path.join(out_path, 'experiment_config.json'), experiment_config)

    print('Loading synthetic dataset...')
    synth_df = load_table(synthetic_data_path)

    print('Validating synthetic data against data_specs...')
    synth_df, report = validate_synthetic_submission(synth_df, data_specs)
    metrics_dir = os.path.join(out_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    write_json(os.path.join(metrics_dir, 'schema_check.json'), report)
    for w in report['warnings']:
        print(f'  WARN: {w}')
    if report['schema_check'] == 'FAIL':
        raise ValueError("Synthetic submission does not match the data specs:\n  - "
                         + "\n  - ".join(report['errors']))
    print(f"  schema check PASSED ({report['n_rows']} rows, "
          f"{len(report['required_cols'])} features)")

    if experiment_config.get('validate_only', False):
        print('validate_only=true -> stopping before evaluation.')
        return report

    print('Loading real train/test splits...')
    realdf_train, realdf_test = load_real_splits(experiment_config, data_specs)

    print('Evaluating utility...')
    model_kwargs = experiment_config['downstream_model_kwargs']
    util_metrics = evaluate_utility(synth_df,
                        realdf_train,
                        realdf_test,
                        out_path,
                        target_col=data_specs['target_var'],
                        target_mapping=data_specs['target_mapping'],
                        drop_cols=[data_specs['identifier_var']],
                        categorical_cols=data_specs['categorical_vars'],
                        model_class=_resolve_model_class(model_kwargs['model_class']),
                        model_kwargs=model_kwargs['hyperparameters'],
                        n_bootstrap=model_kwargs['n_bootstrap'],
                        threshold=model_kwargs['threshold'])

    return util_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a synthetic dataset's downstream utility (TSTR vs TRTR).")
    parser.add_argument('experiment_config',
                        help='Path to the experiment config JSON.')
    parser.add_argument('--synth', default=None,
                        help='Path to the synthetic dataset. Falls back to the '
                             '"synthetic_data_path" key in the config.')
    parser.add_argument('--output-path', default=None,
                        help='Override evaluation_output_path / output_path.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run the format checks only, then exit.')
    args = parser.parse_args()

    with open(args.experiment_config, 'r') as f:
        experiment_config = json.load(f)

    synth_path = args.synth or experiment_config.get('synthetic_data_path')
    if not synth_path:
        parser.error("No synthetic dataset given: pass --synth or set "
                     "'synthetic_data_path' in the config.")

    if args.output_path:
        experiment_config['evaluation_output_path'] = args.output_path
    if args.validate_only:
        experiment_config['validate_only'] = True
    out_path = experiment_config['evaluation_output_path']
    os.makedirs(out_path, exist_ok=True)

    run_utility_eval(experiment_config, synth_path)



if __name__ == "__main__":
    main()

    # Usage
    # uv run evaluate_utility.py <path_to_experiment_config.json> --synth <path_to_synthetic_data.csv>