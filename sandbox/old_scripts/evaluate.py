# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.0",
# ]
# ///

"""
evaluate_utility.py — thin adapter over utility_metrics.py (TSTR / TRTR).

Responsibilities (and nothing else):
  * prep X/y from the flat modeling table using the spec
      - features = everything except target, id, and drop_cols
      - hours 4-5 are ALREADY absent (dropped upstream at cohort construction),
        so this stage never has to police the observation/prediction boundary
  * run the two comparisons on the SAME real test rows:
      - TRTR: train on real_train,  test on real_test   (reference ceiling)
      - TSTR: train on synth_train, test on real_test   (the claim under test)
  * draw ROC / PR / confusion figures (domain-specific, local here)
  * persist JSON-light metrics to <run_dir>/metrics/utility.json, pickling the
    fitted models to <run_dir>/artifacts
  * return {"trtr": ..., "tstr": ...}

The TRTR-vs-TSTR gap (especially on PR-AUC) is the utility headline.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, brier_score_loss,
                             confusion_matrix, roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


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
# Comparison of AUC between TRTR and TSTR                                                           #
# --------------------------------------------------------------------------- #


def bootstrap_metric_difference_ci(y_true, y_prob_trtr, y_prob_tstr, n_bootstrap):
    auc_diffs = []
    ap_diffs = []
    n = n_bootstrap
    rng = np.random.RandomState(42)
    test_size = len(y_true)
    for _ in range(n):
        idx = rng.randint(0, test_size, size=test_size)
        if len(np.unique(y_true[idx])) < 2:      # skip degenerate resamples (containing only 1 class)
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
# Main entry point                                                             #
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
    synth_train_df : synthetic data (generated from the real TRAIN split) -> TSTR train
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

    for res, train_df, name in ((trtr, real_train_df, "trtr"), (tstr, synth_train_df, "tstr")):
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


def main():
    evaluate_utility()


if __name__ == "__main__":
    main()

