# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0.0,<3.0.0",
#     "pandas==2.3.3",
#     "pyarrow>=24.0.0",
#     "matplotlib>=3.10.9",
#     "scikit-learn==1.9.1",
#     "xgboost==3.4.1",
# ]
# ///

"""
Evaluate synthetic-data utility using XGBoost.

TRTR: train on real data, evaluate on real test data.
TSTR: train on synthetic data, evaluate on the same real test data.

The existing "pr_auc" output key denotes average precision.

Usage:
    uv run evaluate_utility.py config.json --synth synthetic.csv
    uv run evaluate_utility.py config.json --synth synthetic.csv --validate-only
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "enable_categorical": True,
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def write_json(path, value):
    def encode(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            default=encode,
            allow_nan=False,
        )


def load_table(path):
    suffix = str(path).lower()

    if suffix.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)

    if suffix.endswith((".tsv", ".txt")):
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spec(spec):
    """Return the ordered feature names and their declared types."""
    target = spec["target_var"]
    features = spec.get("input_vars")

    types = {
        key: [
            column
            for column in spec.get(key, [])
            if column != target
        ]
        for key in (
            "numerical_vars",
            "boolean_vars",
            "categorical_vars",
        )
    }

    declared = [
        column
        for columns in types.values()
        for column in columns
    ]

    if features is None:
        features = declared.copy()

    if not features or len(features) != len(set(features)):
        raise ValueError(
            "input_vars must contain unique feature names"
        )

    if (
        len(declared) != len(set(declared))
        or set(declared) != set(features)
    ):
        raise ValueError(
            "Every input feature must have exactly one declared type"
        )

    metadata = {
        target,
        spec.get("identifier_var"),
        spec.get("split_var"),
    } - {None}

    if set(features) & metadata:
        raise ValueError(
            "Target, identifier and split columns cannot be features"
        )


    return list(features), types



def validate_table(df, spec):
    """Return a normalized table and a structured validation report."""
    features, types = validate_spec(spec)
    target = spec["target_var"]
    identifier = spec.get("identifier_var")
    df = df.copy()

    errors = []
    warnings = []


    if not df.columns.is_unique:
        errors.append("duplicate column names")

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:
        errors.append(
            f"missing required feature columns: {missing}"
        )

    if target not in df.columns:
        errors.append(f"missing target column: '{target}'")

    if identifier and identifier not in df.columns:
        warnings.append(
            f"missing identifier column: '{identifier}' "
            "(ignored by scorer)"
        )

    keep = (
        ([identifier] if identifier else [])
        + features
        + [target]
    )
    extra = [
        column
        for column in df.columns
        if column not in keep
    ]

    if extra:
        warnings.append(
            f"{len(extra)} extra column(s) ignored: {extra[:10]}"
        )

    if df.empty:
        errors.append("table has 0 rows")

    if df.columns.is_unique:
        if target in df.columns:
            import pdb; pdb.set_trace()
            target_col = df[target]
            if target_col.isna().any():
                errors.append("target contains missing values")
            elif not target_col.isin([0, 1]).all():
                errors.append("target must contain only 0 and 1")
            elif target_col.nunique() != 2:
                errors.append("target must contain both classes")
                

        for column in types["numerical_vars"]:
            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if (df[column].notna() & values.isna()).any():
                errors.append(
                    f"non-numeric values in '{column}'"
                )
            elif np.isinf(
                values.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            ).any():
                errors.append(
                    f"infinite values in '{column}'"
                )
            else:
                df[column] = values.astype(float)

        for column in types["boolean_vars"]:
            if column not in df.columns:
                continue

            valid = (
                df[column]
                .dropna()
                .isin([0, 1, True, False])
                .all()
            )

            if not valid:
                errors.append(
                    f"'{column}' must contain 0/1, "
                    "True/False, or missing values"
                )
            else:
                df[column] = (
                    df[column]
                    .astype("boolean")
                    .astype(float)
                )

        for column in types["categorical_vars"]:
            if column in df.columns:
                df[column] = df[column].astype("string")

        if (
            identifier
            and identifier in df.columns
            and df[identifier].duplicated().any()
        ):
            warnings.append(
                f"duplicate values in '{identifier}'"
            )

    report = {
        "schema_check": "FAIL" if errors else "PASS",
        "n_rows": int(len(df)),
        "required_cols": features,
        "missing_cols": missing,
        "ignored_extra_cols": extra,
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        return None, report

    selected = [
        column
        for column in keep
        if column in df.columns
    ]
    return df.loc[:, selected], report


def load_real(path, spec):
    df, report = validate_table(
        load_table(path),
        spec,
    )

    if report["errors"]:
        raise ValueError(
            f"Invalid real dataset {path}: "
            + "; ".join(report["errors"])
        )

    identifier = spec.get("identifier_var")

    if identifier and identifier in df.columns:
        if (
            df[identifier].isna().any()
            or df[identifier].duplicated().any()
        ):
            raise ValueError(
                f"Identifier '{identifier}' must be "
                f"nonmissing and unique in {path}"
            )

    return df


def load_real_splits(config, spec):
    train = config.get("real_train_path")
    test = config.get("real_test_path")

    if bool(train) != bool(test):
        raise ValueError(
            "Specify both real_train_path and "
            "real_test_path, or neither"
        )

    if not train:
        directory = Path(config["real_data_path"])
        train = directory / "real_train.parquet"
        test = directory / "real_test.parquet"

    return load_real(train, spec), load_real(test, spec)


# ---------------------------------------------------------------------------
# Model fitting and evaluation
# ---------------------------------------------------------------------------

def evaluate_arm(train, test, spec, params, threshold):
    from xgboost import XGBClassifier

    features, types = validate_spec(spec)

    X_train = train[features].copy()
    X_test = test[features].copy()

    # XGBoost 2.x requires consistent category codes between fit and predict.
    # Learn each vocabulary from this arm's training data only.
    # Unseen test categories become missing values.
    category_levels = {}

    for column in types["categorical_vars"]:
        levels = sorted(
            X_train[column].dropna().unique().tolist()
        )
        category_levels[column] = levels

        dtype = pd.CategoricalDtype(categories=levels)
        X_train[column] = X_train[column].astype(dtype)
        X_test[column] = X_test[column].astype(dtype)

    y_train = train[spec["target_var"]].astype(int)

    y_test = test[spec["target_var"]].astype(int)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Store vocabularies with the pickled model for later reuse.
    model.utility_category_levels_ = category_levels

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model returned non-finite probabilities"
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=params["random_state"],
        scoring="roc_auc",
    )

    importance_df = pd.DataFrame(
        {
            "permutation_importance_mean": (
                importance.importances_mean
            ),
            "permutation_importance_std": (
                importance.importances_std
            ),
        },
        index=features,
    ).sort_values(
        "permutation_importance_mean",
        ascending=False,
    )

    result = {
        "n_positive": int(y_test.sum()),
        "n_negative": int(len(y_test) - y_test.sum()),
        "class_prevalence_test": float(y_test.mean()),
        "class_prevalence_train": float(y_train.mean()),
        "auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "threshold": float(threshold),
        "model_class": "XGBClassifier",
        "model_kwargs": dict(params),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "permutation_importances": (
            importance_df
            .rename_axis("feature")
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    return (
        result,
        model,
        probabilities,
        predictions,
        importance_df,
    )


def bootstrap_gap(y, real_prob, synth_prob, count):
    """Paired row-bootstrap intervals conditional on the fitted models."""
    rng = np.random.RandomState(42)
    auc_gaps = []
    ap_gaps = []

    for _ in range(count):
        indices = rng.randint(
            0, len(y), size=len(y)
        )
        labels = y[indices]

        if len(np.unique(labels)) < 2:
            continue

        auc_gaps.append(
            roc_auc_score(labels, real_prob[indices])
            - roc_auc_score(labels, synth_prob[indices])
        )
        ap_gaps.append(
            average_precision_score(
                labels, real_prob[indices]
            )
            - average_precision_score(
                labels, synth_prob[indices]
            )
        )

    if not auc_gaps:
        raise ValueError(
            "No valid bootstrap samples; increase n_bootstrap"
        )

    return (
        np.percentile(auc_gaps, [2.5, 97.5]),
        np.percentile(ap_gaps, [2.5, 97.5]),
    )


# ---------------------------------------------------------------------------
# Plots and summary
# ---------------------------------------------------------------------------

def save_figures(y, real_prob, synth_prob, results, directory):
    curves = [
        ("TRTR", real_prob, results["trtr"]),
        ("TSTR", synth_prob, results["tstr"]),
    ]

    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(6, 6))

        for name, probability, metrics in curves:
            if kind == "roc":
                x, z, _ = roc_curve(y, probability)
                label = (
                    f"{name} (AUC={metrics['auc']:.3f})"
                )
            else:
                precision, recall, _ = precision_recall_curve(
                    y, probability
                )
                x, z = recall, precision
                label = (
                    f"{name} (PR-AUC={metrics['pr_auc']:.3f})"
                )

            ax.plot(x, z, label=label)

        if kind == "roc":
            ax.plot(
                [0, 1], [0, 1],
                "k--", alpha=0.4,
            )
            ax.set(
                xlabel="False positive rate",
                ylabel="True positive rate",
                title="ROC — TRTR vs TSTR",
            )
        else:
            ax.axhline(
                y.mean(),
                color="k",
                ls="--",
                alpha=0.4,
                label=f"prevalence={y.mean():.3f}",
            )
            ax.set(
                xlabel="Recall",
                ylabel="Precision",
                title="Precision-Recall — TRTR vs TSTR",
            )

        ax.legend()
        fig.tight_layout()
        fig.savefig(
            directory / f"{kind}_trtr_vs_tstr.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        [0, 1], [0, 1],
        "k--", alpha=0.5,
        label="perfect calibration",
    )

    briers = {}

    for name, probability, _ in curves:
        score = float(
            brier_score_loss(y, probability)
        )
        briers[f"brier_{name.lower()}"] = score

        fraction, mean = calibration_curve(
            y,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(
            mean,
            fraction,
            marker="o",
            label=f"{name} (Brier={score:.3f})",
        )

    ax.set(
        xlabel="Mean predicted probability (per bin)",
        ylabel="Observed fraction positive (per bin)",
        title="Calibration — TSTR vs TRTR (on real test set)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    for extension in ("png", "eps"):
        fig.savefig(
            directory / f"calibration_tstr_vs_trtr.{extension}",
            bbox_inches="tight",
        )

    plt.close(fig)

    briers["brier_diff_tstr_minus_trtr"] = (
        briers["brier_tstr"] - briers["brier_trtr"]
    )

    return briers


def write_summary(path, result):
    real = result["trtr"]
    synth = result["tstr"]

    lines = [
        "UTILITY SUMMARY "
        "(TRTR = train real, TSTR = train synthetic; same test set)",
        "=" * 60,
        (
            "class prevalence (test): "
            f"{real['class_prevalence_test']:.3f} "
            f"(n_pos={real['n_positive']}, "
            f"n_neg={real['n_negative']})"
        ),
        "",
        (
            f"{'':12}{'AUC':>10}{'PR-AUC':>10}"
            f"{'F1':>10}{'recall':>10}"
        ),
    ]

    for name, arm in (
        ("TRTR", real),
        ("TSTR", synth),
    ):
        lines.append(
            f"{name:12}{arm['auc']:>10.3f}"
            f"{arm['pr_auc']:>10.3f}"
            f"{arm['f1']:>10.3f}"
            f"{arm['recall']:>10.3f}"
        )

    for label, gap, low, high in (
        (
            "AUC",
            "auc_gap_trtr_minus_tstr",
            "auc_ci_low",
            "auc_ci_high",
        ),
        (
            "PR-AUC",
            "pr_auc_gap_trtr_minus_tstr",
            "pr_auc_diff_ci_low",
            "pr_auc_diff_ci_high",
        ),
    ):
        lo, hi = result[low], result[high]
        verdict = (
            "gap not statistically distinguishable from 0"
            if lo <= 0 <= hi
            else "gap is distinguishable from 0"
        )

        lines.extend([
            "",
            (
                f"{label} gap (TRTR - TSTR): "
                f"{result[gap]:+.3f} "
                f"95% CI [{lo:+.3f}, {hi:+.3f}]"
            ),
            f"    -> {verdict}",
        ])

    lines.extend([
        "",
        f"threshold for F1/recall: {synth['threshold']}",
    ])

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_utility_eval(config, synthetic_data_path):
    spec = config["data_specs"]
    validate_spec(spec)

    output = Path(config["evaluation_output_path"])
    figures, metrics, artifacts = [
        output / name
        for name in ("figures", "metrics", "artifacts")
    ]

    for directory in (figures, metrics, artifacts):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(
        output / "experiment_config.json",
        config,
    )

    print("Loading and validating synthetic data...")
    synthetic, report = validate_table(
        load_table(synthetic_data_path),
        spec,
    )
    write_json(
        metrics / "schema_check.json",
        report,
    )

    for warning in report["warnings"]:
        print(f"  WARN: {warning}")

    if report["errors"]:
        raise ValueError(
            "Invalid synthetic submission:\n  - "
            + "\n  - ".join(report["errors"])
        )

    print(
        f"  schema check PASSED ({report['n_rows']} rows)"
    )

    if config.get("validate_only", False):
        return report

    settings = config.get(
        "downstream_model_kwargs", {}
    )

    if settings.get("model_class") not in (
        None, "XGBClassifier"
    ):
        raise ValueError(
            "This evaluator supports only XGBClassifier"
        )

    supplied = settings.get("hyperparameters")
    params = (
        dict(DEFAULT_PARAMS)
        if supplied is None
        else dict(supplied)
    )
    params.setdefault("random_state", 42)
    params.setdefault("enable_categorical", True)
    params.setdefault("tree_method", "hist")

    if params["random_state"] is None:
        raise ValueError(
            "Use a fixed random_state for evaluation"
        )

    if params["enable_categorical"] is not True:
        raise ValueError(
            "enable_categorical must be True"
        )

    if params["tree_method"] not in ("hist", "approx"):
        raise ValueError(
            "Use tree_method='hist' or 'approx' "
            "for categorical support"
        )

    threshold = settings.get("threshold", 0.5)
    count = settings.get("n_bootstrap", 100)

    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
    ):
        raise ValueError(
            "n_bootstrap must be a positive integer"
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    print("Loading real train/test splits...")
    real_train, real_test = load_real_splits(
        config, spec
    )
    results = {}
    probabilities = {}

    for name, train in (
        ("trtr", real_train),
        ("tstr", synthetic),
    ):
        print(f"Evaluating {name.upper()}...")

        result, model, probability, prediction, importance = (
            evaluate_arm(
                train,
                real_test,
                spec,
                params,
                threshold,
            )
        )

        model_path = artifacts / f"{name}_model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)

        result["fitted_model_path"] = str(model_path)

        importance.to_csv(
            metrics
            / f"downstream_model_permutation_importance_{name}.csv"
        )

        y = real_test[spec["target_var"]].to_numpy(dtype=int)

        pd.DataFrame(
            confusion_matrix(
                y, prediction, labels=[0, 1]
            ),
            index=["true_neg", "true_pos"],
            columns=["pred_neg", "pred_pos"],
        ).to_csv(
            metrics / f"confusion_{name}.csv"
        )

        results[name] = result
        probabilities[name] = probability

    auc_ci, ap_ci = bootstrap_gap(
        y,
        probabilities["trtr"],
        probabilities["tstr"],
        count,
    )

    results.update({
        "auc_ci_low": auc_ci[0],
        "auc_ci_high": auc_ci[1],
        "pr_auc_diff_ci_low": ap_ci[0],
        "pr_auc_diff_ci_high": ap_ci[1],
        "auc_gap_trtr_minus_tstr": (
            results["trtr"]["auc"]
            - results["tstr"]["auc"]
        ),
        "pr_auc_gap_trtr_minus_tstr": (
            results["trtr"]["pr_auc"]
            - results["tstr"]["pr_auc"]
        ),
    })

    results.update(
        save_figures(
            y,
            probabilities["trtr"],
            probabilities["tstr"],
            results,
            figures,
        )
    )

    write_json(
        metrics / "utility.json",
        results,
    )
    write_summary(
        metrics / "utility_summary.txt",
        results,
    )

    print(
        (metrics / "utility_summary.txt").read_text(
            encoding="utf-8"
        )
    )

    return results


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate synthetic-data utility with XGBoost."
        )
    )
    parser.add_argument(
        "experiment_config",
        help="Path to the experiment config JSON",
    )
    parser.add_argument(
        "--synth",
        help="Synthetic dataset; overrides synthetic_data_path",
    )
    parser.add_argument(
        "--output-path",
        help="Override evaluation_output_path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    with open(
        args.experiment_config,
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    synthetic_path = (
        args.synth
        or config.get("synthetic_data_path")
    )

    if not synthetic_path:
        parser.error(
            "Pass --synth or set synthetic_data_path "
            "in the config"
        )

    if args.output_path:
        config["evaluation_output_path"] = args.output_path

    if args.validate_only:
        config["validate_only"] = True

    run_utility_eval(config, synthetic_path)


if __name__ == "__main__":
    main()