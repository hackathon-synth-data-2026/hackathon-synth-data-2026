"""
Minimal similarity evaluation: real.csv vs synthetic.csv

Usage:
    python evaluate.py real.csv synthetic.csv
    python evaluate.py real.csv synthetic.csv --id patient_id --time charttime
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path

BINS = 20
EPS = 1e-10
warnings.filterwarnings("ignore")  # e.g. autocorr of constant sequences


def histograms(r, s):
    """Shared bins -> two probability vectors (works for numeric and categorical)."""
    if pd.api.types.is_numeric_dtype(r):
        edges = np.histogram_bin_edges(pd.concat([r, s]), bins=BINS)
        p = np.histogram(r, edges)[0].astype(float)
        q = np.histogram(s, edges)[0].astype(float)
    else:
        cats = sorted(set(r) | set(s), key=str)
        p = r.value_counts().reindex(cats, fill_value=0).values.astype(float)
        q = s.value_counts().reindex(cats, fill_value=0).values.astype(float)
    return p / p.sum(), q / q.sum()


def column_metrics(r, s):
    r, s = r.dropna(), s.dropna()
    p, q = histograms(r, s)
    out = {
        "TVD": 0.5 * np.abs(p - q).sum(),
        "KL": stats.entropy(p + EPS, q + EPS),
        "HistOverlap": np.minimum(p, q).sum(),  # 1 = identical
    }
    if pd.api.types.is_numeric_dtype(r):
        out["KS"] = stats.ks_2samp(r, s).statistic
        out["Wasserstein"] = stats.wasserstein_distance(r, s) / (r.std() + EPS)  # in units of real std
    else:
        counts = np.vstack([p * len(r), q * len(s)])
        counts = counts[:, counts.sum(0) > 0]
        out["Chi2_p"] = stats.chi2_contingency(counts)[1]  # high p = no detectable difference
    return out


def discretize(df):
    result = {}

    for name, column in df.items():
        try:
            if pd.api.types.is_numeric_dtype(column):
                # Convert nullable numeric/boolean values to floats:
                # missing values become np.nan.
                values = pd.Series(
                    column.to_numpy(
                        dtype=float,
                        na_value=np.nan,
                    ),
                    index=column.index,
                )
                result[name] = pd.qcut(
                    values,
                    BINS,
                    labels=False,
                    duplicates="drop",
                ).fillna(-1)
            else:
                # Preserve the original categorical treatment.
                result[name] = column.astype(str).fillna(-1)

        except Exception as exc:
            raise RuntimeError(
                f"Discretization failed for column {name!r}; "
                f"dtype={column.dtype}; "
                f"missing={column.isna().sum()}"
            ) from exc

    return pd.DataFrame(result, index=df.index)


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -(p * np.log(p)).sum()


def mi_matrix(df):
    """Pairwise normalized mutual information (arithmetic normalisation), fast via bincount."""
    d = discretize(df).apply(lambda c: pd.factorize(c)[0])
    X, cols = d.values, d.columns
    k = X.max(axis=0) + 1
    H = np.array([_entropy(np.bincount(X[:, i])) for i in range(X.shape[1])])
    M = np.eye(len(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            hij = _entropy(np.bincount(X[:, i] * k[j] + X[:, j]))
            denom = (H[i] + H[j]) / 2
            M[i, j] = M[j, i] = (H[i] + H[j] - hij) / denom if denom > 0 else 0.0
    return pd.DataFrame(M, index=cols, columns=cols)


def cramers_v(a, b):
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table)[0]
    return np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))


def matrix_diff(a, b):
    """Mean absolute difference of the off-diagonal entries (NaNs ignored)."""
    mask = ~np.eye(len(a), dtype=bool)
    return np.nanmean(np.abs(a.values - b.values)[mask])


def fidelity_score(real, syn, cols, num, cat, mi_real, mi_syn, weights=None, seed=0):
    """
    One fidelity number in [0, 1] (1 = indistinguishable from real, 0 = nothing preserved).
      marginals    1 - mean KS (numeric) / 1 - TVD (categorical)
      correlation  Spearman error, relative to a "no dependencies" baseline
      dependency   normalized-MI error, relative to the same baseline
      missingness  1 - mean |missing rate real - missing rate synthetic|
    Baseline = real data with every column shuffled independently: perfect marginals, zero
    dependencies. Matching it scores 0 on correlation/dependency, matching real data scores 1.
    """
    w = weights or {"marginals": 0.4, "correlation": 0.25, "dependency": 0.15, "missingness": 0.2}
    rng = np.random.default_rng(seed)
    base = real[cols].apply(lambda c: pd.Series(rng.permutation(c.values), index=c.index, dtype=c.dtype,))

    def relative(err, err_base):
        return float(np.clip(1 - err / err_base, 0, 1)) if err_base > 0 else 1.0

    marg = [1 - stats.ks_2samp(real[c].dropna(), syn[c].dropna()).statistic for c in num
            if real[c].notna().any() and syn[c].notna().any()]
    marg += [1 - 0.5 * np.abs(np.subtract(*histograms(real[c].dropna(), syn[c].dropna()))).sum() for c in cat]

    comp = {"marginals": float(np.mean(marg))}
    if len(num) > 1:
        cr = real[num].corr("spearman")
        comp["correlation"] = relative(matrix_diff(cr, syn[num].corr("spearman")),
                                       matrix_diff(cr, base[num].corr("spearman")))
    else:
        comp["correlation"] = 1.0
    comp["dependency"] = relative(matrix_diff(mi_real, mi_syn), matrix_diff(mi_real, mi_matrix(base)))
    comp["missingness"] = float(1 - np.mean(np.abs(real[cols].isna().mean() - syn[cols].isna().mean())))
    score = sum(w[k] * comp[k] for k in comp) / sum(w.values())
    return score, comp


def structural(real, syn, id_col, time_col, num_cols):
    """Time series: sequence lengths, step-to-step changes, lag-1 autocorrelation."""
    out = {}
    lens_r, lens_s = real.groupby(id_col).size(), syn.groupby(id_col).size()
    out["SeqLength_KS"] = stats.ks_2samp(lens_r, lens_s).statistic
    for c in num_cols:
        dr = real.sort_values(time_col).groupby(id_col)[c].diff().dropna()
        ds = syn.sort_values(time_col).groupby(id_col)[c].diff().dropna()
        ar = real.sort_values(time_col).groupby(id_col)[c].apply(lambda x: x.autocorr()).mean()
        as_ = syn.sort_values(time_col).groupby(id_col)[c].apply(lambda x: x.autocorr()).mean()
        out[f"{c}: delta_KS"] = stats.ks_2samp(dr, ds).statistic if len(dr) and len(ds) else np.nan
        out[f"{c}: autocorr real/syn"] = f"{ar:.2f} / {as_:.2f}"
    return out


def load_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)

    separator = "\t" if suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=separator)


def evaluate_fidelity(
    real,
    syn,
    *,
    target_col="In-hospital_death",
    id_col="recordid",
    time_col=None,
    split_col=None,
    fidelity_kwargs=None,
):
    """Evaluate complete tables without modifying the caller's dataframes."""
    real, syn = real.copy(), syn.copy()

    if not real.columns.is_unique or not syn.columns.is_unique:
        raise ValueError("Column names must be unique")

    if real.empty or syn.empty:
        raise ValueError("Both tables must contain rows")

    skip = {id_col, time_col, split_col} - {None}
    cols = [column for column in real.columns if column not in skip]

    if target_col in skip:
        raise ValueError(
            "The target cannot also be an ID, time, or split column"
        )

    missing = [column for column in cols if column not in syn.columns]
    if missing:
        raise ValueError(
            f"Synthetic table is missing columns: {missing}"
        )

    if target_col not in real or target_col not in syn:
        raise ValueError(
            f"Both tables must contain target column {target_col!r}"
        )

    if len(cols) < 2:
        raise ValueError(
            "Fidelity evaluation requires at least two evaluated columns"
        )

    for label, df in (("real", real), ("synthetic", syn)):
        target = df[target_col]

        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(
                f"{label} target must contain nonmissing 0/1 values"
            )

        empty_columns = [
            column
            for column in cols
            if df[column].notna().sum() == 0
        ]
        if empty_columns:
            raise ValueError(
                f"{label} has entirely missing columns: {empty_columns}"
            )

    if time_col and not id_col:
        raise ValueError("Structural metrics require an ID column")

    if time_col:
        for label, df in (("real", real), ("synthetic", syn)):
            for column in (id_col, time_col):
                if column not in df:
                    raise ValueError(
                        f"{label} is missing structural column {column!r}"
                    )

    # Preserve the original rule: real-data dtypes determine metric selection.
    # Do not apply utility-specific categorical encoding or imputation here.
    num = [
        column
        for column in cols
        if pd.api.types.is_numeric_dtype(real[column])
    ]
    cat = [column for column in cols if column not in num]

    for column in cols:
        real_numeric = pd.api.types.is_numeric_dtype(real[column])
        syn_numeric = pd.api.types.is_numeric_dtype(syn[column])

        if real_numeric != syn_numeric:
            raise ValueError(
                f"Numeric/categorical dtype mismatch for {column!r}"
            )

    for label, df in (("real", real), ("synthetic", syn)):
        for column in num:
            values = df[column].to_numpy(
                dtype=float,
                na_value=np.nan,
            )
            if np.isinf(values).any():
                raise ValueError(
                    f"{label} column {column!r} contains infinity"
                )

    extras = [
        column
        for column in syn.columns
        if column not in real.columns and column not in skip
    ]
    if extras:
        print(f"WARN: extra synthetic columns ignored: {extras}")

    # Per-column metrics: same calculations as the original script.
    per_col = {}
    for column in cols:
        try:
            per_col[column] = column_metrics(
                real[column],
                syn[column],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Fidelity calculation failed for column {column!r}"
            ) from exc

    per_col = pd.DataFrame(per_col).T

    print("\n=== Per-column similarity ===")
    print(per_col.round(3).to_string())

    correlations = {}

    print(
        "\n=== Correlation preservation "
        "(mean abs diff, 0 = perfect) ==="
    )

    if len(num) > 1:
        for method in ("pearson", "spearman"):
            difference = float(
                matrix_diff(
                    real[num].corr(method),
                    syn[num].corr(method),
                )
            )
            correlations[method] = difference
            print(f"{method:10s}: {difference:.3f}")

    if len(cat) > 1:
        def cv(df):
            return pd.DataFrame(
                [
                    [cramers_v(df[a], df[b]) for b in cat]
                    for a in cat
                ],
                index=cat,
                columns=cat,
            )

        difference = float(matrix_diff(cv(real), cv(syn)))
        correlations["cramers_v"] = difference
        print(f"{'cramers_v':10s}: {difference:.3f}")

    mi_real = mi_matrix(real[cols])
    mi_syn = mi_matrix(syn[cols])
    mi_difference = float(matrix_diff(mi_real, mi_syn))

    print(
        "\n=== Mutual information "
        "(mean abs diff of normalized MI, 0 = perfect) ==="
    )
    print(f"{mi_difference:.3f}")

    structure = {}

    if id_col and time_col:
        structure = structural(
            real,
            syn,
            id_col,
            time_col,
            num,
        )

        print("\n=== Structural similarity (time series) ===")

        for key, value in structure.items():
            displayed = (
                value if isinstance(value, str) else round(value, 3)
            )
            print(f"{key:30s}: {displayed}")

    score, components = fidelity_score(
        real,
        syn,
        cols,
        num,
        cat,
        mi_real,
        mi_syn,
        weights=fidelity_kwargs["score_weights"]
    )

    print("\n=== Fidelity score ===")
    for key, value in components.items():
        print(f"  {key:12s}: {value:.3f}")
    print(f"  {'FIDELITY':12s}: {score:.3f}")

    return {
        "score": float(score),
        "components": components,
        "columns": cols,
        "per_column": per_col,
        "correlation_difference": correlations,
        "mi_difference": mi_difference,
        "mi_real": mi_real,
        "mi_synthetic": mi_syn,
        "structural": structure,
    }


def run_fidelity_eval(
    config,
    synthetic_data_path,
    *,
    config_dir=None,
):
    """Use real_train as the reference and save fidelity reports."""
    base = (
        Path(config_dir)
        if config_dir is not None
        else Path.cwd()
    )

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else base / path

    spec = config["data_specs"]

    if config.get("real_train_path"):
        real_path = resolve(config["real_train_path"])
    else:
        real_path = (
            resolve(config["real_data_path"])
            / "real_train.parquet"
        )

    from .validation import normalize_submission
    real = normalize_submission(
        load_table(real_path), spec)
    synthetic = normalize_submission(
        load_table(synthetic_data_path), spec)

    result = evaluate_fidelity(
        real,
        synthetic,
        target_col=spec["target_var"],
        id_col=spec.get("identifier_var", "recordid"),
        time_col=spec.get("time_var"),
        split_col=spec.get("split_var"),
        fidelity_kwargs=config["fidelity_kwargs"]
    )

    directory = (
        resolve(config["evaluation_output_path"])
        / "fidelity" / "metrics"
    )
    directory.mkdir(parents=True, exist_ok=True)

    result["per_column"].to_csv(
        directory / "fidelity_per_column.csv"
    )
    result["mi_real"].to_csv(
        directory / "fidelity_mi_real.csv"
    )
    result["mi_synthetic"].to_csv(
        directory / "fidelity_mi_synthetic.csv"
    )

    def json_safe(value):
        if isinstance(value, dict):
            return {
                key: json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    summary = {
        key: value
        for key, value in result.items()
        if key not in (
            "per_column",
            "mi_real",
            "mi_synthetic",
        )
    }

    # Preserve undefined results as null rather than inventing a score.
    summary["undefined_components"] = [
        key
        for key, value in result["components"].items()
        if not np.isfinite(value)
    ]

    with (directory / "fidelity.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            allow_nan=False,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fidelity evaluation of complete tables"
    )
    parser.add_argument(
        "experiment_config",
        help="Path to config.json",
    )
    parser.add_argument(
        "--synth",
        help=(
            "Synthetic CSV/Parquet path, "
            "relative to the working directory"
        ),
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Output directory, "
            "relative to the working directory"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.experiment_config).resolve()

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if args.synth:
        synthetic_path = Path(args.synth).resolve()
    elif config.get("synthetic_data_path"):
        synthetic_path = Path(config["synthetic_data_path"])

        if not synthetic_path.is_absolute():
            synthetic_path = (
                config_path.parent / synthetic_path
            )
    else:
        parser.error(
            "Pass --synth or set synthetic_data_path in config.json"
        )

    if args.output_path:
        config["evaluation_output_path"] = str(
            Path(args.output_path).resolve()
        )

    run_fidelity_eval(
        config,
        synthetic_path,
        config_dir=config_path.parent,
    )


if __name__ == "__main__":
    main()