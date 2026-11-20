# Synthetic Data Evaluation

This starterpack evaluates synthetic ICU summary data across three dimensions:

- **Utility:** preservation of mortality-prediction performance.
- **Fidelity:** preservation of statistical properties.
- **Privacy:** detected copying and training-membership distinguishability.

For dataset provenance, variable definitions, units, and preprocessing, see the [data README](data/README.md).

## 1. Quick start

From the starterpack root, evaluate the supplied example:

```bash
uv run --locked python -m hackathon_synth_data_2026.main src/hackathon_synth_data_2026/eval_config.json --synth data/example_synthetic_data.parquet
```

Run submission-schema validation without calculating metrics:

```bash
uv run --locked python -m hackathon_synth_data_2026.main src/hackathon_synth_data_2026/eval_config.json --synth data/example_synthetic_data.parquet --validate-only
```

The evaluator prints a summary and saves detailed results in the configured output directory.


## 2. Installation

### Install uv

The evaluator uses [uv](https://docs.astral.sh/uv/) to manage Python and install dependencies.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal if necessary, then verify installation:

```bash
uv --version
```

Alternative methods are available in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Obtain the starterpack

Clone the repository or download and extract the starterpack ZIP. Open a terminal in its root directory.

Python 3.12 or newer is required. Under uv's default configuration, a compatible Python version can be downloaded automatically if needed.

The first evaluation run requires internet access and may take several minutes while dependencies are installed. No manual virtual-environment activation or separate `pip install` step is needed.

Use `uv run --locked` so evaluation uses the supplied dependency lockfile. Direct execution with `python` or `python3` does not process the inline dependency declarations in `main.py`.

## 3. Folder structure

```text
starterpack/
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── hackathon_synth_data_2026/
│       ├── __init__.py
│       ├── main.py
│       ├── eval_config.json
│       ├── validation.py
│       ├── scoring.py
│       ├── utility.py
│       ├── fidelity.py
│       └── privacy.py
└── data/
    ├── README.md
    ├── real_train.parquet
    ├── real_test.parquet
    └── example_synthetic_data.parquet
```

The evaluator uses:

- `real_train` as the real training reference.
- `real_test` as the utility test set and privacy control set.
- Your synthetic table as the submission.

The fidelity evaluator compares your synthetic table with `real_train`.

## 4. Submission requirements

### Table format

Submit one complete table containing the required features, identifier, and target.

Do **not** submit features (`X`) and target labels (`y`) as separate files.

Parquet is strongly recommended because it preserves column types. CSV is also supported through the evaluator's shared loading and normalization.

Example structure:

```text
recordid | Age | Gender | GCS_first | ... | mortality
```

Each row represents one synthetic record. Synthetic rows do not need to correspond to individual real rows.

### Target column

- The target name must match `data_specs.target_var` in `config.json`.
- For this dataset, the target is `mortality`.
- Labels must contain only:
  - `0`: survival.
  - `1`: in-hospital death.
- Missing target values are not allowed.
- Both classes must be present for utility evaluation.

No target mapping is performed.

### Features and types

Follow the published shared schema:

- Include every required submission column.
- Use unique column names.
- Numerical features must contain numerical values or permitted missing values.
- Boolean features must use accepted boolean or binary representations.
- Categorical features must follow any configured allowed-value rules.
- Infinite numerical values are not allowed.
- Entirely missing feature columns are not accepted.
- Missing values are permitted only where allowed by the schema.

The shared normalizer accepts boolean values, `0`/`1`, and their documented string representations. It does not impute missing values, clip measurements, or repair invalid generated values.

Do not add dataframe index columns to the submission. Save files with `index=False`.

### Identifier

The identifier name is specified by `data_specs.identifier_var`, normally `recordid`.

Under the shared submission rules:

- Include the identifier column.
- Use nonmissing, nonempty, unique synthetic identifiers.
- Generate synthetic identifiers rather than reusing real identifiers.

Identifiers are excluded from predictive features and statistical profile comparisons. The privacy evaluator separately reports identifier collisions.

### Fixed schema and feature sets

All participants use the same required submission schema.

The utility evaluator uses the fixed `input_vars` defined by the organizers. Utility uses input_vars to predict mortality. Fidelity evaluates the features and mortality label. Privacy evaluates full profiles and selected feature subsets, with identifiers checked separately.

Participants may choose their generation approach, but must not change the official evaluation feature sets or omit required columns.

## 5. Preparing and validating a submission

### Optional normalization in a notebook

From a notebook in the starterpack root:

```python
import json
from pathlib import Path

from evaluation.validation import normalize_submission

with Path("evaluation/config.json").open(encoding="utf-8") as handle:
    config = json.load(handle)

synthetic = normalize_submission(
    synthetic,
    config["data_specs"],
)

synthetic.to_parquet(
    "data/submission.parquet",
    index=False,
)
```

The notebook environment needs NumPy and pandas installed. The inline dependencies in `main.py` configure CLI execution; they do not automatically configure a notebook kernel.

Normalizing before export is useful for finding problems early, but it does not replace evaluator-side validation.

### Validate before a full run

```bash
uv run --locked evaluation/main.py evaluation/config.json --synth data/submission.parquet --validate-only
```

Currently, this option runs the utility evaluator's submission-schema checks and skips metric evaluation.

It does **not** guarantee that every fidelity- or privacy-specific requirement is satisfied. Run the complete evaluator before submitting.

## 6. Running the evaluation

### Use the default configuration

By default, `main.py` reads `config.json` from the same directory:

```bash
uv run --locked evaluation/main.py
```

This also uses `synthetic_data_path` from the configuration.

### Specify a configuration

```bash
uv run --locked evaluation/main.py evaluation/config.json
```

### Specify a submission

```bash
uv run --locked evaluation/main.py evaluation/config.json --synth data/submission.parquet
```

### Choose an output directory

```bash
uv run --locked evaluation/main.py evaluation/config.json --synth data/submission.parquet --output-path results/my-run
```

Use a separate output directory for each experiment if you want to preserve earlier reports.

### Show help

```bash
uv run evaluation/main.py --help
```

### Path resolution

- Relative paths inside `config.json` resolve relative to that configuration file.
- Paths supplied through `--synth` and `--output-path` resolve relative to your current working directory.
- CLI arguments override the corresponding configuration paths.
- Setting `validate_only` to `true` in the configuration also skips metric evaluation.

You may use a local configuration copy to change file locations. Official metric settings, feature sets, seeds, and scoring weights are fixed by the organizers.

## 7. Understanding the scores

Higher component scores and final points are better.

### Utility

Utility compares two mortality classifiers using the same XGBoost configuration and the same real test rows:

- **TRTR:** train on real training data, test on real test data.
- **TSTR:** train on synthetic data, test on real test data.

The ranking metric is average precision, stored under the historical output key `pr_auc`.

Utility points measure capped retention of real-trained performance:

```python
utility_score = min(1.0, ap_tstr / ap_trtr)
```

The real-trained reference AP must be positive.

| TRTR AP | TSTR AP | Utility score |
|---:|---:|---:|
| 0.40 | 0.30 | 0.75 |
| 0.40 | 0.40 | 1.00 |
| 0.40 | 0.44 | 1.00 |

Matching or exceeding the real-trained reference earns full utility credit. Improvements remain visible in the raw results but receive no additional ranking points.

AUROC, Brier score, threshold-dependent classification metrics, and TRTR–TSTR comparisons are additional diagnostics; they do not separately contribute to utility points.

The output key `pr_auc` contains **average precision**, not trapezoidal area under the precision–recall curve.

### Fidelity

Fidelity combines four components:

| Component | Interpretation |
|---|---|
| Marginal distributions | Do individual columns have similar distributions? |
| Correlation preservation | Are numeric rank relationships preserved? |
| Dependency preservation | Are broader pairwise associations preserved? |
| Missingness preservation | Are per-column missing-value rates preserved? |

The marginal component uses:

- `1 − KS statistic` for numeric columns.
- `1 − total variation distance` for categorical columns.

Correlation and dependency scores compare synthetic-data errors with an independently shuffled real-data baseline. These components are clipped to the range 0–1.

The fidelity score is the weighted average of the four components. The official component weights are specified in:

```text
fidelity_kwargs.score_weights
```

Additional diagnostics, such as KL divergence, Wasserstein distance, Pearson correlation, and Cramér's V differences, do not introduce extra ranking terms.

A high fidelity score does not establish that the datasets are statistically indistinguishable.

### Privacy

The privacy evaluator examines:

| Indicator | Interpretation |
|---|---|
| Exact-copy rate | How many synthetic profiles exactly match training profiles? |
| Near-copy indicator | Are synthetic profiles unusually close to training records compared with held-out controls? |
| Membership indicator | Do distances to synthetic data distinguish training records from held-out controls? |

The checks use a mixed numeric/categorical distance and evaluate several fixed feature groups and random seeds.

The score is based on the largest aggregated risk:

```text
privacy_score = 1 − largest aggregated risk indicator
```

When missing data prevent comparisons, the evaluator uses conservative coverage bounds rather than automatically treating unknown outcomes as safe.

A high score means low risk detected by these checks. It is not a formal privacy guarantee.

Review the score basis and completion status alongside the numerical score. Technical error-fallback scores must not be treated as valid measured privacy results.

### Final ranking points

The component scores are combined as:

```python
final_points = 100 * (
    w_utility * utility_score
    + w_fidelity * fidelity_score
    + w_privacy * privacy_score
)
```

Weights are nonnegative and sum to 1. With capped utility and valid fidelity/privacy scores, final points range from 0 to 100.

The current scoring helper defaults to equal weights unless other weights are supplied. The `ranking` section of `evaluation.json` records the actual weights and each component's contribution.

Ranking uses unrounded points. Displayed values may be rounded.

This weighted mean allows stronger performance in one dimension to compensate for weaker performance in another, subject to any separately published eligibility requirements.

## 8. Output files

Results are saved under `evaluation_output_path`:

```text
<output>/
├── evaluation.json
├── utility/
│   ├── metrics/
│   │   ├── utility.json
│   │   └── ...
│   ├── figures/
│   └── artifacts/
├── fidelity/
│   └── metrics/
│       ├── fidelity.json
│       ├── fidelity_per_column.csv
│       ├── fidelity_mi_real.csv
│       └── fidelity_mi_synthetic.csv
└── privacy/
    └── metrics/
        ├── privacy.json
        ├── summary.csv
        └── seed_runs.csv
```

Start with **`evaluation.json`**. It contains the combined results and final scoring breakdown.

For troubleshooting:

- Utility reports and figures show predictive performance.
- `fidelity_per_column.csv` identifies columns with distribution differences.
- `privacy.json` contains risk indicators, coverage information, and failure reasons.
- `seed_runs.csv` provides privacy results by feature group and seed.

If evaluation fails before all stages finish, a combined report may not be produced. Do not use a combined report left over from an earlier run.

## 9. Troubleshooting

### Missing required columns

Check spelling and capitalization against the shared schema. Include the target in the same table as the features.

### Invalid target values

Use only `0` and `1`, with no missing labels. Both classes must be present for utility evaluation.

### CSV and Parquet behave differently

CSV readers infer types. Ensure normalization happens after loading and that the metric functions receive the normalized dataframes.

Parquet is the recommended submission format.

### Unexpected `Unnamed: 0` column

This is commonly a saved dataframe index. Export with:

```python
df.to_csv("submission.csv", index=False)
```

### CSV decoding error

Confirm that the file is actually CSV, rather than Parquet or Excel with a changed extension. Export CSV using UTF-8.

### Privacy evaluation is incomplete

Inspect the reasons in `privacy/metrics/privacy.json`. Possible causes include missing required feature-group columns, insufficient real records, incompatible schemas, or invalid real-data identifiers.

The real training and control sets must have unique, nonmissing, disjoint identifiers.

### Entirely missing or infinite columns

Entirely missing feature columns are not accepted. Numerical values must be finite where observed.

### Dependency-related errors

Use the supplied lockfile and the documented `uv run --locked` command. Include the full traceback when reporting a problem.

### Reporting a problem

Send the organizers:

- The command you ran.
- The complete traceback.
- Starterpack version or Git commit.
- Your operating system.
- A minimal reproducing example, where possible.

## 10. Reproducibility and official evaluation

Official evaluation uses fixed:

- Real training and test splits.
- Submission schema and preprocessing rules.
- Utility feature set and XGBoost configuration.
- Fidelity weights.
- Privacy settings, feature groups, budgets, and seeds.
- Final scoring weights.
- Dependency versions.

Do not modify these settings to improve your local ranking score. Organizers will evaluate submissions using the official configuration.

The same submission should be evaluated through the same documented environment and entry point.

## 11. Scope

This challenge studies the utility–fidelity–privacy tradeoff in synthetic ICU summary data.

The observation window underlying the aggregated measurements has not been verified. Utility results therefore assess mortality classification from the supplied summary table; they should not be interpreted as validated admission-time prediction or clinical deployment performance.

For dataset details and usage terms, see the data README.