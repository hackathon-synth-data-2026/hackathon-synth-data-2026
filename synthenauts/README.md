# Synthetic ICU data: CART + statistic matching + learned rules (v4)

We generate a synthetic version of the ICU table that scores **91.39 points** on the official
evaluator: best of 30 seeds, with a mean of 89.0 over all 30. It is fitted on `real_train` only.
**No synthetic patient breaks any rule that every real patient follows.** That covers 138 rules
learned from the data, for example "a patient is in at most one of CCU, CSRU and SICU".

The same pipeline also works on **any table**, through a command-line tool and an upload page.
They report each score honestly, next to the best score a synthetic table can reach on the same
test.

## Our submission

**`submissions/synthenauts_submission.parquet`**: one table with 2,880 synthetic patients and all
121 columns (every required feature, `recordid` and `mortality`), saved with `index=False`.

- **Checked.** It passes `--validate-only`. The full official evaluator, run with the unchanged
  official settings, gives the v4 scores below. Every `recordid` is new; none is a real one.
- **How it was chosen.** It is seed 12 of 30 seeds of the v4 pipeline. We picked it with the
  official evaluator, which scores against `real_test`. The mean over all 30 seeds is 89.0, and
  `submissions/leaderboard_v4.csv` lists every seed.
- **Rebuild it.** `uv run python synthenauts/synth.py --seed 12` gives the same table on our
  machine (see [Reproducibility](#reproducibility)).

## Results

Scored with the official evaluator:

| Version | Pipeline | Points | Utility | Fidelity | Privacy | Patients breaking a real-data rule |
| --- | --- | --- | --- | --- | --- | --- |
| **v4 (our submission)** | best of 30 seeds (mean 89.02) | **91.39** | 0.954 | 0.973 | 0.815 | **0** |
| v3 | best of 30 seeds (mean 89.51) | 91.07 | 0.945 | 0.972 | 0.815 | 293 (121 in two ICU units) |
| v2 | best of 50 seeds (mean 88.81) | 90.72 | 0.978 | 0.945 | 0.799 | 226 |

- **v4 is our submission.** It has the highest score, and every patient is clinically possible.
- Only the v4 table is included. `leaderboard_v4.csv`, `leaderboard_v3.csv` and
  `leaderboard_v2.csv` in `submissions/` list every seed of each version.
- The scoring caps at about **94.4** for any method: utility 1.0, fidelity 1.0, and privacy about
  0.83. Privacy cannot pass about 0.83 because 9% of real patients have too few demographics
  recorded for the distance check.

## Quick start

One-time setup from the repo root (Windows, macOS or Linux):

```bash
pip install uv            # or the installer in the main README
uv sync --locked          # creates .venv with the evaluator's exact dependencies
```

- If `uv` is not on your PATH after `pip install`, use `python -m uv` in its place.
- The first sync downloads large packages. If it stops with a network timeout, set
  `UV_HTTP_TIMEOUT=600` and run it again.
- macOS needs `brew install libomp` for XGBoost.

**Make an ICU submission and score it** (about 2.5 minutes, then about 1 minute):

```bash
uv run python synthenauts/synth.py --seed 12 --out results/submission.parquet
uv run python -m hackathon_synth_data_2026.main synthenauts/eval_config.json --synth results/submission.parquet --output-path results/official
```

The official evaluator writes its scores, plots and PNG figures to `results/official/`. For quick
experiments, `uv run python synthenauts/score.py results/submission.parquet` gives the same points in
about 30 seconds, without plots.

**Open the upload page** for the ICU table or any other table (Streamlit is fetched on the fly;
the lockfile is not changed):

```bash
uv run --with streamlit --with openpyxl streamlit run synthenauts/app.py
```

**Use the general tool from the command line:**

```bash
uv run python synthenauts/anysynth.py my_table.csv --target outcome --out my_table_synthetic.csv
uv run python synthenauts/anysynth.py my_table.parquet          # no column to predict
```

## How the ICU generator works (`synth.py`)

1. **Sequential CART** (synthpop style). Columns are generated one at a time, in order of their
   importance for mortality. For each new row, a decision tree finds real patients who look alike
   so far (at least 10) and copies a value from one of them. Missing values are modelled the same
   way. Every value is a real value; the combinations are new.
2. **Demographic jitter.** Small noise (0.25 × IQR) on age, height and weight, so exact real
   combinations of these quasi-identifiers are not reused.
3. **Correlation restoration.** CART weakens rank correlations by about 13%. Values are
   reordered within each column (Iman-Conover with a minimal-change map) until the correlation
   matrix matches real_train again. Each column keeps its own values and missing cells.
4. **Marginal matching.** Each column is mapped, keeping ranks, onto real_train's distribution.
5. **Clinical repair.** Lowest ≤ first, last, median ≤ highest for every vital sign; diastolic ≤
   systolic; GCS between 3 and 15.
6. **Statistic matching** (`statmatch.py`). 2 million attempts to swap blocks of columns between
   two random rows, each kept only if it brings the correlations and mutual information closer to
   real_train. A block is everything that must stay together: all summaries of a vital sign, a
   lab's first/last pair, or columns tied by a rule.
7. **Learned-rule repair** (new in v4). Every rule that holds in all real rows is enforced; see
   [Data quality](#data-quality-rules-every-patient-follows).
8. **Mortality relabelling.** Labels are redrawn from an XGBoost risk model fitted on real_train,
   at temperature 0.3 (less label noise than real data), with the real death rate.

**What each step adds** (official evaluator, mean over seeds; 3 seeds per row unless noted):

| Pipeline | Points | Utility | Fidelity | Privacy |
| --- | --- | --- | --- | --- |
| CART + repair (first version) | 80.7 | 0.76 | 0.859 | 0.807 |
| + mortality relabelling, temperature 0.3 | 84.8 to 85.5 | 0.88 to 0.90 | 0.859 | 0.807 |
| + marginal matching | 86.4 | 0.92 | 0.863 | 0.805 |
| + correlation restoration, 10 passes | 87.6 | 0.88 | 0.948 | 0.799 |
| + demographic jitter, 0.25 IQR (v2, 50 seeds) | 88.8 | 0.90 | 0.946 | 0.811 |
| + statistic matching, 2M swaps (v3, 30 seeds) | 89.5 | 0.90 | 0.973 | 0.809 |
| + learned-rule repair, tied columns kept together (v4, 30 seeds) | 89.0 | 0.89 | 0.972 | 0.806 |

Utility moves by about ±0.03 between seeds, so differences in that column are mostly noise.
Fidelity and privacy are stable. v4 gives up about 0.5 points of mean score for clinically
possible patients, and its best seed is still the highest.

**How settings were chosen:**
- **Utility settings** (temperature, risk model, leaf sizes) were picked on 5-fold splits of
  real_train, never on real_test.
- **Fidelity** only compares against real_train, so it was tuned on the official metric.
- **Privacy** uses real_test as its control set; the jitter level was confirmed on internal folds.

**Tried without gain:**
- ensembles or bagging for the risk model
- twice as many synthetic rows
- a higher synthetic death rate
- leaf size 5 everywhere (privacy drops)
- smaller leaves for the top 15 predictors
- keeping the top predictors out of the correlation step
- jitter of 0.6 IQR (privacy drops)

## Data quality: rules every patient follows

The official evaluator checks a few clinical rules, but not all of them. We learn every rule that
holds in **all** real rows and would rarely hold by chance: a rule is kept only if unrelated
columns would satisfy it in every row with probability below 0.1%. There are two kinds:

- **Order rules (A ≤ B):** 95 on the ICU table, for example HR_lowest ≤ HR_first and MAP ≤
  systolic pressure. These rediscover our hand-written clinical repair, and add a few more.
- **Category rules (if A = a then …):** 43 on the ICU table, for example "if CCU then not CSRU"
  (the unit flags are mutually exclusive), or "not ventilated in the last 8 hours → ventilation
  duration ≤ 2340".

**What we found in v3:**

| | Real data | v3 (seed 10) | v4 (our submission) |
| --- | --- | --- | --- |
| Patients breaking a real-data rule | 0 | 293 (10%) | 0 |
| Patients in two ICU units at once | 0 | 121 | 0 |
| Share in CCU / CSRU / SICU / no unit | 14.1 / 21.1 / 28.3 / 36.5% | 14.1 / 21.1 / 28.3 / 40.7% | 14.8 / 21.9 / 28.1 / 35.2% |

**How v4 fixes it:**
- **The rules are enforced at the end.** Each rule is enforced after every step that moves
  values. A patient in two units keeps one of them at random, in proportion to how common each
  unit is.
- **Tied columns stay together.** CART draws columns tied by a rule (the three unit flags)
  consistently. The polishing steps then keep them together instead of moving each one on its
  own. That stops the conflicts at the source and keeps the unit shares within about 1 point of
  the real data.

## The general tool and upload page (`anysynth.py`, `app.py`)

The same pipeline, with nothing written for the ICU data. It learns from the data:

- **Column types.** Numbers, dates, true/false and categories are detected. Identifier and
  free-text columns get fresh values and are never copied. Rare categories are pooled.
- **Rules.** Order and category rules, as above. On the ICU table it finds all 138 by itself. On
  the Kaggle stroke data it finds 21, for example "if work_type = children then age ≤ 16".
- **Swap blocks.** Columns linked by a rule, or always recorded together, move as one block.
- **The column to predict.** It can be yes/no, a category, a number, or nothing at all.

**Using the page:**
- **Sidebar.**
  - Upload a CSV, TSV, Parquet, Excel or JSON file, or press *Use the ICU example table*.
  - Pick the column to predict (optional).
  - Press *Generate*.
  - *Options*: seed, number of rows, share held out for the report, identifier columns,
    quasi-identifiers to jitter, fast mode, competition mode, and *Ignore saved results*.
- **Progress bar.** It shows the current stage and the estimated time left.
- **Scores.** Four tiles (overall, utility, fidelity, privacy) and a chart that shows each score
  next to the best achievable score on that test.
- **Tabs:**

  | Tab | Contents |
  | --- | --- |
  | Synthetic data | The output, plus CSV and Parquet downloads |
  | Compare columns | Each column's real vs synthetic distribution, and a similarity score per column, worst first |
  | Relationships | A heatmap of which links between columns changed, and a scatter of any two columns side by side |
  | Rules learned | Both rule types, with how many rows needed fixing |
  | Column types | How each column was understood |
  | Report | The full report as text and JSON |

**Reading the scores honestly.** The page is built so a high number cannot mislead:
- **Scores use unseen rows.** 20% of rows are held out, and the generator never sees them.
  Fidelity, utility and privacy are measured on them.
- **Every score has a "best achievable" reference.** It is the most a synthetic table can honestly
  score on that test:
  - utility 1.000: as useful as real data
  - fidelity: what a fresh sample of real rows scores against the held-out rows
  - privacy 1.000: no sign of copying

  Compare with that, not with 100. On this test the ICU pipeline scores 84.0, against a best
  achievable 90.7.

  It is not called "real data" because real records offer no privacy at all.
- **Utility is not capped.** It compares a model trained on synthetic data with one trained on
  real data, e.g. "performs at 99% of one trained on real data", with a 90% range. Above 100%
  means the outcome looks easier to predict than it is, and is penalised like a shortfall.
- **Privacy takes the worse of two tests.** It is a nearest-row test on all columns and on
  quasi-identifiers (age, gender, …), like the official evaluator's demographics group.
  - **What 1.000 means:** no sign of copying the training rows, not that the data is anonymous.
  - **What it covers:** copying and near-copying, not every possible attack.
- **Warnings and notes:**
  - *Warnings* flag small test sets, few positive cases, competition mode, and fitting without
    held-out rows.
  - A neutral *note* explains tables whose information sits mostly in single columns.

**Faithful by default.** The column to predict keeps the real amount of noise. *Competition mode*
(`--competition`) redraws it with less noise, as the ICU submission does. That raises
hackathon-style scores, but models trained on the data then look more accurate than they would
on real data.

**Saved results:**
- **Reuse.** Every finished result is saved, and the same file with the same settings loads
  instantly.
- **Invalidation.** A result is reused only while the file, every setting and the code are
  unchanged. Editing comments does not count; a real code change does, and the page then says
  why it runs again.
- **Locations:**
  - The ICU example with default settings shows our submission file and its official scores, so
    it opens at once. Any other setting makes a new table and scores it with the page's test.
  - Results made on your machine go to `results/app_cache/`.
  - On the command line, `--cache DIR` does the same.

**Command-line notes:**
- **Formats.** It reads and writes CSV, TSV, Parquet, Excel and JSON; the output format follows
  `--out`. Excel needs `uv run --with openpyxl python synthenauts/anysynth.py ...`.
- **Report.** A report is saved next to the output, as `<out>.report.json`.
- **Switches.** `--holdout 0` fits on every row (the report then has fidelity only), and `--help`
  lists the rest.

**Tested on:**
- **The ICU table.** In competition mode, without a holdout, seeds 0–2 score 89.97, 89.92 and
  90.81 on the official evaluator.
- **The Kaggle stroke data.** Ours 95.3, real data 96.8.
- **Other tables:**
  - a table with dates (admission ≤ discharge is learned)
  - multi-class and numeric targets
  - a 2,000-row table whose generating formula the category rules reveal
  - a 135-row table

## Independent privacy check with Anonymeter (`anonymeter_eval.py`)

[Anonymeter](https://pypi.org/project/anonymeter/) runs three attacks that only see the synthetic
table: **singling out** (find a combination of three values shared by exactly one patient),
**linkability** (link two views of a patient) and **inference** (guess a hidden column).
Each attack is repeated on real_test patients as a control; the risk is the success the control
cannot explain (0 = no leak, 1 = full leak). It is a second opinion, not part of the official score.

Anonymeter pins `numpy<1.27` and `pyarrow<22`, which conflicts with the project, so it runs in a
throw-away environment and the lockfile stays untouched:

```bash
uv run --no-project --python 3.12 --with anonymeter --with pandas --with pyarrow \
    python synthenauts/anonymeter_eval.py synthenauts/submissions/synthenauts_submission.parquet
```

It takes about 20 seconds and writes `results/anonymeter_<file>.json`. Options: `--n-attacks`
(capped at the control size), `--n-cols`, `--secrets`, `--min-observed`, `--seed`.

- **Missing values.** Anonymeter's distance is undefined when only one of two values is missing.
  The attacker therefore uses only the 45 features observed in at least 95% of real_train, and
  only patients complete on them (2,709 of 2,880 train, 681 of 720 control).
- **Sanity check.** Giving it `real_train` as the "synthetic" table gives a risk of 0.98 to 1.0
  on every attack, so the attacks do work.
- **Result for our submission:** linkability 0.002 and inference 0.000 (mortality, gender): no
  detectable leak. **Singling out 0.23** (95% CI 0.19 to 0.27; 0.22 to 0.24 over seeds): the
  attacker finds a unique patient with 26% success against 4% for the control. v2 and v3 give 0.21
  and 0.24. The official privacy metric does not measure this, so it does not show in the 0.815.
- **Reading the number.** Singling out here is a weak, query-based test on three columns of a
  table whose values are all real values. It flags a real pattern worth knowing about, but it is
  not proof of re-identification.

## Reproducibility

- **Same machine.** A seed gives the same file every time: `synth.py --seed 12` rebuilds
  `submissions/synthenauts_submission.parquet` exactly. `synth.py` makes v4 by default, and
  `--no-learned-rules` gives v3 (our v3 pick was seed 10).
- **Other machines.** On a different OS or CPU, tiny floating-point differences can change the
  output a little, so the file in `submissions/` is the reference.
- **Best of N.** `uv run python synthenauts/best_of_n.py --seeds 0-29` scores every seed and keeps one
  leaderboard per version (`results/best_of_n_v4.csv`). It skips seeds already scored, and copies
  the best one to `results/submission.parquet`; `--pick current` restricts the pick to the
  current version.
- **Evaluator config.** `synthenauts/eval_config.json` is a copy of the official config with relative
  paths. The official one points at `/home/irene/...` and does not run elsewhere. Settings,
  features and weights are unchanged.

## Files

| File | Purpose |
| --- | --- |
| `synth.py` | The ICU generator (v4 by default); `--help` lists switches for each step |
| `statmatch.py` | Statistic matching: block swaps between rows (numba) |
| `anysynth.py` | The general tool: learns types, rules and swap blocks from any table, and writes the honest report |
| `app.py` | The upload page (Streamlit) |
| `best_of_n.py` | Scores many seeds and picks the submission |
| `score.py` | Fast scorer built on the official evaluator code; reproduces its points exactly |
| `experiments.py`, `configs.json` | Runs configs: `cv` (utility on internal folds of real_train) or `official` |
| `cv_privacy.py` | Privacy check on internal folds (held-out real_train rows as the control set) |
| `anonymeter_eval.py` | Independent privacy attacks with Anonymeter (singling out, linkability, inference) |
| `tune_label_model.py` | CV comparison of risk models for relabelling |
| `train_importance.py`, `train_imp.json` | Mortality importance ranking on real_train (sets the column order) |
| `submissions/` | Our submission table, and the leaderboards of every version |
| `cart_synth.py`, `repair.py` | Our first version, kept for comparison |
