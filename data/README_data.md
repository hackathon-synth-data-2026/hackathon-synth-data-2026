# ICU Summary Dataset

## Overview

This dataset contains patient-level ICU summaries covering demographics,
severity scores, vital signs, laboratory measurements, and treatment-related
information.

The example prediction task is **in-hospital mortality classification**:
survival versus death.

Records have already been preprocessed and aggregated. This is a tabular
dataset, not a timestamped series of individual clinical observations.

The supplied evaluation toolkit is an optional starting point. Teams may
adapt it or use other evaluation approaches and explain their choices in
their presentations.

## Data schema

The configuration's `data_specs` defines the column names and types used
by the supplied evaluator.

The schema contains **121 columns**:

| Role | Number of columns |
|---|---:|
| Identifier | 1 |
| Target | 1 |
| Boolean input features | 5 |
| Numerical input features | 114 |
| **Total** | **121** |

The 119 columns listed in `input_vars` are the predictors used by the
utility evaluator. Neither the identifier nor the target is a predictor.

Column names are case-sensitive.

## Identifier and target

| Column | Role | Description |
|---|---|---|
| `recordid` | Identifier | Identifies a record; excluded from predictive features |
| `mortality` | Binary target | `0` = survival; `1` = in-hospital death |

A unique `recordid` establishes record-level uniqueness. It does not, by
itself, establish that different records belong to different people.

The target is listed in `categorical_vars`, but its stored values must be
binary labels `0` and `1`. The supplied utility evaluator requires both
classes and does not accept missing target values.

Features and target are stored together in one table, rather than in
separate X and y files.

## Feature dictionary

### Demographics and severity scores

| Column | Description | Schema type |
|---|---|---|
| `Age` | Patient age | Numerical |
| `Gender` | Binary gender field from the source data; interpretation of the two codes is not specified by this schema | Boolean |
| `Height` | Height measurement | Numerical |
| `Weight` | Weight measurement; distinct from the first/last weight columns | Numerical |
| `SAPS-I` | Simplified Acute Physiology Score I | Numerical |
| `SOFA` | Sequential Organ Failure Assessment score | Numerical |

“Numerical” is the evaluator's type designation. Some features, such as
severity scores, are clinically discrete or ordinal rather than continuous.

### ICU and ventilation indicators

| Column | Description | Schema type |
|---|---|---|
| `CCU` | Coronary care unit indicator | Boolean |
| `CSRU` | Cardiac surgery recovery unit indicator | Boolean |
| `SICU` | Surgical ICU indicator | Boolean |
| `MechVentLast8Hour` | Mechanical-ventilation indicator for a final eight-hour period; the reference window requires source confirmation | Boolean |

For indicator fields, `1`/`True` denotes presence and `0`/`False` denotes
absence. Do not interpret a missing value as absence.

The ICU flags do not explicitly encode every possible unit as a separate
column. Do not infer the meaning of an all-zero combination without source
documentation.

### Vital signs and clinical measurements

Each of the following 12 measurement names has five summary columns:

```text
<measurement>_first
<measurement>_last
<measurement>_lowest
<measurement>_highest
<measurement>_median
```

For example, `HR` has `HR_first`, `HR_last`, `HR_lowest`,
`HR_highest`, and `HR_median`.

| Measurement prefix | Description |
|---|---|
| `DiasABP` | Diastolic arterial blood pressure |
| `GCS` | Glasgow Coma Scale score |
| `Glucose` | Glucose |
| `HR` | Heart rate |
| `MAP` | Mean arterial pressure |
| `NIDiasABP` | Non-invasive diastolic blood pressure |
| `NIMAP` | Non-invasive mean arterial pressure |
| `NISysABP` | Non-invasive systolic blood pressure |
| `RespRate` | Respiratory rate |
| `SaO2` | Oxygen saturation field |
| `SysABP` | Systolic arterial blood pressure |
| `Temp` | Temperature |

These contribute **60 numerical columns**.

### Laboratory and other measurements

Each of the following 23 names has two columns:

```text
<measurement>_first
<measurement>_last
```

| Measurement prefix | Description |
|---|---|
| `ALP` | Alkaline phosphatase |
| `ALT` | Alanine aminotransferase |
| `AST` | Aspartate aminotransferase |
| `Albumin` | Albumin |
| `BUN` | Blood urea nitrogen |
| `Bilirubin` | Bilirubin |
| `Cholesterol` | Cholesterol |
| `Creatinine` | Creatinine |
| `FiO2` | Fraction of inspired oxygen |
| `HCO3` | Bicarbonate |
| `HCT` | Hematocrit |
| `K` | Potassium |
| `Lactate` | Lactate |
| `Mg` | Magnesium |
| `Na` | Sodium |
| `PaCO2` | Arterial partial pressure of carbon dioxide |
| `PaO2` | Arterial partial pressure of oxygen |
| `Platelets` | Platelet count |
| `TroponinI` | Troponin I |
| `TroponinT` | Troponin T |
| `WBC` | White blood cell count |
| `Weight` | Weight measurement |
| `pH` | pH |

These contribute **46 numerical columns**.

`Weight`, `Weight_first`, and `Weight_last` are separate columns.
Do not assume that they are interchangeable or identical.

### Additional treatment and output summaries

| Column | Description | Schema type |
|---|---|---|
| `MechVentStartTime` | Encoded start time of mechanical ventilation | Numerical |
| `MechVentDuration` | Mechanical-ventilation duration | Numerical |
| `UrineOutputSum` | Urine-output summary field | Numerical |

The schema does not specify the time origin, units, or exact aggregation
used for these fields. In particular, do not assume that `UrineOutputSum`
is a volume in millilitres without confirming the preprocessing definition.

## Meaning of summary suffixes

| Suffix | Interpretation |
|---|---|
| `_first` | First recorded value within the source observation window |
| `_last` | Last recorded value within that window |
| `_lowest` | Minimum recorded value within that window |
| `_highest` | Maximum recorded value within that window |
| `_median` | Median recorded value within that window |

**The observation window has not been verified.** These names do not
establish whether measurements cover the first 24 hours, first 48 hours,
or the entire ICU stay.

Consequently, this dataset should not be described as supporting
admission-time mortality prediction or an early-warning model without
additional source documentation.

## Units and data quality

The schema specifies types, but not measurement units, clinical reference
ranges, or every preprocessing operation.

- Do not infer units solely from a column name.
- Unusual values may be genuine clinical extremes, recording errors,
  or transformed values.
- Negative or other special values should not automatically be treated
  as missing unless their meaning is documented.
- Apply documented preparation consistently across the real splits.
- Do not assume the table has been comprehensively clinically validated.

Any organizer-applied cleaning, sentinel replacement, or outlier handling
should be documented in the preprocessing notes accompanying the release.

## Missing values

Feature columns may contain missing values.

When using the supplied normalization helper:

- Missing feature values are allowed by default unless
  `nullable_vars` specifies a narrower set.
- Entirely missing feature columns are rejected.
- Missing target values are rejected.
- Missing or empty identifiers are rejected.
- Infinite numerical values are rejected.

Missing values should remain missing values, not be replaced with
arbitrary strings or zeros.

Different generators may require their own missing-data handling. Explain
any imputation or reconstruction in your methodology.

## Files and split roles

The starterpack uses the following file roles:

| File | Intended use |
|---|---|
| `real_train.parquet` | Input for synthetic-data generation and real-trained utility reference |
| `real_test.parquet` | Held-out utility evaluation and privacy control records |
| `example_synthetic_data.parquet` | Demonstration input for the evaluator |

The fidelity reference is selected by the evaluator implementation; consult
the evaluation README for the released configuration.

Although the test file may be available in the starterpack, using it to fit
the generator removes its interpretation as unseen evaluation/control data.
Disclose such use if you choose a different experimental design.

## Preparing synthetic data for the supplied evaluator

Generate a table containing:

1. All 119 features listed in `input_vars`.
2. A `mortality` label for each synthetic record.
3. A nonmissing, unique synthetic `recordid`.

Synthetic rows do not need to correspond to real rows. Generate new
identifiers rather than reusing real identifiers.

Parquet is recommended because it preserves column types more reliably
than CSV:

```python
synthetic.to_parquet("submission.parquet", index=False)
```


The schema requirements above apply to the supplied toolkit. Teams using
different tools should document their input handling and evaluation choices.

## Loading the data

```python
import pandas as pd

real_train = pd.read_parquet("data/real_train.parquet")
real_test = pd.read_parquet("data/real_test.parquet")

X_train = real_train.drop(columns=["recordid", "mortality"])
y_train = real_train["mortality"]
```

For the official predictor order, select columns from the configuration:

```python
X_train = real_train[data_specs["input_vars"]]
y_train = real_train[data_specs["target_var"]]
```

## Source and usage terms

The dataset was obtained from:

[ICU Patient Outcome Prediction — Kaggle](https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu)

The Kaggle listing identifies its license as **CC BY-NC-SA 4.0**.

The precise original source release, measurement window, and upstream
redistribution permissions have not been independently verified in this
documentation. The Kaggle listing alone does not establish those details.

Consult the source documentation and applicable terms before further use
or redistribution. Retain attribution and document modifications.

## Scope

This dataset is provided for exploring synthetic-data generation and the
tradeoffs between predictive utility, statistical fidelity, and measured
privacy risk.

Neither the source table nor high evaluation scores establish clinical
validity, deployment suitability, or a formal privacy guarantee.