
#!/usr/bin/env python3
"""Build admission-level MIMIC OMOP mortality data.

Usage:
    python build_mimic_mortality_folder.py \
        --folder /path/to/unzipped_omop \
        --outdir /path/to/output

Requires pandas, numpy, scikit-learn.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# MIMIC source item IDs paired with expected measurement units.
ITEMS = {
    "heart_rate": (["220045"], "bpm"),
    "resp_rate": (["220210", "224690"], "insp/min"),
    "spo2": (["220277"], "%"),
    "sbp": (["220179", "220050"], "mmHg"),
    "dbp": (["220180", "220051"], "mmHg"),
    "map": (["220181", "220052"], "mmHg"),
    "temperature_c": (["223761"], "°F"), #this concept id is for temperature in fahrenheit
    "creatinine": (["50912"], "mg/dL"),
    "bun": (["51006"], "mg/dL"),
    "glucose": (["50931"], "mg/dL"),
    "hemoglobin": (["51222"], "g/dL"),
    "hematocrit": (["51221"], "%"),
    "wbc": (["51301"], "K/uL"),
    "platelets": (["51265"], "K/uL"),
    "sodium": (["50983"], "mEq/L"),
    "potassium": (["50971"], "mEq/L"),
    "chloride": (["50902"], "mEq/L"),
    "bicarbonate": (["50882"], "mEq/L"),
    "anion_gap": (["50868"], "mEq/L"),
    "calcium": (["50893"], "mg/dL"),
    "inr": (["51237"], None),
    "pt": (["51274"], "sec"),
    "lactate": (["50813"], "mmol/L"),
}

# NOTE: this is limit in fahrenheit despite the 'c' in the key 
# because we first extract the temperature in fahrenheit using the concept id given above
# and then convert it to celcius later before saving the final dataset
LIMITS = {
    "heart_rate": (20, 250),
    "resp_rate": (3, 80),
    "spo2": (40, 100),
    "sbp": (40, 300),
    "dbp": (15, 200),
    "map": (25, 220),
    "temperature_c": (85, 110), #
    "creatinine": (0.1, 25),
    "bun": (1, 250),
    "glucose": (20, 1000),
    "hemoglobin": (2, 25),
    "hematocrit": (5, 75),
    "wbc": (0.1, 300),
    "platelets": (1, 2000),
    "sodium": (90, 190),
    "potassium": (1, 10),
    "chloride": (60, 150),
    "bicarbonate": (3, 55),
    "anion_gap": (0, 55),
    "calcium": (2, 20),
    "inr": (0.5, 20),
    "pt": (5, 200),
    "lactate": (0.1, 35),
}


def main(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    folder = Path(args.folder).expanduser().resolve()

    if not folder.is_dir():
        raise NotADirectoryError(
            f"OMOP input folder does not exist: {folder}"
        )

    def read(name, cols=None):
        matches = sorted(
            p for p in folder.rglob("*.csv")
            if p.stem.lower() == name.lower()
        )

        if not matches:
            raise FileNotFoundError(
                f"Cannot find {name}.csv under {folder}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple files named {name}.csv found: {matches}"
            )

        return pd.read_csv(
            matches[0],
            usecols=cols,
            low_memory=False
        )

    # =========================================================
    # STEP 1: Construct the admission-level baseline
    # =========================================================

    hospital_concepts = [8870, 262, 9201]

    v = read("visit_occurrence", [
        "person_id",
        "visit_occurrence_id",
        "visit_concept_id",
        "visit_start_datetime",
        "visit_end_datetime",
    ])

    p = read("person", [
        "person_id",
        "year_of_birth",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
    ])

    d = read("death", [
        "person_id",
        "death_datetime",
        "death_date",
    ])

    for table in (v, p, d):
        table["person_id"] = table.person_id.astype("string")

    v["visit_occurrence_id"] = (
        v.visit_occurrence_id.astype("string")
    )

    v["visit_concept_id"] = pd.to_numeric(
        v.visit_concept_id,
        errors="coerce"
    )

    v = v[
        v.visit_concept_id.isin(hospital_concepts)
    ].copy()

    v["admit_dt"] = pd.to_datetime(
        v.visit_start_datetime,
        errors="coerce"
    )

    v["end_dt"] = pd.to_datetime(
        v.visit_end_datetime,
        errors="coerce"
    )

    if v[["admit_dt", "end_dt"]].isna().any().any():
        raise ValueError(
            "Hospital encounters contain invalid timestamps"
        )

    if (v.end_dt < v.admit_dt).any():
        raise ValueError(
            "Hospital encounter ends before admission"
        )

    if v.visit_occurrence_id.duplicated().any():
        raise ValueError(
            "Duplicate visit_occurrence_id"
        )

    v = v.sort_values([
        "person_id",
        "admit_dt",
        "visit_occurrence_id",
    ]).copy()

    v["prior_hospital_admissions"] = (
        v.groupby("person_id").cumcount()
    )

    base = v.merge(
        p,
        on="person_id",
        how="left",
        validate="many_to_one"
    )

    if base.year_of_birth.isna().any():
        raise ValueError(
            "Missing demographics or year of birth"
        )

    base["age_at_admission"] = (
        base.admit_dt.dt.year
        - base.year_of_birth.astype(int)
    )

    # =========================================================
    # STEP 2: Assign in-hospital mortality
    # =========================================================

    d["death_time"] = (
        pd.to_datetime(
            d.death_datetime,
            errors="coerce"
        ).fillna(
            pd.to_datetime(
                d.death_date,
                errors="coerce"
            )
        )
    )

    death_matches = base[[
        "person_id",
        "visit_occurrence_id",
        "admit_dt",
        "end_dt",
    ]].merge(
        d[["person_id", "death_time"]],
        on="person_id",
        how="left"
    )

    death_matches["within_stay"] = (
        (death_matches.death_time >= death_matches.admit_dt)
        & (death_matches.death_time <= death_matches.end_dt)
    )

    deaths = (
        death_matches
        .groupby("visit_occurrence_id", sort=False)
        .within_stay.any()
    )

    base["hospital_mortality"] = (
        base.visit_occurrence_id
        .map(deaths)
        .fillna(False)
        .astype(int)
    )

    base = base[[
        "person_id",
        "visit_occurrence_id",
        "visit_concept_id",
        "visit_start_datetime",
        "visit_end_datetime",
        "age_at_admission",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "prior_hospital_admissions",
        "hospital_mortality",
    ]].copy()

    # Save intermediate cohort for reproducibility.
    base.to_csv(
        out / "mimic_omop_admission_mortality.csv",
        index=False
    )

    baseline_count = len(base)
    baseline_deaths = int(
        base.hospital_mortality.sum()
    )

    print(
        f"Intermediate baseline: {len(base)} admissions, "
        f"{base.person_id.nunique()} patients, "
        f"{baseline_deaths} deaths"
    )

    # =========================================================
    # STEP 3: Apply 24-hour landmark eligibility
    # =========================================================

    base["admit"] = pd.to_datetime(
        base.visit_start_datetime
    )

    base["discharge"] = pd.to_datetime(
        base.visit_end_datetime
    )

    base["stay_hours"] = (
        (base.discharge - base.admit)
        .dt.total_seconds() / 3600
    )

    excluded = base.loc[
        ~(base.stay_hours >= 24),
        [
            "person_id",
            "visit_occurrence_id",
            "visit_start_datetime",
            "visit_end_datetime",
            "stay_hours",
            "hospital_mortality",
        ]
    ].copy()

    excluded["exclusion_reason"] = (
        "stay_shorter_than_24_hours"
    )

    excluded.to_csv(
        out / "excluded_admissions.csv",
        index=False
    )

    base = base.loc[
        base.stay_hours >= 24
    ].copy()

    assert base.visit_occurrence_id.is_unique

    base["cutoff"] = (
        base.admit + pd.Timedelta(hours=24)
    )

    # =========================================================
    # STEP 4: Extract first-24-hour measurements
    # =========================================================

    m = read("measurement", [
        "person_id",
        "visit_occurrence_id",
        "measurement_datetime",
        "measurement_source_value",
        "unit_source_value",
        "value_as_number",
    ])

    m["visit_occurrence_id"] = (
        m.visit_occurrence_id.astype("string")
    )

    m = m[
        m.visit_occurrence_id.isin(base.visit_occurrence_id)
        & m.value_as_number.notna()
    ].copy()

    m["measurement_datetime"] = pd.to_datetime(
        m.measurement_datetime,
        errors="coerce"
    )

    m = m.merge(
        base[[
            "visit_occurrence_id",
            "admit",
            "cutoff",
        ]],
        on="visit_occurrence_id",
        how="inner",
        validate="many_to_one"
    )

    m = m[
        (m.measurement_datetime >= m.admit)
        & (m.measurement_datetime < m.cutoff)
    ].copy()

    m["itemid"] = (
        m.measurement_source_value
        .astype("string")
        .str.strip()
    )

    features = base[[
        "visit_occurrence_id"
    ]].copy()

    coverage = {}

    for feature, (codes, unit) in ITEMS.items():

        sub = m[
            m.itemid.isin(codes)
        ].copy()

        if unit is not None:
            sub = sub[
                sub.unit_source_value
                .astype("string")
                .str.strip() == unit
            ]

        lo, hi = LIMITS[feature]

        sub = sub[
            sub.value_as_number.between(lo, hi)
        ].copy()

        # Convert valid Fahrenheit temperatures to Celsius
        if feature == "temperature_c":
                sub["value_as_number"] = (
                        (sub["value_as_number"] - 32) * 5 / 9
                )

        agg = (
            sub.groupby("visit_occurrence_id")
            .value_as_number
            .agg(["mean", "count"])
        )

        agg.columns = [
            feature + "_mean_24h",
            feature + "_n_24h",
        ]

        features = features.merge(
            agg,
            left_on="visit_occurrence_id",
            right_index=True,
            how="left",
            validate="one_to_one"
        )

        features[feature + "_n_24h"] = (
            features[feature + "_n_24h"]
            .fillna(0)
            .astype(int)
        )

        coverage[feature] = int(
            features[
                feature + "_mean_24h"
            ].notna().sum()
        )

    # =========================================================
    # STEP 5: Historical comorbidities
    # =========================================================

    c = read("condition_occurrence", [
        "person_id",
        "condition_start_datetime",
        "condition_start_date",
        "condition_source_value",
    ])

    c["person_id"] = (
        c.person_id.astype("string")
    )

    c["diagnosis_time"] = (
        pd.to_datetime(
            c.condition_start_datetime,
            errors="coerce"
        ).fillna(
            pd.to_datetime(
                c.condition_start_date,
                errors="coerce"
            )
        )
    )

    c["icd"] = (
        c.condition_source_value
        .astype("string")
        .str.upper()
        .str.replace(".", "", regex=False)
    )

    patterns = {
        "diabetes_history": r"^(?:250|E0[89]|E1[0-3])",
        "hypertension_history": r"^(?:40[1-5]|I1[0-6])",
        "heart_failure_history": r"^(?:428|I50)",
        "copd_history": r"^(?:49[0-2]|496|J4[1-4])",
        "mi_history": r"^(?:410|412|I2[12])",
    }

    c = c[
        c.person_id.isin(base.person_id)
    ].copy()

    for feat, pat in patterns.items():
        c[feat] = (
            c.icd.str.contains(
                pat,
                regex=True,
                na=False
            ).astype("int8")
        )

    histories = []

    for row in base[[
        "person_id",
        "visit_occurrence_id",
        "admit",
    ]].itertuples(index=False):

        prior = c[
            (c.person_id == row.person_id)
            & (c.diagnosis_time < row.admit)
        ]

        histories.append({
            "visit_occurrence_id": row.visit_occurrence_id,
            **{
                feat: (
                    int(prior[feat].max())
                    if len(prior) else 0
                )
                for feat in patterns
            }
        })

    features = features.merge(
        pd.DataFrame(histories),
        on="visit_occurrence_id",
        validate="one_to_one"
    )

    # =========================================================
    # STEP 6: Assemble final dataset
    # =========================================================

    df = base.merge(
        features,
        on="visit_occurrence_id",
        validate="one_to_one"
    )

    df["hospital_mortality"] = (
        df.hospital_mortality.astype(int)
    )

    # Exclude discharge times, LOS, and death timestamps.
    columns = [
        "person_id",
        "visit_occurrence_id",
        "hospital_mortality",
        "age_at_admission",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "visit_concept_id",
        "prior_hospital_admissions",
    ] + [
        col for col in features.columns
        if col != "visit_occurrence_id"
    ]

    df = df[columns].copy()

    # =========================================================
    # STEP 7: Patient-disjoint train/test split
    # =========================================================

    groups = df.person_id.to_numpy()
    candidates = []

    for seed in range(100):

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        for tr, te in sgkf.split(
            df,
            df.hospital_mortality,
            groups=groups
        ):
            train_part = df.iloc[tr]
            test_part = df.iloc[te]

            if (
                test_part.hospital_mortality.sum() < 1
                or train_part.hospital_mortality.sum() < 1
            ):
                continue

            score = (
                abs(
                    test_part.hospital_mortality.mean()
                    - train_part.hospital_mortality.mean()
                ) * 4
                + abs(len(test_part) / len(df) - 0.2)
                + abs(
                    test_part.person_id.nunique()
                    / df.person_id.nunique()
                    - 0.2
                )
            )

            candidates.append(
                (score, seed, tr, te)
            )

    if not candidates:
        raise RuntimeError(
            "No split with positive cases in both groups"
        )

    _, seed, tr, te = min(
        candidates,
        key=lambda x: x[0]
    )

    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    assert set(train.person_id).isdisjoint(
        set(test.person_id)
    )

    assert set(train.visit_occurrence_id).isdisjoint(
        set(test.visit_occurrence_id)
    )

    df["split"] = "train"
    df.loc[test.index, "split"] = "test"

    df.to_csv(
        out / "mortality_full_with_split.csv",
        index=False
    )

    train.to_csv(
        out / "mortality_real_train.csv",
        index=False
    )

    test.to_csv(
        out / "mortality_real_test.csv",
        index=False
    )

    # =========================================================
    # STEP 8: Generate dataset report
    # =========================================================

    report = {
        "baseline": {
            "admissions": baseline_count,
            "deaths": baseline_deaths,
        },
        "excluded_short_stays": len(excluded),
        "landmark_hours": 24,
        "cohort_rule": (
            "visit_concept_id in (8870,262,9201); "
            "duration >= 24h"
        ),
        "feature_window": (
            "[admission, admission+24h)"
        ),
        "split_seed": seed,
        "total": {
            "admissions": len(df),
            "patients": int(df.person_id.nunique()),
            "deaths": int(df.hospital_mortality.sum()),
        },
        "train": {
            "admissions": len(train),
            "patients": int(train.person_id.nunique()),
            "deaths": int(train.hospital_mortality.sum()),
            "mortality_rate": float(
                train.hospital_mortality.mean()
            ),
        },
        "test": {
            "admissions": len(test),
            "patients": int(test.person_id.nunique()),
            "deaths": int(test.hospital_mortality.sum()),
            "mortality_rate": float(
                test.hospital_mortality.mean()
            ),
        },
        "measurement_coverage_admissions": coverage,
        "warning": (
            "Small demo dataset. "
            "Do not use identifiers as predictors. "
            "Fit preprocessing on training data only."
        ),
    }

    (
        out / "mortality_dataset_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    # =========================================================
    # STEP 9: Generate methodology documentation
    # =========================================================

    methodology = """# Mortality dataset construction

## Source files
- person.csv
- visit_occurrence.csv
- death.csv
- measurement.csv
- condition_occurrence.csv

## Intermediate cohort
1. Select visit_concept_id 8870, 262, 9201.
2. Join patient demographics.
3. Calculate age at admission.
4. Count earlier hospital admissions chronologically.
5. Assign mortality if death occurred during admission.
6. Save mimic_omop_admission_mortality.csv.

## Feature extraction
7. Exclude hospital stays shorter than 24 hours.
8. Extract measurements from the first 24 hours.
9. Filter physiologically implausible values.
10. Calculate measurement means and counts.
11. Add historical comorbidity flags.

## Train/test split
12. Split by patient, not admission.
13. Approximately stratify by mortality.
14. Save the full dataset and train/test partitions.

## Important limitations
- Very small dataset and few mortality events.
- Repeated admissions from the same patient are correlated.
- Source measurement units and diagnosis mappings need validation.
- Admission concept IDs are specific to the selected cohort.
- The model predicts mortality at the 24-hour landmark.
- Patient and visit identifiers are not predictive features.
"""

    (
        out / "BUILD_METHODS.md"
    ).write_text(
        methodology,
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
        help="Directory containing unzipped OMOP CSV files"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for generated output files"
    )

    main(parser.parse_args())
