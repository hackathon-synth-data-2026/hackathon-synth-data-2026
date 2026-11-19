# Mortality dataset construction

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
