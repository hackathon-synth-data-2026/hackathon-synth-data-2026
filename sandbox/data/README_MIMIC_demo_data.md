
# Clinical Feature Definitions and Selection Rationale

The mortality dataset includes vital signs and laboratory measurements
recorded during the first 24 hours of hospitalization.

These features were selected to capture cardiovascular, respiratory,
renal, metabolic, hematological, and coagulation abnormalities that
may be associated with in-hospital mortality.

The plausibility limits are preprocessing thresholds, **not normal
clinical reference ranges**. They are intended to exclude likely
measurement errors while retaining clinically meaningful abnormalities.

The limits are heuristic and have not been formally validated against
this particular MIMIC-IV extract.

## 1. Vital Signs

| Feature | Clinical meaning | Reason for inclusion | Plausibility limits |
|---|---|---|---|
| `heart_rate` | Heartbeats per minute | Tachycardia or bradycardia may indicate infection, shock, bleeding, or physiological stress | 20–250 bpm |
| `resp_rate` | Respiratory rate | Indicates respiratory distress, metabolic acidosis, or acute deterioration | 3–80 breaths/min |
| `spo2` | Peripheral oxygen saturation | Reflects oxygenation and potential respiratory failure | 40–100% |
| `sbp` | Systolic blood pressure | Hypotension may indicate circulatory shock; severe hypertension may indicate cardiovascular stress | 40–300 mmHg |
| `dbp` | Diastolic blood pressure | Provides information about vascular resistance and circulatory status | 15–200 mmHg |
| `map` | Mean arterial pressure | Important indicator of organ perfusion and hemodynamic stability | 25–220 mmHg |
| `temperature_c` | Body temperature | Fever or hypothermia may indicate infection, inflammation, or severe illness | 29.4–43.3 °C |

## 2. Kidney Function and Electrolytes

| Feature | Clinical meaning | Reason for inclusion | Plausibility limits |
|---|---|---|---|
| `creatinine` | Blood creatinine concentration | Marker of kidney function and potential acute or chronic renal impairment | 0.1–25 mg/dL |
| `bun` | Blood urea nitrogen | Reflects renal function, hydration status, and protein metabolism | 1–250 mg/dL |
| `sodium` | Blood sodium concentration | Abnormalities may indicate fluid imbalance and neurological or systemic complications | 90–190 mEq/L |
| `potassium` | Blood potassium concentration | Severe abnormalities can cause potentially fatal cardiac arrhythmias | 1–10 mEq/L |
| `chloride` | Blood chloride concentration | Helps characterize electrolyte and acid-base disturbances | 60–150 mEq/L |
| `bicarbonate` | Serum bicarbonate concentration | Low levels may indicate metabolic acidosis associated with shock, renal failure, or infection | 3–55 mEq/L |
| `anion_gap` | Calculated difference between measured cations and anions | Helps identify metabolic disturbances, including certain forms of metabolic acidosis | 0–55 mEq/L |
| `calcium` | Total blood calcium concentration | Severe abnormalities may affect cardiac, muscular, and neurological function | 2–20 mg/dL |

## 3. Blood Counts and Coagulation

| Feature | Clinical meaning | Reason for inclusion | Plausibility limits |
|---|---|---|---|
| `hemoglobin` | Hemoglobin concentration | Indicates anemia, blood loss, and reduced oxygen-carrying capacity | 2–25 g/dL |
| `hematocrit` | Percentage of blood volume occupied by red blood cells | Complements hemoglobin in assessing anemia, bleeding, and hemoconcentration | 5–75% |
| `wbc` | White blood cell count | Elevated or reduced counts may indicate infection, inflammation, or bone marrow dysfunction | 0.1–300 K/µL |
| `platelets` | Platelet count | Low counts may indicate coagulation disorders, sepsis-associated consumption, or bone marrow dysfunction | 1–2000 K/µL |
| `inr` | International normalized ratio | Standardized measure of clotting; may indicate anticoagulation, liver dysfunction, or coagulopathy | 0.5–20 (dimensionless) |
| `pt` | Prothrombin time | Measures clotting time and complements INR in assessing coagulation abnormalities | 5–200 seconds |

## 4. Metabolic Function and Tissue Perfusion

| Feature | Clinical meaning | Reason for inclusion | Plausibility limits |
|---|---|---|---|
| `glucose` | Blood glucose concentration | Hypoglycemia and hyperglycemia may indicate metabolic dysfunction or physiological stress | 20–1000 mg/dL |
| `lactate` | Blood lactate concentration | Elevated values may indicate impaired tissue perfusion, metabolic stress, or altered lactate clearance | 0.1–35 mmol/L |

## 5. Derived Features

For each physiological or laboratory variable, two admission-level
features are calculated using measurements from the first 24 hours.

| Feature type | Example | Interpretation |
|---|---|---|
| Mean (`_mean_24h`) | `heart_rate_mean_24h` | Arithmetic mean of valid measurements recorded during the first 24 hours |
| Count (`_n_24h`) | `heart_rate_n_24h` | Number of valid measurements recorded during the first 24 hours |

Measurements are filtered by source item ID, expected unit, timestamp,
and plausibility range before aggregation.

A missing mean indicates that no valid measurement was available.
The corresponding count is zero.

Measurement counts may provide information about monitoring intensity,
but they may also reflect hospital workflows rather than patient
physiology.

## 6. Feature Selection Rationale

The feature set was selected based on three considerations:

1. **Clinical relevance:** Coverage of major physiological systems
   involved in acute illness and clinical deterioration.
2. **Established clinical practice:** Partial overlap with measurements
   used in severity scoring systems such as SOFA and APACHE II.
   The dataset does not implement either score.
3. **Data availability:** Selection of commonly recorded measurements
   that can be extracted from the MIMIC-IV OMOP `measurement` table.

This is a pragmatic, clinically motivated feature set rather than
a formally optimized or validated predictor selection.

## 7. Methodological Limitations

- Plausibility thresholds are heuristic and should be documented
  and validated against the source data.
- Feature means discard temporal trends, variability, and extreme
  values within the observation window.
- Missing laboratory measurements should not be interpreted as zero.
- Measurement frequency can encode clinical workflow patterns.
- Some variables are strongly correlated, such as hemoglobin and
  hematocrit, or PT and INR.
- The dataset is small, with very few mortality events, so
  downstream predictive performance estimates are uncertain.
- For a true 24-hour landmark prediction, patients must still be
  alive and hospitalized at the prediction time.
- Feature availability, source item IDs, units, and exclusion counts
  should be checked before synthetic-data generation.
