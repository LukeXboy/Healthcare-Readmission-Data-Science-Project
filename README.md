# Hospital Readmission Prediction Within 30 Days

**Healthcare Data Science & Machine Learning | Work in Progress**  
**Python · Pandas · scikit-learn · XGBoost · Google Colab**

> **Project status (November 5, 2026):** Initial patient-aware modeling and validation experiments are complete. Feature expansion, final model selection, calibration, interpretation, and final test evaluation are **not yet complete**. This is an educational portfolio study—not a clinically validated prediction system.

## 1. Project overview

**Research question:** Can information available by hospital discharge help predict whether an encounter involving a patient with diabetes will be followed by a **recorded readmission within 30 days**?

This project demonstrates the end-to-end methodology of a healthcare classification problem:

- Define an encounter-level binary prediction target and prediction time.
- Prevent overlap of individual patients across development, validation, and test partitions.
- Build preprocessing and classification pipelines with scikit-learn.
- Compare baseline and nonlinear models using metrics suited to imbalanced outcomes.
- Examine threshold-dependent precision, recall, and operational workload.
- Tune hyperparameters with **patient-grouped cross-validation**.
- Audit feature availability, missingness, calibration, and limitations before any final claims.

**Prediction point:** At hospital discharge. Candidate predictors must be available at or before discharge. The features used so far are a provisional baseline subset; the final feature eligibility audit is still ongoing.

## 2. Dataset and attribution

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008)
- **DOI:** https://doi.org/10.24432/C5230J
- **Coverage:** 101,766 hospital encounters from US hospitals/integrated delivery networks during 1999–2008.
- **Observation grain:** One record per hospital **encounter**, not one record per person.
- **Identifiers:** \`encounter_id\` is an encounter identifier; \`patient_nbr\` groups encounters belonging to the same patient.
- **Outcome:** The original \`readmitted\` column has categories \`<30\`, \`>30\`, and \`NO\`.

The original dataset is not committed to this repository. See the UCI source for attribution, licensing, definitions, and terms of use.

### Binary target

| Original \`readmitted\` | \`readmitted_30d\` | Interpretation |
|---|---:|---|
| \`<30\` | 1 | Recorded early readmission |
| \`>30\` | 0 | Recorded readmission after 30 days |
| \`NO\` | 0 | No readmission recorded in the dataset |

\`readmitted\` and \`readmitted_30d\` are **never included as input features**. The target reflects records in this historical dataset and does not mean that a readmission was preventable.

## 3. Patient-disjoint evaluation design

Splits use \`patient_nbr\` as a grouping key so the same patient cannot appear across model-development, validation, and test partitions.

1. \`GroupShuffleSplit(test_size=0.2, random_state=42)\` separates **test patients** from the initial training pool.
2. A second \`GroupShuffleSplit(test_size=0.2, random_state=123)\` divides the initial training pool into **development training** and **validation**.
3. All three pairwise patient-ID intersection checks returned **zero overlap**.
4. The **test partition has not been used to fit, tune, select, or evaluate any model** in this project to date.

### Verified split sizes

| Measure | Development training | Validation | Held-out test |
|---|---:|---:|---:|
| Encounters | 65,274 | 16,339 | 20,153 |
| Distinct patients | 45,771 | 11,443 | 14,304 |
| Positive class prevalence | 11.29% | 11.26% | 10.67% |

Total: **101,766 encounters**, with **zero patient overlap** between partitions. The approximate 64% / 16% / 20% encounter split follows from patient-level allocation; the exact proportions of encounters need not match the proportions of patients.

**Important methodological limitation:** Patient-disjoint splitting assesses generalization to different patients in the same historical data source. It does not replace chronological or external-hospital validation. \`GroupShuffleSplit\` does not guarantee stratification; observed prevalence was checked.

## 4. Initial feature set and preprocessing

The first model comparison uses **14 features**.

**Numerical (8):** \`time_in_hospital\`, \`num_lab_procedures\`, \`num_procedures\`, \`num_medications\`, \`number_outpatient\`, \`number_emergency\`, \`number_inpatient\`, \`number_diagnoses\`.

**Categorical (6):** \`age\`, \`gender\`, \`admission_type_id\`, \`admission_source_id\`, \`change\`, \`diabetesMed\`.

Although \`admission_type_id\` and \`admission_source_id\` are stored as integers, they represent categorical codes and are **one-hot encoded**, not treated as continuous numbers.

A scikit-learn \`ColumnTransformer\` is combined with classifiers through \`Pipeline\`:

| Feature type | Preparation |
|---|---|
| Numerical | Median imputation → StandardScaler |
| Categorical | Most-frequent imputation → OneHotEncoder(\`handle_unknown='ignore'\`) |

All fitted preprocessing steps learn only from the relevant development-training fold or training partition. Selected 14 features showed no missing values after the initial handling of \`?\` placeholders in the development audit; the imputation components remain in the pipeline for robustness.

**Excluded from model inputs:** \`encounter_id\`, \`patient_nbr\`, \`readmitted\`, \`readmitted_30d\`; \`discharge_disposition_id\` is excluded from the initial baseline pending a more detailed eligibility review. \`patient_nbr\` is retained separately for grouping.

Feature availability at discharge still requires a careful audit, particularly for historical utilization definitions and administrative codes.

## 5. Models trained and validation results

Three initial classifiers have been trained and compared on the **same patient-disjoint validation set**. Results below are **validation metrics**, not independent test or clinical performance estimates.

| Model | Validation ROC-AUC | Validation Average Precision (AP) |
|---|---:|---:|
| Logistic Regression | 0.6339 | 0.2000 |
| Random Forest (200 trees, max depth 12, min leaf 10) | 0.6417 | 0.1997 |
| **XGBoost (initial)** | **0.6434** | **0.2043** |
| XGBoost (random-search selected) | 0.6425 | 0.2019 |

**Validation early-readmission prevalence:** approximately **0.1126 (11.26%)**. This is a useful reference for interpreting Average Precision under a constant-score baseline.

**Interpretation:** XGBoost currently leads the initial comparison by a small margin, but the differences are modest and have not been assessed with uncertainty intervals. No result establishes clinical suitability.

### Initial XGBoost configuration

\`\`\`python
XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
\`\`\`

### Training vs. validation: XGBoost

| Metric | Development training | Validation |
|---|---:|---:|
| ROC-AUC | 0.6615 | 0.6434 |
| Average Precision | 0.2248 | 0.2043 |

The observed training–validation gap is modest; there is no strong indication of **severe** overfitting in this experiment. Both training and validation discrimination remain limited, so more regularization alone may not solve the core problem.

## 6. Logistic Regression threshold analysis

The positive outcome occurs in roughly 11% of encounters, so a default probability threshold of 0.50 performs poorly on recall. Using **validation predictions only**, the following thresholds were compared:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.1479 | 0.6547 | 0.2414 |
| 0.20 | 0.2751 | 0.1251 | 0.1720 |
| 0.30 | 0.4208 | 0.0593 | 0.1039 |
| 0.40 | 0.4597 | 0.0310 | 0.0581 |
| 0.50 | 0.5082 | 0.0169 | 0.0326 |

At **0.50**, the validation confusion matrix was:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 14,470 | 30 |
| Actual positive | 1,808 | 31 |

At this threshold, recall is approximately **1.69%**: only **31 of 1,839** observed early-readmission encounters are identified.

**Do not interpret 0.10 as the final recommended threshold.** Although it has the highest F1 among these five choices, its low precision would flag many encounters. Choosing a policy requires explicit follow-up capacity, costs, benefit assumptions, and properly calibrated risks. The model has **not** been evaluated for real-world intervention impact.

## 7. Patient-grouped hyperparameter search

The project ran a small \`RandomizedSearchCV\` experiment for XGBoost:

- **CV strategy:** \`StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)\`.
- **Grouping:** \`groups=dev_train_df['patient_nbr']\` passed to search fitting.
- **Objective:** Maximize **Average Precision**.
- **Search size:** 6 sampled configurations × 3 grouped folds = 18 cross-validation fits, followed by refitting the best configuration.
- **Best mean cross-validation AP:** **0.1983**.

Selected hyperparameters:

| Parameter | Selected value |
|---|---:|
| \`max_depth\` | 4 |
| \`min_child_weight\` | 10 |
| \`subsample\` | 0.8 |
| \`reg_lambda\` | 1 |

Other base settings included \`n_estimators=200\` and \`learning_rate=0.05\`.

The selected model obtained **0.6425 ROC-AUC** and **0.2019 AP** on the separate validation partition—slightly below the initial XGBoost configuration. This experiment demonstrates grouped model selection but **does not show an improvement from tuning**.

## 8. Progress and next steps

### Completed in working Google Colab sessions

- [x] Loaded and examined the source data.
- [x] Created the binary early-readmission target.
- [x] Verified patient-disjoint development training, validation, and test partitions.
- [x] Implemented a 14-feature scikit-learn preprocessing pipeline.
- [x] Trained Logistic Regression, Random Forest, and XGBoost.
- [x] Calculated validation ROC-AUC and Average Precision.
- [x] Examined Logistic Regression precision/recall/F1 at five thresholds.
- [x] Compared XGBoost development-training and validation performance.
- [x] Ran a grouped hyperparameter search for XGBoost.

### Upcoming

- [ ] Audit \`diag_1\`, \`diag_2\`, \`diag_3\`, \`A1Cresult\`, \`max_glu_serum\`, and \`medical_specialty\`: missingness, category counts, data meaning, and discharge-time availability.
- [ ] Engineer clinically and temporally defensible features and compare models without leaking outcome information.
- [ ] Establish an explicit dummy/constant-prediction benchmark and compare models with uncertainty where practical.
- [ ] Check probability calibration and evaluate operationally relevant thresholds/top-risk selection.
- [ ] Analyze feature contributions and subgroup behavior; document fairness and data-quality limitations.
- [ ] Finalize the model using development/validation evidence, then perform **one final evaluation** on the untouched test set.
- [ ] Export the latest Google Colab notebook to GitHub, ensure it runs reproducibly, and finalize conclusions.
- [ ] Optionally prototype nonclinical inference/deployment, clearly separated from any patient-care use.

**Next technical task:** Audit the six candidate clinical and administrative features listed above using only the development-training set.

## 9. Repository notebook and reproducibility status

**Notebook currently published:** [\`01_data_audit.ipynb\`](01_data_audit.ipynb). It contains the initial dataset audit, first patient-disjoint train/test split, and binary target creation. **The later three-way split, preprocessing, trained models, and tuning results have been completed in a working Colab session but are not yet included in the committed notebook.** Those results above were reviewed from session outputs and will become fully reproducible once the latest notebook is exported and committed.

[Open the current repository notebook in Colab](https://colab.research.google.com/github/LukeXboy/Healthcare-Readmission-Data-Science-Project/blob/main/01_data_audit.ipynb).

**Current loading caveat:** The published notebook reads an uploaded Excel-converted file at:

\`\`\`python
df = pd.read_excel("/content/diabetic_data.csv.xlsx", na_values=["?"])
\`\`\`

To rerun that notebook unchanged, upload the corresponding workbook to the Colab runtime. Alternatively, download and extract the official \`diabetic_data.csv\` from UCI and **change the loading cell** to \`pd.read_csv("/content/diabetic_data.csv", na_values=["?"])\`. The published notebook does **not** currently download the source automatically.

Until a complete updated notebook and environment requirements are committed, this README documents **verified progress**, not a fully automated end-to-end reproduction.

## 10. Limitations and responsible use

- **Historical, geographically different cohort:** US 1999–2008 encounter data should not be assumed representative of contemporary Canadian patients or hospitals.
- **Outcome limitations:** A recorded early readmission is not the same as preventable readmission, and this observational dataset does not measure the causal effect of any intervention.
- **Prediction-time eligibility:** Discharge-time feature availability and disposition/eligibility require further review. No feature list should be treated as final until this audit is complete.
- **Data and label quality:** Missing categories, coding differences, changing hospital practices, and potential censoring can limit reliability.
- **Evaluation:** Patient-disjoint splits prevent patient overlap, but do not constitute temporal, external-hospital, or prospective validation.
- **Subgroups and calibration:** Fairness, calibration, uncertainty, and clinical workflow impact have not yet been fully assessed.
- **Clinical safety:** Do not use this portfolio model for patient diagnosis, clinical risk stratification, or care decisions. Any real clinical application requires governance, evidence, and independent prospective validation.

## Citation

Clore, J., Cios, K., DeShazo, J. P., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
