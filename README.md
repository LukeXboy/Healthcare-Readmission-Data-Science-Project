# Hospital Readmission Prediction Within 30 Days

**Healthcare Machine Learning Portfolio Project | Work in Progress**  
**Python · Pandas · scikit-learn · Google Colab**

## Overview

This project explores whether information available **at or before hospital discharge** can help identify diabetes-related inpatient encounters associated with a **recorded readmission within 30 days**. It is designed to demonstrate a rigorous end-to-end classification workflow, including data auditing, patient-aware evaluation, preprocessing, baseline modeling, and interpretable model assessment.

**Current stage:** Data exploration and target/split setup. **No predictive model has been trained or evaluated yet.**

> **Important:** This is a retrospective educational project using historical US hospital data. It is **not** a clinically validated model, a medical recommendation system, or evidence of performance in present-day Canadian hospitals.

## Data source

**Dataset:** [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008) — UCI Machine Learning Repository  
**DOI:** [10.24432/C5230J](https://doi.org/10.24432/C5230J)  
**Original authors:** John Clore, Krzysztof Cios, Jonathan P. DeShazo, and Beata Strack  
**License:** CC BY 4.0 (see the [official dataset page](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008) for attribution and terms)

The dataset contains **101,766 hospital encounters** from **130 US hospitals and integrated delivery networks** during **1999–2008**. An encounter is not equivalent to a unique patient: `encounter_id` identifies a visit, while `patient_nbr` identifies a patient who may have multiple encounters. The dataset includes demographics, diagnoses, medications, healthcare utilization indicators, and an original three-category readmission label.

The original data are not bundled with this repository. The Colab notebook downloads the source archive from UCI.

## Research question and target

**Question:** Can an ML classifier distinguish encounters associated with an early readmission from those without a recorded early readmission?

The original `readmitted` column is mapped into a binary target:

| Original value | `readmitted_30d` | Meaning |
|---|---:|---|
| `<30` | 1 | Recorded readmission within 30 days |
| `>30` | 0 | Recorded readmission after 30 days |
| `NO` | 0 | No recorded readmission |

**Prediction point:** Hospital discharge. Only predictors available by that point should be considered, subject to a feature-level leakage review. The original `readmitted` column, new target, and record identifiers must not be used directly as model predictors.

The label captures **recorded** readmissions in this dataset; it does not establish whether a readmission was preventable, and the released data do not support claims of prospective clinical effectiveness.

## Progress so far (November 2, 2026)

### Completed

- [x] Selected and loaded the UCI diabetes hospital dataset in **Google Colab**.
- [x] Examined the encounter-level structure and identified repeated patient records as an evaluation concern.
- [x] Created a **patient-disjoint 80/20 train/test split** using `GroupShuffleSplit` grouped on `patient_nbr`.
- [x] Confirmed **zero overlapping patients** between training and testing.
- [x] Checked the original readmission-class distributions after splitting.
- [x] Created `readmitted_30d` in both partitions using `(readmitted == '<30').astype(int)`.
- [x] Checked the binary target prevalence in training and testing.

### Validated split

| Measure | Training | Testing |
|---|---:|---:|
| Encounters | 81,613 | 20,153 |
| Unique patients | 57,214 | 14,304 |
| Positive-class prevalence (`readmitted_30d = 1`) | 11.28% | 10.67% |
| Negative-class prevalence (`readmitted_30d = 0`) | 88.72% | 89.33% |

**Overlapping patient IDs: 0.** The groups were split by patient, not by individual encounter. The approximately 80/20 allocation is at the **patient** level; encounter counts need not follow the same ratio exactly. `GroupShuffleSplit` is not inherently stratified, so class distributions were checked afterward.

### In progress / not yet completed

- [ ] Finalize the data audit, missingness review, and data dictionary in the published notebook.
- [ ] Audit feature availability at discharge, including `discharge_disposition_id` and historical utilization fields.
- [ ] Define a patient-disjoint **validation** strategy; keep the test set untouched during model development.
- [ ] Build a leakage-safe preprocessing pipeline fitted on training data only.
- [ ] Train a majority-class/dummy baseline and logistic regression.
- [ ] Compare tree-based models and evaluate imbalanced-class performance.
- [ ] Review calibration, decision thresholds, interpretability, and subgroup performance.
- [ ] Publish verified model results, limitations, and reproducible artifacts.

**No ROC-AUC, PR-AUC, F1, accuracy, or deployment claims are reported yet.** The class prevalence percentages above describe the target distribution, not model performance.

## Evaluation plan

The early-readmission class is relatively uncommon (~11% of encounters), so raw accuracy is not sufficient. Planned measures include:

- **Average precision (PR-AUC)** and **ROC-AUC** for ranking performance.
- **Precision, recall, F1, and confusion matrices** at explicitly chosen decision thresholds.
- **Calibration** to assess how well predicted probabilities align with observed frequencies.
- **Subgroup diagnostics** and limitations to examine performance variation.

Any hyperparameter selection, feature engineering, or threshold tuning will be performed using training/validation data, **not the final test set**. Patient-level separation prevents patient-ID overlap but does not constitute chronological or external validation.

## How to run the current Colab notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload **`Healthcare_Readmission_Colab_Day1_Data_Audit.ipynb`** using **File → Upload notebook**.
3. Run its setup/download cells. The notebook retrieves the original dataset from UCI and loads it using Pandas; no Google Drive mount or GPU is required for the initial audit.
4. Work through the audit and add the subsequent patient-level split and binary-target cells to your working notebook.
5. Save/export the updated notebook before ending the Colab session (temporary runtime storage can be lost).

The shared starter notebook contains the initial audit tasks; the patient split and binary target steps were subsequently **executed in the working Colab session** and verified from output. They should be saved into the published notebook for full reproducibility.

## Next milestone

**Feature eligibility and leakage audit.** Review identifier columns, future outcomes, historical utilization fields, discharge status, missingness, and whether each candidate predictor is actually available at the intended prediction point. Only then will we create the preprocessing pipeline and first logistic regression baseline.

## Limitations and responsible use

- **Age and geography:** Data are from US hospitals in 1999–2008; findings may not generalize to current Canadian healthcare settings.
- **Repeated encounters:** Patient-disjoint splitting reduces one kind of evaluation contamination but is not a substitute for temporal validation or testing at an external hospital.
- **Label interpretation:** The outcome is a recorded readmission category, not avoidable readmission or a causal effect.
- **Feature availability:** Some administrative fields require careful review for discharge-time availability, eligibility, and potential target proxies.
- **Fairness and missingness:** Clinical, demographic, and coding differences can affect model behavior; subgroup analysis is planned, not yet completed.
- **Clinical use:** No model from this portfolio should be used for diagnosis, treatment, patient prioritization, or operational deployment without appropriate external validation and clinical governance.

## Citation

Clore, J., Cios, K., DeShazo, J. P., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
