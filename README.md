# Predicting Hospital Readmissions for Diabetic Patients

**General Assembly Data Analytics Bootcamp (DABPT03) — Capstone Project**
**Ram Jeevan | February 2026**

---

## Overview

This project builds two machine learning models to predict hospital readmissions for diabetic patients at the point of discharge. Each patient receives a risk score (0–100%) that clinical teams can use to triage care and that hospital planners can use to forecast resource needs.

- **Model 1 — Any Readmission:** Identifies patients likely to return at any point after discharge.
- **Model 2 — Early Readmission (<30 days):** Identifies patients at risk of rapid return, which often signals a care quality gap.

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Original records:** 101,766 inpatient encounters across 130 US hospitals
- **Final records after cleaning:** 99,340

---

## Results

| Metric | Any Readmission | 30-Day Readmission |
|---|---|---|
| Model | LightGBM | LightGBM (threshold=0.30) |
| AUC-ROC | 0.692 | 0.671 |
| Recall | 55.9% | 49.7% |
| F1-Score | 0.595 | 0.280 |
| Precision | 0.636 | 0.194 |

The 30-day model uses 3:1 class weighting and a tuned decision threshold of 0.30 to address severe class imbalance (11.4% positive class). This increases recall from 0% to 49.7% — catching nearly half of all early readmissions.

---

## Methodology

### Data Cleaning
- Removed columns with >83% missing values (weight, HbA1c, glucose serum)
- Removed death/hospice discharges (2,423 rows) — these patients cannot be readmitted
- Converted ICD-9 codes into 9 clinical diagnosis groups (Circulatory, Respiratory, Diabetes, etc.)
- Filtered medications to retain only those with >5% prescription rate (7 of 23)
- Replaced missing values in race, payer code, and medical specialty with informative categories

### Feature Engineering
- **28 total features** for LightGBM (10 numeric, 14 categorical, 4 binary)
- Two binary target variables: any readmission, and readmission within 30 days
- Age brackets converted to midpoint numeric values

### Models Compared
| Algorithm | Notes |
|---|---|
| Logistic Regression | 1,000 iterations; 88 one-hot features; StandardScaler |
| Random Forest | 100 trees; max_depth=10 |
| LightGBM | 500 trees; num_leaves=64; learning_rate=0.02; native categorical support |

LightGBM was selected as the final model for both targets, winning on AUC, F1, and Recall.

### Key Findings from EDA
- **Top predictors:** Number of lab procedures, number of medications, discharge disposition, time in hospital, age
- **Highest-risk discharge type:** Psychiatric (61.2% readmission rate)
- **Highest-risk diagnosis:** Diabetes as primary diagnosis (52.6% readmission rate)
- **Medication insight:** Insulin dose increases correlate with a 53% readmission rate vs 43.6% for patients not on insulin — dose instability is a stronger predictor than medication type alone

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling_any_readmission.ipynb
│   └── 04_modelling_30day_readmission.ipynb
├── data/
│   └── diabetic_data.csv          # Source: UCI ML Repository
├── outputs/
│   ├── model_any_readmission.pkl
│   └── model_30day_readmission.pkl
├── reports/
│   ├── Technical_Report.pdf
│   └── Presentation_Slides.pptx
└── README.md
```

---

## Dashboard

An interactive Tableau dashboard provides two views for clinical and planning teams:

1. **Diagnosis Heatmap** — Average predicted readmission probability by primary, secondary, and tertiary diagnosis combination. Patients with Diabetes + Genitourinary diagnoses exceed 60% probability; Injury + Musculoskeletal combinations fall below 40%.
2. **Primary Diagnosis Treemap** — Proportional view of patient volume and risk level by diagnosis group.

Filters allow teams to slice by age, race, time in hospital, predicted risk range, and actual readmission outcome.

---

## Limitations

- Data is from US hospitals (1999–2008) and may not translate directly to Singapore's healthcare context
- No weight, HbA1c, or glucose data available — all had >83% missing values
- No social determinants (housing, family support, income)
- 30-day model precision is low (19%) due to severe class imbalance
- AUC of 0.69 is meaningful but not definitive — models should support, not replace, clinical judgement

---

## Recommendations

1. Deploy both models in parallel: any-readmission for general triage, 30-day for urgent resource planning
2. Allow clinical teams to adjust the decision threshold based on ward capacity
3. Retrain on Singapore data with local features (polyclinic visits, subsidy tier, living alone status)
4. Build a feedback loop to track predicted vs actual outcomes and improve models over time

---

## Future Work

A model predicting **first admissions** (rather than readmissions) for at-risk diabetic patients would support Singapore's preventive healthcare priorities. With approximately 1 in 5 hospital admissions being diabetes-related, early risk scoring could reduce both clinical burden and patient harm.

---

## Tools & Libraries

- **Python:** pandas, scikit-learn, LightGBM, matplotlib, seaborn
- **Visualisation:** Tableau
- **Environment:** Jupyter Notebook

---

## Citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Beis, D. (2014). *Diabetes 130-US hospitals for years 1999–2008* [Data set]. UCI Machine Learning Repository.
