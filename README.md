# ⚕️ ChronicML — Chronic Condition Classifier on EHR Data

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An end-to-end machine learning pipeline and interactive Streamlit dashboard for classifying chronic medical conditions from **Synthea™ synthetic EHR data** — with temporal distribution shift detection, hyperparameter tuning, and continual learning adaptation.

---

## Overview

This project builds a complete ML pipeline on Electronic Health Record (EHR) data to predict whether a clinical encounter involves a chronic condition. The pipeline handles multi-table data merging, feature engineering from vitals and patient history, model training across three algorithm families, and cross-temporal evaluation — all surfaced through an interactive Streamlit dashboard with 12 analysis pages.

---

## Problem Statement

EHR datasets grow over time, and models trained on historical data often degrade as patient populations and clinical practices shift. This project addresses that problem by:

1. **Framing** chronic condition presence at each encounter as a binary classification task
2. **Splitting** data temporally (pre vs. post January 2015) to simulate real-world deployment drift
3. **Measuring** how each model family degrades on the later time period
4. **Adapting** models via weighted fine-tuning on post-2015 data and comparing performance before and after

---

## Features Implemented

- **Data ingestion**: Merges 7 required CSVs (patients, encounters, observations, conditions, medications, allergies, procedures) with chunked reading for large files
- **Feature engineering**: Pivots observation vitals per encounter, computes per-patient history aggregates (number of prior encounters, conditions, medications, allergies, procedures), one-hot encodes demographics and encounter class, and computes age at encounter
- **Target labelling**: Keyword matching on condition descriptions and encounter reason fields to create a binary chronic/non-chronic label
- **Temporal split**: Divides data at `2015-01-01` into D1 (pre-2015) and D2 (post-2015) for drift evaluation
- **Preprocessing**: Median imputation on D1, zero-variance feature removal, StandardScaler fit on D1 only
- **Model training**: 4 Decision Trees (depth 5/10/15/20), 4 SVM-RBF variants (C = 0.1/1/10/100), 2 MLP architectures (64-32, 128-64-32) — 10 models total
- **Hyperparameter tuning**: GridSearchCV with 3-fold StratifiedKFold on each model family; SVM and MLP subsampled for tractability
- **Cross-dataset evaluation**: Each model evaluated on both D1 test and D2 test sets; performance gap (D1 − D2) computed per model
- **Continual learning**: Best model per family re-trained on weighted combination of D1 + D2 training data (D2 weighted ×2); MLP uses warm-start fine-tuning
- **Anomaly detection**: Isolation Forest on the feature matrix; anomaly score distribution plotted
- **Temporal drift analysis**: KS-test per feature comparing D1 vs. D2 distributions; PSI (Population Stability Index) computed for top features
- **Dimensionality reduction**: PCA (2D and 3D) and t-SNE projections of the feature space, coloured by class and by temporal split
- **Feature importance**: Decision tree feature importances ranked and plotted
- **Bias-variance analysis**: Training vs. test accuracy curves across model complexity (tree depth / C / layers)
- **Precision-Recall analysis**: PR curves and average precision scores for all models on both datasets
- **PDF report generation**: Downloadable multi-page summary report with model tables and drift findings
- **Dashboard**: 12-page Streamlit interface with interactive Plotly charts and static Matplotlib plots

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Dashboard framework and UI |
| `pandas` | Data loading, merging, feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, tuning, evaluation, PCA, t-SNE, Isolation Forest |
| `scipy` | KS-test for drift detection, chi-square tests |
| `plotly` | Interactive charts in the dashboard |
| `matplotlib` / `seaborn` | Static plots and EDA figures |
| `pickle` | Saving trained models to disk |

---

## Dataset Information

**Source**: [Synthea™](https://synthetichealth.github.io/synthea/) — synthetic patient generator by The MITRE Corporation. No real patient data is used.

### Required Files (7)

| File | Content |
|---|---|
| `patients.csv` | Demographics: age, gender, race, income, healthcare costs |
| `encounters.csv` | Clinical visits with timestamps, class, cost, reason |
| `observations.csv` | Vitals and lab results (large — read in 100k-row chunks) |
| `conditions.csv` | Diagnosed conditions per encounter |
| `medications.csv` | Prescriptions per patient |
| `allergies.csv` | Allergy records |
| `procedures.csv` | Medical procedures |

### Optional Files (improve feature coverage)

`careplans.csv`, `immunizations.csv`, `devices.csv`, `payer_transitions.csv`

> ⚠️ File names must match exactly (all lowercase, `.csv`). The pipeline checks for each file by name and shows a status indicator in the sidebar.

### Features Used

**Vitals / Labs** (from `observations.csv`): body height, weight, BMI, systolic BP, diastolic BP, heart rate, respiratory rate, pain severity, glucose, creatinine, calcium, sodium, potassium, chloride, CO2, QALY, DALY, GFR

**History** (computed): number of prior encounters, conditions, medications, allergies, procedures, encounter duration (minutes)

**Demographics**: age at encounter, gender, race (one-hot), ethnicity, marital status, income, healthcare expenses and coverage, encounter class (one-hot)

---

## Dashboard Structure

The dashboard is organized into four tab groups:

### 🗂️ Data
| Page | Contents |
|---|---|
| 🏠 Home | Pipeline overview and step-by-step guide |
| 📊 Data Overview | Class distribution, feature statistics, encounter class breakdown |
| 🔬 EDA | Feature distributions, correlation heatmap, temporal trends |

### 🧠 Models
| Page | Contents |
|---|---|
| 🤖 Model Performance | Accuracy, F1, ROC-AUC, confusion matrices, ROC curves for all 10 models on D1 and D2 |
| ⚙️ Hyperparameter Tuning | GridSearchCV results, best parameters, cross-validation score tables |
| 📐 Bias-Variance | Training vs. validation accuracy curves across model complexity |
| 🔍 Feature Representation | PCA 2D/3D and t-SNE projections of the feature space |
| 🌟 Feature Importance | Top-K feature importances from Decision Tree |

### 📉 Drift & Adaptation
| Page | Contents |
|---|---|
| 📉 Temporal Shift | Per-feature KS-test results, PSI scores, D1 vs. D2 distribution comparisons |
| 🔴 Anomaly Detection | Isolation Forest outlier flagging and anomaly score distribution |
| 🔄 Continual Learning | Before vs. after fine-tuning metrics for DT, SVM, MLP on D2 test set |

### 📝 Summary
| Page | Contents |
|---|---|
| 📝 Summary | Full findings, model ranking, key insights, downloadable PDF report |

---

## Project Structure

```
chronic-condition-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── dashboard.py          ← Single-file Streamlit application (~4,300 lines)
│
├── data/
│   ├── patients.csv
│   ├── encounters.csv
│   ├── observations.csv
│   ├── conditions.csv
│   ├── medications.csv
│   ├── allergies.csv
│   └── procedures.csv    (and optional CSVs)
│
└── assets/
    └── screenshots/      ← Dashboard screenshots
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/chronic-condition-classifier.git
cd chronic-condition-classifier

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the dashboard
# The --server.maxUploadSize flag is required if uploading observations.csv (>200 MB)
streamlit run dashboard.py --server.maxUploadSize=2048
```

Then in the browser:
1. Use the sidebar file uploader to upload all required CSVs
2. The sidebar shows a green ✓ for each detected file
3. Click **▶ Run ML Pipeline** — takes 5–15 minutes depending on dataset size
4. A live progress checklist shows which stage is running
5. Navigate the 12 dashboard pages when the pipeline completes

---

## Pipeline Configuration

| Parameter | Value |
|---|---|
| Temporal split date | `2015-01-01` |
| Train / test split | 80% / 20% (stratified) |
| Models trained | 4 Decision Trees, 4 SVM-RBF, 2 MLPs (10 total) |
| Hyperparameter search | GridSearchCV, 3-fold StratifiedKFold, scored by macro-F1 |
| SVM training cap | 15,000 rows (stratified subsample for speed) |
| Continual learning | Weighted refit — D2 samples weighted ×2; MLP uses warm-start |
| Anomaly detection | Isolation Forest |
| Dimensionality reduction | PCA, t-SNE |

---

## Results / Insights

> Results vary by dataset size and random seed. The following reflects typical behaviour on the Massachusetts Synthea cohort.

- Decision Trees at moderate depth (10–15) consistently outperform very shallow or very deep trees on both D1 and D2 test sets
- All models show a measurable **performance gap** between D1 (in-distribution) and D2 (post-2015) test sets, confirming temporal distribution shift
- KS-test and PSI analysis identify **lab values and cost features** as the most drifted features over time
- Continual learning (weighted fine-tuning on D2) recovers a portion of the performance gap for all three model families
- Isolation Forest flags a small percentage of encounters as anomalous, concentrated in extreme lab value ranges

---

## Screenshots

| View | Preview |
|---|---|
| Home / Pipeline Overview | `assets/screenshots/home.png` |
| Class Distribution | `assets/screenshots/data_overview.png` |
| ROC Curves — All Models | `assets/screenshots/model_performance.png` |
| Hyperparameter Tuning Grid | `assets/screenshots/hptuning.png` |
| Temporal Drift (KS-test) | `assets/screenshots/temporal_shift.png` |
| Continual Learning Results | `assets/screenshots/continual_learning.png` |
| PCA / t-SNE Projections | `assets/screenshots/feature_representation.png` |

---

## Skills Demonstrated

- **Machine Learning**: Multi-family classifier training (DT, SVM-RBF, MLP), GridSearchCV hyperparameter tuning, cross-validation, bias-variance analysis, model persistence with pickle
- **Data Engineering**: Multi-table EHR data merging, chunked CSV reading for large files, feature engineering from raw clinical records, one-hot encoding, median imputation, StandardScaler
- **Distribution Shift & Adaptation**: Temporal train/test splitting, KS-test for feature drift, PSI computation, continual learning via weighted refit and warm-start fine-tuning
- **Anomaly Detection**: Isolation Forest on tabular clinical data
- **Visualization**: Interactive Plotly charts, Streamlit multi-page dashboard, Matplotlib/Seaborn EDA, PCA and t-SNE projections, PDF report generation
- **Domain Knowledge**: EHR data structures, clinical coding, chronic condition classification, payer and utilization features

---

## Future Improvements

- Add SHAP values for model explainability (currently feature importances from DT only)
- Support online / incremental learning with `SGDClassifier` (imported but not fully integrated)
- Add a calibration plot to assess probability reliability for each model
- Allow user-configurable temporal split date via the sidebar
- Cache pipeline results to disk so the dashboard can reload without re-running

---

## Contributors

| Name | ID |
|---|---|
| Harshal Shah | 2023A7PS0055H |
| Marmik Sapovadia | 2023A7PS0057H |
| Riya Doshi | 2023AAPS0210H |
| Archit Khandelwal | 2023AAPS0184H |

*BITS Pilani · BITS F464 Machine Learning · Assignment 2*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Dataset generated by [Synthea™](https://synthetichealth.github.io/synthea/) (The MITRE Corporation). No real patient data is used.
