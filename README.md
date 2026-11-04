# OfferSense AI: Intelligent Job Offer Acceptance Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-008080.svg)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/kuldeepjangra/global-ai-hiring-analytics-dataset-2020-2026)

**OfferSense AI** is a production-grade machine learning system and multi-page web application designed for talent acquisition teams, recruitment leaders, and HR analytics specialists. It predicts whether a prospective candidate is likely to **Accept** or **Reject** an employment offer based on candidate expectations, compensation alignment, market competitiveness, and workplace preferences.

---

## Table of Contents
- [Project Overview & Business Value](#project-overview--business-value)
- [Dataset Source & Specification](#dataset-source--specification)
- [Feature Engineering & Noise Reduction](#feature-engineering--noise-reduction)
- [Machine Learning Architecture & Modeling](#machine-learning-architecture--modeling)
- [Model Performance Benchmark](#model-performance-benchmark)
- [Streamlit Web Application Modules](#streamlit-web-application-modules)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Quick Start](#installation--quick-start)
- [Tech Stack](#tech-stack)

---

## Project Overview & Business Value

In modern technical recruitment, offer reneging and declined job offers result in substantial financial losses, project timeline disruptions, and wasted recruiter bandwidth. Traditional recruitment lacks quantitative, data-backed risk assessment prior to offer rollout.

**OfferSense AI** bridges this gap by:
1. **Calibrated Offer Risk Assessment**: Calculating accurate acceptance probabilities before extending formal offer letters.
2. **Compensation Optimization**: Identifying whether salary gaps, equity, or hike percentages are competitive enough to secure target talent.
3. **Class-Balanced Predictive Modeling**: Employing positive class weighting (`scale_pos_weight=2.01`) to overcome real-world recruitment imbalances (67% rejection vs. 33% acceptance rates).
4. **Actionable HR Advisory**: Generating dynamic compensation adjustments and workplace policy recommendations tailored to individual candidate profiles.

---

## Dataset Source & Specification

The underlying data is powered by the **Global AI Hiring Analytics Dataset (2020–2026)**:

- **Source**: [Kaggle - Global AI Hiring Analytics Dataset (2020-2026)](https://www.kaggle.com/datasets/kuldeepjangra/global-ai-hiring-analytics-dataset-2020-2026)
- **Author**: Kuldeep Jangra
- **Dataset Size**: 200,000 candidate application records × 28 raw features
- **Target Variable**: `offer_accepted` (Binary Classification: `1` = Accepted, `0` = Rejected)
- **Target Distribution**: ~66.8% Rejected (`0`) vs. ~33.2% Accepted (`1`)

---

## Feature Engineering & Noise Reduction

### 1. Removal of Noisy & Irrelevant Features
To ensure high generalization, prevent overfitting, and eliminate non-predictive noise, the following 11 features (plus non-predictive identifiers) were explicitly purged from model inputs:
- `major` / Major Specialization
- `university_tier` / University Tier
- `city` / Candidate City
- `education_level` / Degree / Education Level
- `kaggle_score` / Kaggle Competitions Score
- `linkedin_connections` / LinkedIn Network Size
- `internship_count` / Prior Internship Count
- `github_repositories` / GitHub Repo Count
- `hackathons_participated` / Hackathon Count
- `leadership_experience` / Leadership Role Flag
- `company_industry` / Employer Industry
- *(Metadata dropped: `candidate_id`, `publications_count`, `application_month`, `application_year`)*

### 2. Active Clean Features (14 Core Factors)
The clean, production dataset trains on 14 predictive factors:
- **Numerical Features (9)**:
  - `age`: Candidate age (18–65)
  - `years_experience`: Total years of professional experience (0–35)
  - `expected_salary_usd`: Candidate's base expectation in USD
  - `offered_salary_usd`: Proposed offer package in USD
  - `company_rating`: Glassdoor / employer review rating (1.0–5.0)
  - `salary_offer_gap_usd`: Net dollar gap ($\text{Offered} - \text{Expected}$)
  - `offer_to_expected_salary_ratio`: Salary offer multiple ($\frac{\text{Offered}}{\text{Expected}}$)
  - `salary_hike_pct`: Relative percentage raise ($\frac{\text{Offered} - \text{Expected}}{\text{Expected}} \times 100$)
  - `exp_to_age_ratio`: Career density indicator ($\frac{\text{Years Experience}}{\text{Age}}$)
- **Categorical Features (5)**:
  - `gender`: Male, Female, Other
  - `country`: USA, India, Singapore, UK, Germany, Canada, Japan, France, Australia
  - `remote_preference`: Hybrid, Remote, Onsite
  - `relocation_willingness`: Yes, No
  - `company_size`: Startup, Small, Medium, Enterprise

---

## Machine Learning Architecture & Modeling

### 1. Solving Class Imbalance (`scale_pos_weight`)
The dataset exhibits an inherent **66.8% Rejection / 33.2% Acceptance** skew. Default classifiers prioritize the majority class (rejection), yielding low recall on accepted offers (~39%).
- We incorporated an exact positive class ratio weight:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} \approx 2.01$$
- In linear and tree-based models (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`), `class_weight='balanced'` was applied.
- This boosted accepted offer recall to **65.5%**, test accuracy to **70.2%**, and ROC-AUC to **76.0%**.

### 2. 5-Fold Stratified Cross-Validation
- Implemented `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across all models to ensure every validation fold preserves identical 66.8% / 33.2% class proportions without data leakage.

### 3. Hyperparameter Tuning (RandomizedSearchCV + GridSearchCV)
- **Stage 1 (RandomizedSearchCV)**: Coarse random exploration across `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
- **Stage 2 (GridSearchCV)**: Local exhaustive grid search around optimal parameters.

### 4. End-to-End Production Pipeline (`models/pipeline.pkl`)
- Combines a `ColumnTransformer`:
  - `StandardScaler()` on all continuous numerical inputs.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on discrete categorical attributes.
- Chained with the tuned `XGBClassifier(scale_pos_weight=2.01)`.
- Serialized to [`models/pipeline.pkl`](models/pipeline.pkl) for single-call raw inference.

---

## Model Performance Benchmark

Models evaluated on an 80/20 stratified test split (40,000 held-out samples):

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Best Model / Pipeline)** | **70.24%** | **54.31%** | **64.71%** | **59.06%** | **75.97%** |
| **Random Forest Classifier** | 70.82% | 55.39% | 61.82% | 58.43% | 75.74% |
| **Logistic Regression** | 69.91% | 54.01% | 62.36% | 57.89% | 75.01% |
| **Decision Tree Classifier** | 68.64% | 52.20% | 64.69% | 57.78% | 74.22% |

### Live Inference Validation
- **Competitive Offer** (+$15,000 hike, 4.6 employer rating, 6 yrs exp):
  - **Verdict**: **ACCEPT** (74.0% Acceptance Probability)
- **Deficit Offer** (-$20,000 deficit, 2.3 employer rating, 2 yrs exp):
  - **Verdict**: **REJECT** (82.1% Rejection Probability)

---

## Streamlit Web Application Modules

The application is structured as a multi-page interactive dashboard:

```
app.py (Home Dashboard)
├── 1_Dataset.py               -> Interactive Data Explorer & Column Schema
├── 2_Recruitment_Insights.py  -> Global Trends, Acceptance Rates & Salary Gap Visuals
├── 3_Model_Training.py        -> Stratified CV, Hyperparameter Tuning & Pipeline Architecture
├── 4_Model_Comparison.py      -> Performance Matrix (F1, Accuracy, ROC-AUC, Radar Charts)
├── 5_Prediction.py            -> Real-Time Candidate Scoring & Multi-Model Consensus
└── 6_Result.py                -> Candidate Decision Dossier & Downloadable PDF/CSV Report
```

### Key UI Features
- **One-Click Candidate Presets**: Instantly populate the form with:
  - `Sample Candidate-A (High Acceptance)`: Competitive package and strong company rating.
  - `Sample Candidate-B (Offer-at-Risk)`: Below-expectation compensation and startup risk.
  - `Reset to Default (Market Standard)`: Balanced median market profile.
- **Algorithm Switcher**: Switch real-time inference between the Production Pipeline, Standalone XGBoost, Random Forest, Decision Tree, or Logistic Regression.
- **Multi-Model Consensus Table**: View how all 5 models score the candidate side-by-side.
- **Dynamic HR Advisory**: Generates tailored compensation adjustments based on market salary gaps.

---

## Jupyter Notebooks

Located in the [`notebooks/`](notebooks/) directory:

1. [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb):
   - Loading 200,000 rows and filtering out noisy features.
   - Target distribution and class balance inspection.
   - Correlation heatmaps, salary gap distributions, and country/company-size acceptance rates.
   - Statistical feature ranking via ANOVA F-test (`SelectKBest`) and Random Forest feature importances.

2. [`notebooks/Model_Training.ipynb`](notebooks/Model_Training.ipynb):
   - Feature engineering implementation (`salary_hike_pct`, `exp_to_age_ratio`).
   - Categorical encoding and feature scaling persistence.
   - 5-Fold Stratified K-Fold cross-validation benchmarking.
   - Hyperparameter optimization using `RandomizedSearchCV` and `GridSearchCV`.
   - Construction, fitting, evaluation, and serialization of `models/pipeline.pkl`.

---

## Project Directory Structure

```text
JOB_OFFER_ACCEPTANCE_PREDICTION/
│
├── dataset/
│   └── job_offer_acceptance_dataset.csv  # Clean 200k record dataset
│
├── models/
│   ├── best_model.pkl                    # Tuned XGBoost binary
│   ├── pipeline.pkl                      # Full ColumnTransformer + XGBoost pipeline
│   ├── xgboost.pkl                       # Standalone XGBoost model
│   ├── random_forest.pkl                 # Tuned Random Forest model
│   ├── decision_tree.pkl                 # Decision Tree model
│   ├── logistic.pkl                      # Logistic Regression model
│   ├── encoders.pkl                      # Categorical LabelEncoders
│   ├── scaler.pkl                        # StandardScaler binary
│   ├── feature_names.pkl                 # Ordered active feature list
│   └── model_metrics.json                # Persisted performance metrics
│
├── notebooks/
│   ├── EDA.ipynb                         # Exploratory Data Analysis notebook
│   └── Model_Training.ipynb              # Machine learning training notebook
│
├── pages/
│   ├── 1_Dataset.py                      # Dataset summary and feature viewer
│   ├── 2_Recruitment_Insights.py         # Visual analytics & recruitment charts
│   ├── 3_Model_Training.py               # Model training & CV inspection
│   ├── 4_Model_Comparison.py             # Performance comparison dashboard
│   ├── 5_Prediction.py                   # Real-time offer acceptance prediction
│   └── 6_Result.py                       # Final verdict report & export
│
├── utils/
│   ├── data_loader.py                    # Dataset and artifact loader utilities
│   ├── model_evaluation.py               # Evaluation metrics & confusion matrices
│   ├── page_state.py                     # Streamlit multi-page state management
│   ├── prediction.py                     # HR recommendation logic
│   └── preprocessing.py                  # Domain feature engineering & encoding
│
├── assets/
│   └── logo.png                          # Brand asset
│
├── app.py                                # Main Streamlit application entry point
├── train_model.py                        # Standalone model training CLI script
├── requirements.txt                      # Project Python dependencies
└── README.md                             # Comprehensive project documentation
```

---

## Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/JOB_OFFER_ACCEPTANCE_PREDICTION.git
cd JOB_OFFER_ACCEPTANCE_PREDICTION
```

### 3. Create and Activate Virtual Environment
- **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **On macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Retrain Models & Pipeline (Optional)
To retrain models from scratch and update all serialized binaries in `models/`:
```bash
python train_model.py
```

### 6. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Tech Stack

- **Core**: Python 3.14
- **Web Interface**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, Plotly
- **Model Serialization**: Joblib

---

## Acknowledgments & Data Citation

- Dataset provided by **Kuldeep Jangra** on Kaggle: [Global AI Hiring Analytics Dataset (2020-2026)](https://www.kaggle.com/datasets/kuldeepjangra/global-ai-hiring-analytics-dataset-2020-2026).
- Built with Scikit-Learn and Streamlit open-source ecosystems.
