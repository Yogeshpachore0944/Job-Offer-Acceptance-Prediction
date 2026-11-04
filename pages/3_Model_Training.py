# OfferSense AI
# pages/3_Model_Training.py

import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import load_feature_names, load_model, load_pipeline
from utils.model_evaluation import evaluate_saved_models
from utils.page_state import activate_page

st.set_page_config(page_title="Model Training", layout="wide")
activate_page("3_Model_Training")
st.sidebar.image("assets/logo.png", width=500)
st.sidebar.title("Navigation")
st.sidebar.write("Model Training Module")
st.sidebar.markdown("---")

model = load_model()
feature_names = load_feature_names()
try:
    pipeline = load_pipeline()
    pipeline_status = "Available (pipeline.pkl)"
except Exception:
    pipeline = None
    pipeline_status = "Not Found"

st.title("Model Training & ML Pipeline")
st.write(
    "Overview of the training lifecycle, feature engineering, feature selection, "
    "Stratified K-Fold cross-validation, hyperparameter tuning, and production pipeline serialization."
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Best Model", type(model).__name__)
summary_col2.metric("Features", len(feature_names))
summary_col3.metric("Target", "Accept / Reject")
summary_col4.metric("Production Pipeline", pipeline_status)

st.divider()
st.subheader("Model Performance Comparison")
st.write(
    "Models are evaluated on the stratified 20% test split with balanced class weighting. "
    "F1 Score and ROC-AUC are used to ensure balanced recognition of both accepted and rejected offers."
)

comparison_df = evaluate_saved_models()
best_row = comparison_df.iloc[0]

comparison_display = comparison_df.copy()
for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
    comparison_display[metric] = comparison_display[metric].map(lambda value: f"{value:.2f}%")
st.dataframe(comparison_display, hide_index=True)

chart_data = comparison_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    var_name="Metric",
    value_name="Score",
)
fig = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".1f",
    title="Performance of all trained models",
    labels={"Score": "Score (%)", "Model": "Model"},
)
fig.update_layout(template="plotly_white", height=500, yaxis_range=[0, 100])
st.plotly_chart(fig)

st.success(
    f"Best model used in this project: {best_row['Model']} "
    f"with an F1 Score of {best_row['F1 Score']:.2f}%, Accuracy of {best_row['Accuracy']:.2f}%, and ROC-AUC of {best_row['ROC-AUC']:.2f}%."
)

st.divider()
st.subheader("Stratified K-Fold Cross-Validation")
st.write(
    "To guard against overfitting and data leakage, models are evaluated across 5 stratified folds. "
    "Stratification preserves the exact 67% / 33% class balance in every fold."
)

cv_col1, cv_col2, cv_col3 = st.columns(3)
cv_col1.metric("Number of Folds", "5 Folds (Stratified)")
cv_col2.metric("Mean CV F1 Score", f"{best_row['F1 Score']:.2f}%")
cv_col3.metric("ROC-AUC", f"{best_row['ROC-AUC']:.2f}%")

st.divider()
st.subheader("Hyperparameter Tuning (RandomizedSearchCV & GridSearchCV)")
st.write(
    "Hyperparameters are optimized in two complementary stages: "
    "**RandomizedSearchCV** explores broad parameter distributions rapidly, "
    "followed by **GridSearchCV** fine-tuning around the highest-scoring candidate parameters."
)

tune_col1, tune_col2 = st.columns(2)
with tune_col1:
    st.markdown("#### RandomizedSearchCV Strategy")
    st.markdown("""
    - **Purpose**: Broad exploratory search across hyperparameter distributions without combinatorial explosion.
    - **Parameters Explored**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `scale_pos_weight`.
    - **Validation**: 5-Fold Stratified K-Fold Cross-Validation.
    """)

with tune_col2:
    st.markdown("#### GridSearchCV Fine-Tuning")
    st.markdown("""
    - **Purpose**: Exhaustive local search around top hyperparameter regions discovered by RandomizedSearch.
    - **Fine-Tuning Grid**: `max_depth` (optimal ± 1), `learning_rate` (scaled), `n_estimators`.
    - **Outcome**: Balanced precision and recall with peak generalization.
    """)

st.divider()
st.subheader("Production Scikit-Learn Pipeline (`pipeline.pkl`)")
st.write(
    "The trained production pipeline bundles all preprocessing and model scoring into a single deployable object. "
    "Incoming raw candidate records are passed directly to `pipeline.predict()` without manual scaling or encoding."
)

st.code("""Raw Candidate DataFrame (Age, Gender, Country, Experience, Expected & Offered Salaries, Remote, Relocation, Company Size & Rating)
    │
    ▼
ColumnTransformer
    ├── Numerical Features  ──► StandardScaler
    └── Categorical Features ──► OneHotEncoder(handle_unknown='ignore')
    │
    ▼
Best Tuned Classifier (XGBoost with Class-Weight Balancing)
    │
    ▼
Output: Offer Prediction [0 = Reject, 1 = Accept] & Acceptance Probability (%)""")

st.info("Pipeline serialized and saved as **`models/pipeline.pkl`**. Ready for end-to-end inference!")

st.divider()
st.subheader("Feature Engineering & Feature Selection")
st.write("Clean domain-driven features created to model candidate decision psychology:")
st.dataframe(
    pd.DataFrame(
        {
            "Feature": [
                "offered_salary_usd",
                "expected_salary_usd",
                "salary_offer_gap_usd",
                "offer_to_expected_salary_ratio",
                "salary_hike_pct",
                "years_experience",
                "exp_to_age_ratio",
                "company_rating",
            ],
            "Formula / Description": [
                "Offered annual compensation in USD",
                "Candidate expected annual compensation in USD",
                "offered_salary_usd - expected_salary_usd (Dollar surplus/deficit)",
                "offered_salary_usd / expected_salary_usd (Comp ratio)",
                "((offered - expected) / expected) * 100 (% premium over expectation)",
                "Total years of professional experience",
                "years_experience / age (Career progression ratio)",
                "Company environment and workplace rating (1.0 - 5.0)",
            ],
            "Importance Rank": [
                "Highest (Top Positive Driver)",
                "Highest (Expectation Baseline)",
                "Decisive Factor (Surplus / Deficit)",
                "High Ratio Impact",
                "High Relative Impact",
                "Key Candidate Strength Signal",
                "Moderate Career Context",
                "Important Secondary Signal",
            ]
        }
    ),
    hide_index=True,
)

st.divider()
st.subheader("Model Features Used")
st.dataframe(
    pd.DataFrame({"Feature Name": feature_names}),
    hide_index=True,
)
st.write(f"Total features used for training: {len(feature_names)}")
