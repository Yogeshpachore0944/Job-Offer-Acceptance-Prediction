# OfferSense AI
# pages/5_Prediction.py

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_encoders,
    load_feature_names,
    load_model,
    load_named_model,
    load_pipeline,
    load_scaler,
    fix_sklearn_compatibility,
)
from utils.prediction import hr_recommendation
from utils.page_state import activate_page

activate_page("5_Prediction")

st.set_page_config(page_title="Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #f5f7fb; }
    .block-container { padding: 2rem 3rem; }
    .title { font-size: 38px; font-weight: bold; color: #0E4C92; margin-bottom: 5px; }
    .subtitle { color: #555; font-size: 16px; margin-bottom: 20px; }
    .template-box { background: #eef4fc; border: 1px solid #c7dcfa; padding: 12px 18px; border-radius: 8px; margin-bottom: 15px; }
    .metric-badge { padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .footer { text-align: center; color: gray; margin-top: 35px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all_resources():
    pipe = load_pipeline()
    best_m = load_model()
    encs = load_encoders()
    feats = load_feature_names()
    sc = load_scaler()
    models_dict = {
        "Production Pipeline (Tuned XGBoost - Recommended)": ("pipeline", pipe),
        "XGBoost Classifier": ("xgboost.pkl", load_named_model("xgboost.pkl")),
        "Random Forest Classifier": ("random_forest.pkl", load_named_model("random_forest.pkl")),
        "Decision Tree Classifier": ("decision_tree.pkl", load_named_model("decision_tree.pkl")),
        "Logistic Regression": ("logistic.pkl", load_named_model("logistic.pkl")),
    }
    return pipe, best_m, encs, feats, sc, models_dict


pipeline, best_model, encoders, feature_names, scaler, models_dict = load_all_resources()
pipe = pipeline

st.markdown("<div class='title'>Job Offer Acceptance Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict whether a candidate is likely to accept or decline a job offer using calibrated ML models.</div>",
    unsafe_allow_html=True,
)

# Preset candidate profiles
accept_demo = {
    "age": 30,
    "gender": "Male",
    "country": "USA",
    "years_experience": 6.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 100000,
    "remote_preference": "Hybrid",
    "relocation_willingness": "Yes",
    "company_size": "Enterprise",
    "company_rating": 4.6,
}

reject_demo = {
    "age": 25,
    "gender": "Female",
    "country": "India",
    "years_experience": 2.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 65000,
    "remote_preference": "Onsite",
    "relocation_willingness": "No",
    "company_size": "Startup",
    "company_rating": 2.3,
}

default_candidate = {
    "age": 28,
    "gender": "Male",
    "country": "USA",
    "years_experience": 4.0,
    "expected_salary_usd": 85000,
    "offered_salary_usd": 95000,
    "remote_preference": "Hybrid",
    "relocation_willingness": "Yes",
    "company_size": "Enterprise",
    "company_rating": 4.2,
}


def apply_candidate_data(data):
    st.session_state["pred_age"] = int(data["age"])
    st.session_state["pred_gender"] = str(data["gender"])
    st.session_state["pred_country"] = str(data["country"])
    st.session_state["pred_experience"] = float(data["years_experience"])
    st.session_state["pred_expected_salary"] = int(data["expected_salary_usd"])
    st.session_state["pred_offered_salary"] = int(data["offered_salary_usd"])
    st.session_state["pred_remote"] = str(data["remote_preference"])
    st.session_state["pred_relocation"] = str(data["relocation_willingness"])
    st.session_state["pred_company_size"] = str(data["company_size"])
    st.session_state["pred_company_rating"] = float(data["company_rating"])


# Initialize default session state if not set
if "pred_age" not in st.session_state:
    apply_candidate_data(default_candidate)

with st.sidebar:
    st.image("assets/logo.png", width=500)
    st.title("Navigation")
    st.write("Prediction Module")
    st.markdown("---")
    st.write("### Model Options")
    selected_model_label = st.selectbox(
        "Choose Prediction Algorithm",
        list(models_dict.keys()),
        index=0,
        key="selected_model_option",
    )
    st.markdown("---")
    st.write("### Active Features (14)")
    st.caption("- Candidate Age & Years Experience")
    st.caption("- Expected & Offered Salary ($)")
    st.caption("- Salary Offer Gap & Hike %")
    st.caption("- Experience-to-Age Ratio")
    st.caption("- Work Arrangement & Relocation")
    st.caption("- Company Size & Company Rating")

# Template profile selectors
st.markdown("### Profile Presets & Quick Fill")
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button("Sample Candidate-A (High Acceptance)", use_container_width=True, type="secondary"):
        apply_candidate_data(accept_demo)
        st.session_state["prediction_demo"] = "Sample Candidate-A (High Acceptance Profile)"
        st.rerun()

with btn_col2:
    if st.button("Sample Candidate-B (Offer-at-Risk)", use_container_width=True, type="secondary"):
        apply_candidate_data(reject_demo)
        st.session_state["prediction_demo"] = "Sample Candidate-B (Offer-at-Risk Profile)"
        st.rerun()

with btn_col3:
    if st.button("Reset to Default (Market Standard)", use_container_width=True, type="secondary"):
        apply_candidate_data(default_candidate)
        st.session_state["prediction_demo"] = "Custom Input"
        st.rerun()


def on_template_change():
    selected = st.session_state.get("prediction_demo")
    if selected == "Sample Candidate-A (High Acceptance Profile)":
        apply_candidate_data(accept_demo)
    elif selected == "Sample Candidate-B (Offer-at-Risk Profile)":
        apply_candidate_data(reject_demo)
    elif selected == "Sample Candidate-C (Market Standard Profile)":
        apply_candidate_data(default_candidate)


template_options = [
    "Custom Input",
    "Sample Candidate-A (High Acceptance Profile)",
    "Sample Candidate-B (Offer-at-Risk Profile)",
    "Sample Candidate-C (Market Standard Profile)",
]

st.selectbox(
    "Or Choose Candidate Profile Template Dropdown",
    template_options,
    key="prediction_demo",
    on_change=on_template_change,
)

gender_options = ["Male", "Female", "Other"]
country_options = ["USA", "India", "Singapore", "UK", "Germany", "Canada", "Japan", "France", "Australia"]
remote_options = ["Hybrid", "Remote", "Onsite"]
reloc_options = ["Yes", "No"]
size_options = ["Enterprise", "Medium", "Small", "Startup"]

# Form inputs
st.markdown("### 1. Candidate Background")
bg_col1, bg_col2, bg_col3, bg_col4 = st.columns(4)

with bg_col1:
    age = st.number_input("Age", min_value=18, max_value=65, step=1, key="pred_age")

with bg_col2:
    gender = st.selectbox("Gender", gender_options, key="pred_gender")

with bg_col3:
    country = st.selectbox("Country", country_options, key="pred_country")

with bg_col4:
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=35.0, step=0.5, key="pred_experience")

st.markdown("### 2. Compensation & Salary Alignment")
comp_col1, comp_col2, comp_col3 = st.columns(3)

with comp_col1:
    expected_salary = st.number_input("Expected Salary ($)", min_value=20000, max_value=300000, step=5000, key="pred_expected_salary")

with comp_col2:
    offered_salary = st.number_input("Offered Salary ($)", min_value=20000, max_value=350000, step=5000, key="pred_offered_salary")

with comp_col3:
    salary_gap = offered_salary - expected_salary
    hike_pct = (salary_gap / expected_salary * 100) if expected_salary > 0 else 0
    st.metric(
        "Offer Difference vs Expectation",
        f"${salary_gap:+,}",
        f"{hike_pct:+.1f}%",
        delta_color="normal" if salary_gap >= 0 else "inverse"
    )

st.markdown("### 3. Workplace & Company Conditions")
work_col1, work_col2, work_col3, work_col4 = st.columns(4)

with work_col1:
    remote_pref = st.selectbox("Work Arrangement", remote_options, key="pred_remote")

with work_col2:
    relocation = st.selectbox("Relocation Willingness", reloc_options, key="pred_relocation")

with work_col3:
    company_size = st.selectbox("Company Size", size_options, key="pred_company_size")

with work_col4:
    company_rating = st.slider("Company Rating", min_value=1.0, max_value=5.0, step=0.1, key="pred_company_rating")

st.markdown("---")

predict = st.button("Predict Offer Acceptance", type="primary", use_container_width=True)

if predict:
    salary_offer_gap = offered_salary - expected_salary
    offer_ratio = offered_salary / expected_salary if expected_salary > 0 else 1.0
    hike_percentage = (salary_offer_gap / expected_salary * 100) if expected_salary > 0 else 0.0
    exp_age_ratio = experience / age if age > 0 else 0.0

    raw_input_df = pd.DataFrame([{
        "age": int(age),
        "gender": str(gender),
        "country": str(country),
        "years_experience": float(experience),
        "expected_salary_usd": int(expected_salary),
        "offered_salary_usd": int(offered_salary),
        "remote_preference": str(remote_pref),
        "relocation_willingness": str(relocation),
        "company_size": str(company_size),
        "company_rating": float(company_rating),
        "salary_offer_gap_usd": int(salary_offer_gap),
        "offer_to_expected_salary_ratio": float(offer_ratio),
        "salary_hike_pct": float(hike_percentage),
        "exp_to_age_ratio": float(exp_age_ratio),
    }])

    model_kind, model_obj = models_dict[selected_model_label]

    try:
        with st.spinner(f"Scoring with {selected_model_label}..."):
            fix_sklearn_compatibility(model_obj)
            if model_kind == "pipeline":
                pred = int(model_obj.predict(raw_input_df)[0])
                probs = model_obj.predict_proba(raw_input_df)[0]
            else:
                # Standalone models were trained on preprocessor-transformed features
                preprocessor = pipeline.named_steps["preprocessor"]
                fix_sklearn_compatibility(preprocessor)
                X_proc = preprocessor.transform(raw_input_df)
                pred = int(model_obj.predict(X_proc)[0])
                probs = model_obj.predict_proba(X_proc)[0]

            reject_prob = float(probs[0] * 100)
            accept_prob = float(probs[1] * 100)
            label = "Accept" if pred == 1 else "Reject"
            confidence = accept_prob if pred == 1 else reject_prob
            recommendation = hr_recommendation(pred, accept_prob)

        st.session_state.candidate = {
            "age": int(age),
            "gender": str(gender),
            "country": str(country),
            "years_experience": float(experience),
            "expected_salary_usd": float(expected_salary),
            "offered_salary_usd": float(offered_salary),
            "salary_offer_gap_usd": float(salary_offer_gap),
            "salary_hike_pct": float(hike_percentage),
            "company_rating": float(company_rating),
            "remote_preference": str(remote_pref),
            "relocation_willingness": str(relocation),
            "company_size": str(company_size),
            "Model_Used": str(selected_model_label),
            "Prediction": int(pred),
            "Prediction_Label": str(label),
            "Acceptance_Probability": float(accept_prob),
            "Rejection_Probability": float(reject_prob),
            "Prediction_Confidence": float(confidence),
            "HR_Recommendation": str(recommendation),
        }

        st.success(f"Prediction generated successfully using **{selected_model_label}**!")
        st.subheader("Prediction Result")
        result_col, accept_col, reject_col = st.columns(3)
        if label == "Accept":
            result_col.success(f"### ACCEPT ({accept_prob:.1f}%)")
        else:
            result_col.error(f"### REJECT ({reject_prob:.1f}%)")
        accept_col.metric("Acceptance Probability", f"{accept_prob:.2f}%")
        reject_col.metric("Rejection Probability", f"{reject_prob:.2f}%")

        st.subheader("Confidence Gauge")
        st.progress(float(confidence) / 100)
        st.write(f"Algorithm Confidence: **{confidence:.2f}%**")

        st.subheader("HR Advisory Recommendation")
        st.info(recommendation)

        # Multi-model cross-comparison on this exact candidate
        with st.expander("Compare Predictions Across All 5 Model Algorithms on This Candidate", expanded=True):
            comparison_rows = []
            preprocessor = pipeline.named_steps["preprocessor"]
            fix_sklearn_compatibility(preprocessor)
            X_proc_comp = preprocessor.transform(raw_input_df)

            for m_lbl, (m_type, m_inst) in models_dict.items():
                fix_sklearn_compatibility(m_inst)
                if m_type == "pipeline":
                    m_p = int(m_inst.predict(raw_input_df)[0])
                    m_prob = m_inst.predict_proba(raw_input_df)[0]
                else:
                    m_p = int(m_inst.predict(X_proc_comp)[0])
                    m_prob = m_inst.predict_proba(X_proc_comp)[0]

                comparison_rows.append({
                    "Algorithm": m_lbl,
                    "Verdict": "ACCEPT" if m_p == 1 else "REJECT",
                    "Acceptance Probability (%)": round(float(m_prob[1] * 100), 2),
                    "Rejection Probability (%)": round(float(m_prob[0] * 100), 2),
                })

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

st.markdown("---")
st.markdown(
    """
    <div class='footer'>
        <h4>OfferSense AI</h4>
        <p>Machine Learning Based Job Offer Acceptance Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)
