import streamlit as st


ATS_STATE_KEYS = (
    "ats_resume_file",
    "resume_text",
    "ats_result",
    "ats_verdict",
)

PREDICTION_STATE_KEYS = (
    "candidate",
    "prediction_demo",
    "pred_age",
    "pred_gender",
    "pred_country",
    "pred_experience",
    "pred_expected_salary",
    "pred_offered_salary",
    "pred_remote",
    "pred_relocation",
    "pred_company_size",
    "pred_company_rating",
)


def activate_page(page_name):
    previous_page = st.session_state.get("_active_page")

    if previous_page != page_name and page_name != "7_ATS_Score":
        for key in ATS_STATE_KEYS:
            st.session_state.pop(key, None)

    if previous_page != page_name and page_name == "7_ATS_Score":
        for key in ATS_STATE_KEYS:
            st.session_state.pop(key, None)

    if previous_page != page_name and page_name == "5_Prediction":
        for key in PREDICTION_STATE_KEYS:
            st.session_state.pop(key, None)

    if previous_page != page_name and page_name not in {"5_Prediction", "6_Result"}:
        for key in PREDICTION_STATE_KEYS:
            st.session_state.pop(key, None)

    st.session_state["_active_page"] = page_name
