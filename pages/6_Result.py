# OfferSense AI
# pages/6_Result.py

from datetime import datetime

import pandas as pd
import streamlit as st
from utils.page_state import activate_page

activate_page("6_Result")

st.sidebar.image("assets/logo.png", width=500)
st.sidebar.title("Navigation")
st.sidebar.write("Prediction Result Module")
st.sidebar.markdown("---")
st.title("Prediction Result")

if "candidate" not in st.session_state or "Prediction" not in st.session_state.candidate:
    st.info("No prediction available. Click Predict Offer Acceptance on the Prediction page first.")
    st.stop()

data = st.session_state.candidate

st.subheader("Candidate Information")
left_col, mid_col, right_col = st.columns(3)
with left_col:
    st.write(f"**Age:** {data.get('age')}")
    st.write(f"**Gender:** {data.get('gender')}")
    st.write(f"**Country:** {data.get('country')}")

with mid_col:
    st.write(f"**Experience:** {data.get('years_experience')} years")
    st.write(f"**Work Arrangement:** {data.get('remote_preference')}")
    st.write(f"**Relocation Willing:** {data.get('relocation_willingness')}")

with right_col:
    expected = float(data.get("expected_salary_usd", 0))
    offered = float(data.get("offered_salary_usd", 0))
    gap = float(data.get("salary_offer_gap_usd", offered - expected))
    hike = float(data.get("salary_hike_pct", (gap / expected * 100) if expected > 0 else 0))
    st.write(f"**Expected Salary:** ${expected:,.0f}")
    st.write(f"**Offered Salary:** ${offered:,.0f}")
    st.write(f"**Salary Offer Gap:** ${gap:+,.0f} ({hike:+.1f}%)")
    st.write(f"**Company:** {data.get('company_size')} | Rating: {data.get('company_rating')}/5.0")

st.divider()

prediction = data.get("Prediction_Label", "Unknown")
accept_prob = data.get("Acceptance_Probability", 0.0)
reject_prob = data.get("Rejection_Probability", 0.0)
model_used = data.get("Model_Used", "Production Pipeline")

st.markdown(f"### Final Verdict: **{prediction}**")
st.caption(f"Evaluated with: **{model_used}**")

if prediction == "Accept":
    st.success(f"Candidate is predicted to **ACCEPT** the offer with {accept_prob:.1f}% probability.")
else:
    st.error(f"Candidate is predicted to **REJECT** the offer with {reject_prob:.1f}% probability.")

probability_col, rejection_col = st.columns(2)
probability_col.metric("Acceptance Probability", f"{accept_prob:.2f}%")
rejection_col.metric("Rejection Probability", f"{reject_prob:.2f}%")

st.divider()
st.subheader("Explanation / Key Decision Drivers")
reasons = []

if gap > 0:
    reasons.append(f"Offered salary exceeds candidate expectations by ${gap:,.0f} (+{hike:.1f}%). Positive compensation advantage strongly drives acceptance.")
elif gap == 0:
    reasons.append("Offered salary exactly matches candidate expectations.")
else:
    reasons.append(f"Offered salary is below candidate expectations by ${abs(gap):,.0f} ({hike:.1f}%). Below-expectation salary significantly elevates offer rejection risk.")

comp_rating = float(data.get("company_rating", 3.0))
if comp_rating >= 4.0:
    reasons.append(f"Strong company rating ({comp_rating:.1f}/5.0) increases company appeal and trust.")
elif comp_rating < 3.0:
    reasons.append(f"Below-average company rating ({comp_rating:.1f}/5.0) weakens candidate commitment.")

exp = float(data.get("years_experience", 0))
if exp >= 5:
    reasons.append(f"Senior candidate ({exp:.1f} years experience) with multiple competitive market opportunities.")
else:
    reasons.append(f"Early-to-mid career professional ({exp:.1f} years experience).")

remote_pref = data.get("remote_preference")
if remote_pref in ["Remote", "Hybrid"]:
    reasons.append(f"Flexible work model ({remote_pref}) aligns with high-retention candidate preferences.")

for reason in reasons:
    st.write("•", reason)

st.divider()
st.subheader("Downloadable Report")
report_df = pd.DataFrame(
    [
        {
            "Prediction": prediction,
            "Acceptance_Probability": round(accept_prob, 2),
            "Rejection_Probability": round(reject_prob, 2),
            "Age": data.get("age"),
            "Gender": data.get("gender"),
            "Country": data.get("country"),
            "Experience_Years": data.get("years_experience"),
            "Expected_Salary_USD": expected,
            "Offered_Salary_USD": offered,
            "Salary_Gap_USD": gap,
            "Salary_Hike_Pct": round(hike, 2),
            "Work_Arrangement": remote_pref,
            "Company_Size": data.get("company_size"),
            "Company_Rating": data.get("company_rating"),
        }
    ]
)

st.download_button(
    label="Download Prediction Report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_report.csv",
    mime="text/csv",
)
st.caption(f"Report generated on {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
