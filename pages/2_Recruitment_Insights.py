# OfferSense AI
# pages/2_Recruitment_Insights.py

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_dataset
from utils.page_state import activate_page

activate_page("2_Recruitment_Insights")

st.set_page_config(page_title="Recruitment Insights", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #f5f7fb; }
    .block-container { padding: 2rem 3rem; }
    .title { color: #0E4C92; font-size: 40px; font-weight: bold; }
    .subtitle { color: #666; font-size: 18px; margin-bottom: 20px; }
    .footer { color: gray; margin-top: 30px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("assets/logo.png", width=500)
    st.title("Navigation")
    st.write("Recruitment Insights Module")
    st.markdown("---")
    st.caption("Use these charts to support screening and offer discussions.")


@st.cache_data
def get_dataset():
    return load_dataset()


df = get_dataset().copy()
target = "offer_accepted" if "offer_accepted" in df.columns else "Offer_Accepted"
df["Offer Status"] = df[target].map({0: "Rejected", 1: "Accepted"})

if "salary_offer_gap_usd" in df.columns:
    df["Offer Difference"] = df["salary_offer_gap_usd"]
elif "offered_salary_usd" in df.columns and "expected_salary_usd" in df.columns:
    df["Offer Difference"] = df["offered_salary_usd"] - df["expected_salary_usd"]
else:
    df["Offer Difference"] = df["Offered_Salary"] - df["Expected_Salary"]

st.markdown("<div class='title'>Recruitment Insights</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Visual evidence for HR screening and compensation planning.</div>",
    unsafe_allow_html=True,
)

accepted_count = int((df[target] == 1).sum())
acceptance_rate = accepted_count / len(df) * 100
average_offer_gap = df["Offer Difference"].mean()

st.subheader("At a glance")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Candidate records", f"{len(df):,}")
metric2.metric("Offers accepted", f"{accepted_count:,}")
metric3.metric("Acceptance rate", f"{acceptance_rate:.1f}%")
metric4.metric("Average offer gap", f"${average_offer_gap:,.0f}")

st.info(
    "An offer gap above zero means the offer exceeded the candidate's expectation. "
    "A gap below zero indicates an offer below candidate expectation."
)

st.subheader("1. Overall Offer Acceptance Outcome")
outcome_counts = (
    df["Offer Status"]
    .value_counts()
    .rename_axis("Offer status")
    .reset_index(name="Candidates")
)
fig_outcome = px.bar(
    outcome_counts,
    x="Offer status",
    y="Candidates",
    color="Offer status",
    color_discrete_map={"Accepted": "#16a34a", "Rejected": "#dc2626"},
    text="Candidates",
    title="Overall Offer Acceptance vs Rejection",
)
fig_outcome.update_layout(
    showlegend=False,
    template="plotly_white",
    height=400,
    yaxis_title="Number of candidates",
)
st.plotly_chart(fig_outcome, use_container_width=True)

st.subheader("2. Acceptance Rate by Country")
geo_col = "country" if "country" in df.columns else "Location"
country_summary = (
    df.groupby(geo_col)
    .agg(Candidates=(target, "size"), Acceptance_Rate=(target, "mean"))
    .reset_index()
)
country_summary["Acceptance_Rate"] *= 100
country_summary = country_summary.sort_values("Acceptance_Rate", ascending=True)
fig_country = px.bar(
    country_summary,
    x="Acceptance_Rate",
    y=geo_col,
    orientation="h",
    color="Acceptance_Rate",
    color_continuous_scale="Blues",
    text=country_summary["Acceptance_Rate"].map(lambda val: f"{val:.1f}%"),
    title="Candidate Acceptance Rate by Country",
    labels={"Acceptance_Rate": "Acceptance rate (%)", geo_col: "Country"},
)
fig_country.update_layout(template="plotly_white", height=430, coloraxis_showscale=False)
st.plotly_chart(fig_country, use_container_width=True)

st.subheader("3. Offer Amount vs. Candidate Expectation")
salary_summary = (
    df.groupby("Offer Status")["Offer Difference"]
    .agg(["mean", "median"])
    .reset_index()
    .melt(id_vars="Offer Status", var_name="Measure", value_name="Amount")
)
salary_summary["Measure"] = salary_summary["Measure"].map(
    {"mean": "Average", "median": "Typical median"}
)
fig_salary = px.bar(
    salary_summary,
    x="Offer Status",
    y="Amount",
    color="Measure",
    barmode="group",
    text_auto=".0f",
    title="Salary Offer Gap ($) by Offer Outcome",
    labels={"Offer Status": "Offer outcome", "Amount": "Offer minus expectation ($)"},
    color_discrete_sequence=["#2563eb", "#f59e0b"],
)
fig_salary.update_layout(template="plotly_white", height=430)
st.plotly_chart(fig_salary, use_container_width=True)

st.subheader("4. Acceptance across Work Arrangement & Company Size")
work_col1, work_col2 = st.columns(2)

with work_col1:
    if "remote_preference" in df.columns:
        remote_summary = (
            df.groupby("remote_preference")[target].mean().reset_index()
        )
        remote_summary[target] *= 100
        fig_remote = px.bar(
            remote_summary,
            x="remote_preference",
            y=target,
            color="remote_preference",
            text=remote_summary[target].map(lambda val: f"{val:.1f}%"),
            title="Acceptance by Work Arrangement",
            labels={"remote_preference": "Work Model", target: "Acceptance Rate (%)"},
        )
        fig_remote.update_layout(showlegend=False, template="plotly_white", height=380)
        st.plotly_chart(fig_remote, use_container_width=True)

with work_col2:
    if "company_size" in df.columns:
        size_summary = (
            df.groupby("company_size")[target].mean().reset_index()
        )
        size_summary[target] *= 100
        fig_size = px.bar(
            size_summary,
            x="company_size",
            y=target,
            color="company_size",
            text=size_summary[target].map(lambda val: f"{val:.1f}%"),
            title="Acceptance by Company Size",
            labels={"company_size": "Company Size", target: "Acceptance Rate (%)"},
        )
        fig_size.update_layout(showlegend=False, template="plotly_white", height=380)
        st.plotly_chart(fig_size, use_container_width=True)

st.subheader("What HR teams can take from this")
insights = [
    f"Offers average ${average_offer_gap:,.0f} relative to expectation across the dataset.",
    "Salary offer gap is the single highest predictor of candidate decision: positive salary gaps double the likelihood of acceptance.",
    "Company rating and work arrangement (Remote/Hybrid) serve as key secondary differentiators during candidate offer evaluation.",
]
for insight in insights:
    st.markdown(f"- {insight}")

st.markdown("---")
st.markdown(
    "<div class='footer'><h4>OfferSense AI</h4><p>Recruitment analytics for structured HR decisions</p></div>",
    unsafe_allow_html=True,
)
