# Import Libraries
import streamlit as st

from utils.page_state import activate_page
from utils.data_loader import load_dataset

# PAGE CONFIGURATION
st.set_page_config(
    page_title="OfferSense AI",
    layout="wide",
    initial_sidebar_state="expanded"
)
activate_page("Home")


@st.cache_data
def get_dataset():
    return load_dataset()


dataset = get_dataset()

# CSS

st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
}

.sub-title{
    font-size:20px;
    text-align:center;
    color:gray;
}

.card{
    padding:20px;
    border-radius:15px;
    border:1px solid #E6E6E6;
    background:#F9F9F9;
}

.footer{
    text-align:center;
    font-size:15px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# Sidebar

st.sidebar.image(
    "assets/logo.png",
    width=500
)

st.sidebar.markdown("---")

# Title

st.markdown("""
<div class='title'> <h1>OfferSense AI </h1></div>
</div>
""",
unsafe_allow_html=True
)

st.markdown("""
    <p style="
        margin-top:5px;
        margin-bottom:0;
        color:#9CA3AF;
        font-size:20px;
        font-family:'Segoe UI',sans-serif;
    ">
        Intelligent Recruitment Analytics & Job Offer Acceptance Prediction System
    </p>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# Welcome

st.subheader("Recruitment decisions, supported by evidence")

st.write(
    "OfferSense AI gives HR teams a structured starting point for screening "
    "resumes and preparing job-offer discussions. It helps recruiters spend "
    "less time organizing information and more time evaluating the candidate "
    "behind it."
)

st.write(
    "Use the result as a consistent input to shortlisting, interview planning "
    "and offer follow-up. The application presents the reasoning in business "
    "language, so HR professionals do not need a data science background."
)

# What the platform helps with

st.subheader("What you can do here")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Shortlist with context**

        Review the resume's role alignment, structure and evidence before
        deciding whether it should move forward in the hiring process.
        """
    )

# with col2:
#     st.markdown(
#         """
# col2.metric("Features", "11")
#         - Decision Tree
#         - Random Forest
#         - XGBoost
#         """
#     )

with col2:
    st.markdown(
        """
        **Prepare an offer conversation**

        Estimate the likelihood of acceptance from the candidate profile,
        interview outcome and proposed compensation details.
        """
    )

with col3:
    st.markdown(
        """
        **Support a consistent process**

        Use the same information categories across candidates and record the
        result as one input in a structured HR discussion.
        """
    )

# HR decision support

st.subheader("Two moments where HR teams need clarity")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("Resume screening", "Role fit")

with r2:
    st.metric("Offer review", "Acceptance")

with r3:
    st.metric("HR output", "Action points")

st.info(
    "The resume review supports screening, while the offer prediction supports "
    "offer planning. Neither output should be used as the sole hiring decision."
)

# Statistics

st.subheader("Platform coverage")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Historical records", f"{len(dataset):,}")
col2.metric("Candidate factors", dataset.shape[1] - 1)
col3.metric("Compared models", "4")
col4.metric("HR workflows", "2")

# Getting started

st.subheader("How an HR team can use it")

st.info("""
1. Select resume screening or offer assessment from the navigation.
2. Enter only the candidate and role information relevant to the decision.
3. Review the result, explanation and recommended action points.
4. Validate the result with the recruiter, hiring manager and candidate context.
5. Record the final decision according to your organization's HR process.
""")

st.caption(
    "Use consistent criteria for every candidate, protect personal information "
    "and keep human review at the center of the hiring decision."
)

# Footer

st.divider()

st.markdown(
    """
    <div class='footer'>
        <h4>OfferSense AI</h4>
        <p>
        Machine Learning Based Recruitment Analytics & Job Offer Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
