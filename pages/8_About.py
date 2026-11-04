# OfferSense AI
# pages/8_About.py

import pandas as pd
import streamlit as st
from utils.page_state import activate_page

activate_page("8_About")

# CSS

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#0E4C92;
}

.subtitle{
    font-size:18px;
    color:#666;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar

st.sidebar.image(
    "assets/logo.png",
    width=500
)
st.sidebar.title("Navigation")
st.sidebar.write("About")

st.sidebar.markdown("---")

# Header

st.markdown("""
<div class='title'>About OfferSense AI</div>

<div class='subtitle'>
An Intelligent Recruitment Analytics and Job Offer Prediction System using Machine Learning.
</div>
""", unsafe_allow_html=True)

st.divider()

# Product purpose

st.subheader("Built for practical HR decision support")

st.write(
    "OfferSense AI is designed for HR teams managing candidate screening, "
    "interview follow-up and job-offer discussions. It organizes relevant "
    "candidate information into signals that can be reviewed consistently by "
    "recruiters and hiring managers."
)

st.write(
    "The product supports the HR process; it does not replace interviews, "
    "reference checks, compensation conversations or professional judgment. "
    "Its purpose is to make the next HR action clearer and more consistent."
)

st.subheader("How HR teams should interpret the results")

meaning = pd.DataFrame({
    "Result": [
        "Resume review",
        "Offer acceptance prediction",
        "HR recommendation",
    ],
    "How to use it": [
        "Use the score to structure an initial screening conversation and identify what the resume still needs to demonstrate.",
        "Use the outcome and probabilities to decide whether salary, role expectations or timing need further discussion.",
        "Use the recommendation to prepare follow-up questions; confirm the candidate's real expectations directly.",
    ],
})
st.table(meaning)

st.info(
    "The outputs are estimates based on the information entered or uploaded. "
    "They must not be the sole reason to reject, select or make an offer to a candidate."
)

st.divider()

# Technology Stack

st.subheader("Technology Stack")

tech = pd.DataFrame({
    "Category":[
        "Programming Language",
        "Framework",
        "Machine Learning",
        "NLP",
        "Data Processing",
        "Data Visualization",
        "Document Processing",
        "Model Storage"
    ],
    "Technology":[
        "Python",
        "Streamlit",
        "Scikit-learn, XGBoost",
        "TF-IDF, Cosine Similarity",
        "Pandas, NumPy",
        "Matplotlib",
        "PyMuPDF, python-docx",
        "Joblib"
    ]
})

st.table(tech)

st.divider()

# HR use cases

st.subheader("Where the platform fits in the HR process")

modules = pd.DataFrame({
    "Module":[
        "Initial screening",
        "Interview preparation",
        "Offer planning",
        "Candidate comparison",
        "Hiring review"
    ],
    "Purpose":[
        "Create a consistent first review before a recruiter sends the candidate forward.",
        "Identify questions about role fit, skills and evidence before the interview.",
        "Review compensation and candidate factors before finalizing an offer.",
        "Compare candidates using the same decision categories and review standards.",
        "Bring model output, recruiter context and hiring-manager judgment together."
    ]
})

st.table(modules)

st.divider()

# How the platform is built

st.subheader("Machine Learning Methodology")

st.write(
    "The offer analysis is trained as a supervised classification problem. "
    "Historical recruitment records are prepared, relevant factors are "
    "transformed, and several models are compared before the deployed model "
    "produces an outcome and probability estimate."
)

st.write(
    "The resume review is deliberately explainable. It uses document text, "
    "role-related language, section structure and evidence in the uploaded "
    "resume to create a practical screening assessment. It does not claim to "
    "reproduce the private scoring rules of any commercial ATS."
)

st.subheader("Responsible use")
st.markdown(
    """
    - Keep candidate information accurate, current and relevant to the role.
    - Review the explanation instead of relying on a single score.
    - Ask candidates about uncertainties in salary expectations, availability or role fit.
    - Apply the same process consistently and follow your organization's hiring and privacy policies.
    - Treat the model as decision support; the final decision remains with qualified people.
    """
)

st.divider()

# HR value

st.subheader("Expected value for HR teams")

business = pd.DataFrame({
    "Business Benefit":[
        "Consistent screening",
        "Better offer preparation",
        "Faster recruiter review",
        "Structured discussions",
        "Clearer candidate follow-up"
    ],
    "Impact":[
        "Gives recruiters a repeatable starting point for deciding who needs closer review.",
        "Highlights where compensation, timing or expectations may need clarification.",
        "Reduces repetitive organization of candidate and offer information.",
        "Creates a common evidence base for recruiters and hiring managers.",
        "Turns the result into questions and next steps for the candidate conversation."
    ]
})

st.table(business)

st.divider()

# Model Evaluation

st.subheader("Model Evaluation")

evaluation = pd.DataFrame({
    "Module":[
        "Job Offer Acceptance Prediction",
        "Job Offer Acceptance Prediction",
        "Job Offer Acceptance Prediction",
        "Job Offer Acceptance Prediction",
        "Job Offer Acceptance Prediction",
        "Job Offer Acceptance Prediction"
    ],
    "Metric / Output":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "ATS Score (0–100)"
    ],
    "Purpose":[
        "Measures overall classification performance.",
        "Evaluates correctness of acceptance predictions.",
        "Measures ability to identify accepting candidates.",
        "Balances precision and recall.",
        "Measures discrimination between acceptance and rejection.",
        "Measures resume–job compatibility using explainable NLP features."
    ]
})

st.table(evaluation)

st.write(
    "Important: the Resume Selection Score is an application-level compatibility estimate, "
    "not an official score from a commercial ATS provider."
)

# Footer

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        <h4> OfferSense AI </h4>
        <p>
        Machine Learning Based Job Offer Acceptance Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
