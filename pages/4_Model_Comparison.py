import pandas as pd
import streamlit as st

from utils.data_loader import load_dataset, load_feature_names
from utils.model_evaluation import evaluate_saved_models
from utils.page_state import activate_page


st.set_page_config(page_title="Model Comparison", layout="wide")
activate_page("4_Model_Comparison")

with st.sidebar:
    st.image("assets/logo.png", width=500)
    st.title("Navigation")
    st.write("Model Comparison Module")
    st.markdown("---")


@st.cache_data
def get_model_comparison():
    return evaluate_saved_models()


@st.cache_data
def get_dataset():
    return load_dataset()


dataset = get_dataset()
comparison_df = get_model_comparison()
best_model = comparison_df.iloc[0]

target_col = "offer_accepted" if "offer_accepted" in dataset.columns else "Offer_Accepted"
outcome_counts = dataset[target_col].value_counts()

st.title("Model Comparison")
st.write(
    "Compare the saved classifiers on the same stratified 20% test split "
    "from the current offer acceptance dataset."
)

with st.container(horizontal=True):
    st.metric("Models evaluated", len(comparison_df), border=True)
    st.metric("Best model", best_model["Model"], border=True)
    st.metric("Best F1 score", f"{best_model['F1 Score']:.2f}%", border=True)
    st.metric("Candidate factors", len(load_feature_names()), border=True)

st.subheader("Current dataset outcome balance")
rejected_count = int(outcome_counts.get(0, 0))
accepted_count = int(outcome_counts.get(1, 0))
balance_columns = st.columns(2)
balance_columns[0].metric("Rejected offers", f"{rejected_count:,}")
balance_columns[1].metric("Accepted offers", f"{accepted_count:,}")

st.subheader("Performance comparison")
comparison_display = comparison_df.copy()
for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
    comparison_display[metric] = comparison_display[metric].map(lambda value: f"{value:.2f}%")
st.dataframe(comparison_display, hide_index=True)

chart_data = comparison_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
]
st.bar_chart(chart_data, y_label="Score (%)")
st.caption(
    "Scores are calculated from saved models and the same held-out test rows. "
    "F1 is used to rank models because it balances precision and recall."
)

st.subheader("Model selection")
st.write(
    f"{best_model['Model']} currently leads with an F1 score of "
    f"{best_model['F1 Score']:.2f}% and Accuracy of {best_model['Accuracy']:.2f}%. "
    "Re-run `train_model.py` after updating the dataset to refresh these saved models and metrics."
)
