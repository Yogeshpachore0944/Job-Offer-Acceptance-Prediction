# OfferSense AI
# pages/1_Dataset.py

import streamlit as st
import pandas as pd

from utils.data_loader import load_dataset
from utils.page_state import activate_page

st.set_page_config(page_title="Dataset Overview", layout="wide")
activate_page("1_Dataset")

# Custom CSS

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#0E4C92;
}

.subtitle{
    color:#666;
    font-size:18px;
    margin-bottom:20px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 12px rgba(0,0,0,0.08);
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
st.sidebar.write("Dataset Module")

st.sidebar.markdown("---")


# Load Dataset

@st.cache_data
def get_dataset():

    return load_dataset()

df = get_dataset()

# Header

st.markdown(
    """
    <div class='title'>
        Dataset Overview
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
        Explore the dataset used to train the OfferSense AI prediction model.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("This page provides a complete overview of the dataset including structure, statistics, missing values, duplicates and downloadable data.")

st.markdown("---")

# Dataset Statistics

total_rows = df.shape[0]
total_columns = df.shape[1]
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{total_rows:,}"
    )

with col2:
    st.metric(
        "Total Features",
        total_columns
    )

with col3:
    st.metric(
        "Missing Values",
        missing_values
    )

with col4:
    st.metric(
        "Duplicate Rows",
        duplicate_rows
    )

st.markdown("---")

# Offer outcome balance

st.subheader("Offer outcome balance")
target_col = "offer_accepted" if "offer_accepted" in df.columns else "Offer_Accepted"
outcome_counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
rejected_count = int(outcome_counts.loc[0])
accepted_count = int(outcome_counts.loc[1])
rejected_rate = rejected_count / total_rows if total_rows else 0
accepted_rate = accepted_count / total_rows if total_rows else 0
balance_col1, balance_col2 = st.columns(2)
balance_col1.metric("Rejected offers", f"{rejected_count:,}", f"{rejected_rate:.1%}")
balance_col2.metric("Accepted offers", f"{accepted_count:,}", f"{accepted_rate:.1%}")

st.markdown("---")

# Dataset Shape

st.subheader("Dataset Shape")

shape_col1, shape_col2 = st.columns(2)

with shape_col1:
    st.write(f"**Rows :** {total_rows:,}")

with shape_col2:
    st.write(f"**Columns :** {total_columns}")

st.markdown("---")

# Dataset Preview

st.subheader("Dataset Preview")

rows_to_show = st.slider(
    "Number of Rows to Display",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

st.dataframe(
    df.head(rows_to_show),
    hide_index=True
)

st.markdown("---")

# Interactive Dataset Explorer

st.subheader("Dataset Explorer")

search_column = st.selectbox(
    "Select a Column",
    df.columns
)

search_value = st.text_input(
    f"Search in '{search_column}'"
)

if search_value:

    filtered_df = df[
        df[search_column]
        .astype(str)
        .str.contains(search_value, case=False, na=False)
    ]

    st.write(f"Matching Records: **{len(filtered_df):,}**")

    st.dataframe(
        filtered_df,
        hide_index=True
    )

else:

    st.dataframe(
        df,
        hide_index=True
    )

st.markdown("---")

# Column Information

st.subheader("Column Information")

column_info = pd.DataFrame({

    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Non-Null Count": df.notnull().sum().values

})

st.dataframe(
    column_info,
    hide_index=True
)

st.markdown("---")

# Missing Value Analysis

st.subheader("Missing Value Analysis")

missing_df = pd.DataFrame({

    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Missing (%)": (
        df.isnull().sum() / len(df) * 100
    ).round(2)

})

if missing_df["Missing Values"].sum() == 0:

    st.success("No Missing Values found in the dataset.")

else:

    st.dataframe(
        missing_df,
        hide_index=True
    )

st.markdown("---")

# Duplicate Record Analysis

st.subheader("Duplicate Record Analysis")

duplicates = df.duplicated().sum()

if duplicates == 0:

    st.success("No Duplicate Records found.")

else:

    st.warning(f"⚠ {duplicates:,} duplicate records detected.")

st.markdown("---")

# Statistical Summary

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(include="all"),
)

st.markdown("---")

# Download Dataset

st.subheader("Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Dataset (CSV)",
    data=csv,
    file_name="job_offer_acceptance_dataset.csv",
    mime="text/csv",
)

st.markdown("---")

# Dataset Insights

st.subheader("Dataset Insights")

st.write(f"""
##### Dataset Highlights

- Total Records : {total_rows:,}

- Total Features : {total_columns}

- Missing Values : {missing_values}

- Duplicate Records : {duplicate_rows}

- Ready for Machine Learning
""")

st.markdown("---")

# Machine Learning Workflow

st.subheader("Machine Learning Workflow")

workflow = """
Dataset
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Comparison
    ↓
Best Model Selection
    ↓
Prediction
"""

st.code(workflow)

st.markdown("---")

# Project Summary

st.subheader("About this Dataset")

st.write("""
- This dataset is designed for predicting whether a candidate is likely to accept a job offer.

- It captures comprehensive candidate credentials (experience, academic tier, portfolio metrics, Kaggle score, GitHub repositories), company factors (industry, size, rating), and compensation details (expected vs. offered salary).

- The dataset serves as the foundation for the OfferSense AI prediction system and is used throughout the EDA, Model Training, Model Comparison and Prediction modules.
""")

st.markdown("---")

# Footer

st.markdown(
    """
    <div class='footer'>
        <h4>OfferSense AI</h4>
        <p>
        Machine Learning Based Job Offer Acceptance Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
