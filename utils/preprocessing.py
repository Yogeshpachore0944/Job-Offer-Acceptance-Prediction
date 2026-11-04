# ==========================================
# Import Libraries
# ==========================================

import pandas as pd

DROPPED_FEATURES = [
    "candidate_id",
    "major",
    "university_tier",
    "city",
    "education_level",
    "kaggle_score",
    "linkedin_connections",
    "internship_count",
    "github_repositories",
    "hackathons_participated",
    "leadership_experience",
    "company_industry",
    "publications_count",
    "application_month",
    "application_year",
]


# ==========================================
# Feature Engineering
# ==========================================

def feature_engineering(df):

    df = df.copy()

    # Drop removed/noisy features if present
    to_drop = [c for c in DROPPED_FEATURES if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Core salary feature engineering
    if "offered_salary_usd" in df.columns and "expected_salary_usd" in df.columns:
        if "salary_offer_gap_usd" not in df.columns:
            df["salary_offer_gap_usd"] = df["offered_salary_usd"] - df["expected_salary_usd"]

        if "offer_to_expected_salary_ratio" not in df.columns:
            df["offer_to_expected_salary_ratio"] = df["offered_salary_usd"] / df["expected_salary_usd"].replace(0, 1)

        df["salary_hike_pct"] = (df["salary_offer_gap_usd"] / df["expected_salary_usd"].replace(0, 1)) * 100

        if "years_experience" in df.columns and "age" in df.columns:
            df["exp_to_age_ratio"] = df["years_experience"] / df["age"].replace(0, 1)

    return df


# ==========================================
# Encode Categorical Columns
# ==========================================

def encode_features(df, encoders):

    df = df.copy()

    for col, enc in encoders.items():
        if col in df.columns:
            known_classes = set(enc.classes_)
            df[col] = df[col].apply(lambda val: val if val in known_classes else enc.classes_[0])
            df[col] = enc.transform(df[col].astype(str))

    return df


# ==========================================
# Arrange Feature Order
# ==========================================

def arrange_features(df, feature_names):

    return df[feature_names]


# ==========================================
# Scaling
# ==========================================

def scale_features(df, scaler):

    return scaler.transform(df)
