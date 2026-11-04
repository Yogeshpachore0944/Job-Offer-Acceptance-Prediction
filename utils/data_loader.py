# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "job_offer_acceptance_dataset.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

ENCODER_PATH = BASE_DIR / "models" / "encoders.pkl"

FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():

    return pd.read_csv(DATASET_PATH)


# ==========================================
# Load Best Model
# ==========================================

def load_model():

    return joblib.load(MODEL_PATH)


# ==========================================
# Load Scaler
# ==========================================

def load_scaler():

    return joblib.load(SCALER_PATH)


# ==========================================
# Load Encoders
# ==========================================

def load_encoders():

    return joblib.load(ENCODER_PATH)


# ==========================================
# Load Feature Names
# ==========================================

def load_feature_names():

    return joblib.load(FEATURE_PATH)


# ==========================================
# Load Production Pipeline
# ==========================================

def load_pipeline():

    return joblib.load(PIPELINE_PATH)


# ==========================================
# Load Named Model
# ==========================================

def load_named_model(model_filename: str):
    path = BASE_DIR / "models" / model_filename
    if path.exists():
        return joblib.load(path)
    return joblib.load(MODEL_PATH)

