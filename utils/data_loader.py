# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import joblib
import numpy as np
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
# Sklearn Cross-Version Compatibility
# ==========================================

def fix_sklearn_compatibility(obj):
    """
    Recursively ensures unpickled scikit-learn transformers and pipelines
    are compatible across versions (e.g., handles the missing `_fill_dtype`
    attribute introduced in scikit-learn 1.8 when loading pickles from 1.7).
    """
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            fix_sklearn_compatibility(step)
    if hasattr(obj, "transformers_"):
        for _, trans, _ in obj.transformers_:
            fix_sklearn_compatibility(trans)
    if hasattr(obj, "transformers"):
        for item in obj.transformers:
            if len(item) >= 2:
                fix_sklearn_compatibility(item[1])
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            fix_sklearn_compatibility(step)
    if type(obj).__name__ == "SimpleImputer":
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = getattr(obj, "_fit_dtype", np.float64)
    return obj


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():

    return pd.read_csv(DATASET_PATH)


# ==========================================
# Load Best Model
# ==========================================

def load_model():

    model = joblib.load(MODEL_PATH)
    return fix_sklearn_compatibility(model)


# ==========================================
# Load Scaler
# ==========================================

def load_scaler():

    scaler = joblib.load(SCALER_PATH)
    return fix_sklearn_compatibility(scaler)


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

    pipe = joblib.load(PIPELINE_PATH)
    return fix_sklearn_compatibility(pipe)


# ==========================================
# Load Named Model
# ==========================================

def load_named_model(model_filename: str):
    path = BASE_DIR / "models" / model_filename
    if path.exists():
        model = joblib.load(path)
    else:
        model = joblib.load(MODEL_PATH)
    return fix_sklearn_compatibility(model)


