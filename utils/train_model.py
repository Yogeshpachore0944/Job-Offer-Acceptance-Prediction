import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.preprocessing import DROPPED_FEATURES, feature_engineering


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "job_offer_acceptance_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df_clean = feature_engineering(df)

target_col = "offer_accepted" if "offer_accepted" in df_clean.columns else "Offer_Accepted"
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    encoders[col] = le

feature_names = X_encoded.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Balanced positive class weighting (~2.01 for 67% / 33% distribution)
scale_pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))

models = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced"),
    "xgboost": XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    ),
}
model_display_names = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

scores = {}
comparison_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    scores[name] = f1_score(y_test, predictions, zero_division=0)
    comparison_rows.append(
        {
            "Model": model_display_names[name],
            "Accuracy": accuracy_score(y_test, predictions) * 100,
            "Precision": precision_score(y_test, predictions, zero_division=0) * 100,
            "Recall": recall_score(y_test, predictions, zero_division=0) * 100,
            "F1 Score": scores[name] * 100,
            "ROC-AUC": roc_auc_score(y_test, probabilities) * 100,
        }
    )
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")

best_name = max(scores, key=scores.get)
best_model_obj = models[best_name]

# Build and train production end-to-end Pipeline on raw features
raw_X = X.copy()
raw_y = y.copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        ),
    ),
])

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    raw_X, raw_y, test_size=0.20, random_state=42, stratify=raw_y
)

print("Fitting End-to-End Pipeline on cleaned dataset...")
pipeline.fit(X_raw_train, y_raw_train)
joblib.dump(pipeline, MODEL_DIR / "pipeline.pkl")
print(f"Production pipeline saved to {MODEL_DIR / 'pipeline.pkl'}")

joblib.dump(best_model_obj, MODEL_DIR / "best_model.pkl")
joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# Generate dataset hash based on cleaned features
data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
metrics = {
    "dataset_sha256": data_hash,
    "models": comparison_rows,
}
(MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"Features used ({len(feature_names)}): {feature_names}")
print("F1 scores:", scores)
print("Best model:", best_name)
print("Model training completed successfully.")
