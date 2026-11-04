import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils.data_loader import (
    load_dataset,
    load_encoders,
    load_feature_names,
    load_scaler,
)
from utils.preprocessing import (
    arrange_features,
    encode_features,
    feature_engineering,
    scale_features,
)


MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def evaluate_saved_models():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    metrics_path = model_dir / "model_metrics.json"
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "job_offer_acceptance_dataset.csv"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    if metrics_path.exists():
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved_metrics.get("dataset_sha256") == dataset_hash:
            return (
                pd.DataFrame(saved_metrics["models"])
                .sort_values("F1 Score", ascending=False)
                .reset_index(drop=True)
            )

    data = feature_engineering(load_dataset().copy())
    target_col = "offer_accepted" if "offer_accepted" in data.columns else "Offer_Accepted"
    drop_cols = [c for c in [target_col, "candidate_id"] if c in data.columns]
    
    data = encode_features(data, load_encoders())
    features = arrange_features(data.drop(drop_cols, axis=1), load_feature_names())
    scaled_features = scale_features(features, load_scaler())
    target = data[target_col]
    
    _, test_features, _, test_target = train_test_split(
        scaled_features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    rows = []
    for model_name, filename in MODEL_FILES.items():
        saved_model = joblib.load(model_dir / filename)
        predictions = saved_model.predict(test_features)
        probabilities = saved_model.predict_proba(test_features)[:, 1]
        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(test_target, predictions) * 100,
                "Precision": precision_score(test_target, predictions, zero_division=0) * 100,
                "Recall": recall_score(test_target, predictions, zero_division=0) * 100,
                "F1 Score": f1_score(test_target, predictions, zero_division=0) * 100,
                "ROC-AUC": roc_auc_score(test_target, probabilities) * 100,
            }
        )

    result = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    saved_metrics = {
        "dataset_sha256": dataset_hash,
        "models": result.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(saved_metrics, indent=2), encoding="utf-8")
    return result
