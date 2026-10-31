"""
Supervised Machine Learning Pipeline for Flood Detection and Probability Prediction.
Compares Logistic Regression baseline, Random Forest, and HistGradientBoosting.
Performs probability calibration, evaluates with ROC-AUC, PR-AUC, F1, Recall, and Confusion Matrix,
and saves production pipeline artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Scikit-learn algorithms & tools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss, classification_report
)

# Custom project modules
from data_preprocessing import validate_and_clean_data, split_data_by_time
from feature_engineering import extract_engineered_features, get_feature_names


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "historical_flood_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "flood_model.joblib")


def evaluate_model_performance(y_true, y_pred, y_prob, model_name="Model"):
    """
    Compute comprehensive metrics prioritizing Recall while reporting Precision,
    F1-score, ROC-AUC, PR-AUC, Brier score, and Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # False alarm rate (fall-out = FP / (FP + TN))
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model_name": model_name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "false_alarm_rate": round(float(far), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    return metrics


def train_and_compare_models(data_path=DATA_PATH, save_artifacts=True):
    """
    Train and rigorously compare algorithms on temporal train/test split.
    """
    print("=" * 70)
    print("AI-BASED FLOOD DETECTION & PROBABILITY PREDICTION PIPELINE")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    raw_df = pd.read_csv(data_path)
    clean_df = validate_and_clean_data(raw_df, verbose=True)

    # Temporal split: Train (2018-2022), Test (2023-2024)
    train_df, test_df = split_data_by_time(clean_df, split_date="2023-01-01")
    print(f"\nTime-based Split:")
    print(f"Training set (2018-2022): {len(train_df):,} samples | Floods: {train_df['flood_next_24h'].sum()} ({train_df['flood_next_24h'].mean()*100:.2f}%)")
    print(f"Testing set  (2023-2024): {len(test_df):,} samples | Floods: {test_df['flood_next_24h'].sum()} ({test_df['flood_next_24h'].mean()*100:.2f}%)")

    # Feature extraction
    X_train_raw = extract_engineered_features(train_df)
    y_train = train_df["flood_next_24h"].values

    X_test_raw = extract_engineered_features(test_df)
    y_test = test_df["flood_next_24h"].values

    feature_names = get_feature_names()
    print(f"\nFeature Matrix: {len(feature_names)} domain-engineered features.")

    # Fit standard scaler for models that benefit from scaling (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # -------------------------------------------------------------
    # 1. Baseline Model: Balanced Logistic Regression
    # -------------------------------------------------------------
    print("\n[1/3] Training Logistic Regression Baseline...")
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # Evaluate Logistic Regression
    prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    # Classification threshold of 0.50
    pred_lr = (prob_lr >= 0.50).astype(int)
    metrics_lr = evaluate_model_performance(y_test, pred_lr, prob_lr, "Logistic Regression (Balanced)")

    # -------------------------------------------------------------
    # 2. Tree Ensemble: Random Forest Classifier
    # -------------------------------------------------------------
    print("[2/3] Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    rf_clf.fit(X_train_raw, y_train)

    prob_rf = rf_clf.predict_proba(X_test_raw)[:, 1]
    pred_rf = (prob_rf >= 0.50).astype(int)
    metrics_rf = evaluate_model_performance(y_test, pred_rf, prob_rf, "Random Forest")

    # -------------------------------------------------------------
    # 3. Gradient Boosting: HistGradientBoostingClassifier
    # -------------------------------------------------------------
    print("[3/3] Training HistGradientBoosting Classifier...")
    # Calculate positive class weight for imbalanced data
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=8,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42
    )
    hgb_clf.fit(X_train_raw, y_train)

    prob_hgb = hgb_clf.predict_proba(X_test_raw)[:, 1]
    pred_hgb = (prob_hgb >= 0.50).astype(int)
    metrics_hgb = evaluate_model_performance(y_test, pred_hgb, prob_hgb, "HistGradientBoosting")

    # -------------------------------------------------------------
    # Select Best Model & Apply Probability Calibration
    # -------------------------------------------------------------
    candidate_metrics = [metrics_lr, metrics_rf, metrics_hgb]
    # In flood forecasting, we rank models primarily by ROC-AUC and Recall (while monitoring False Alarm Rate)
    # A combined score: 0.5 * PR-AUC + 0.5 * Recall
    def rank_score(m):
        return 0.5 * m["pr_auc"] + 0.5 * m["recall"]

    best_metrics = max(candidate_metrics, key=rank_score)
    best_name = best_metrics["model_name"]
    print(f"\nBest Performing Base Model: {best_name} (PR-AUC: {best_metrics['pr_auc']:.4f}, Recall: {best_metrics['recall']:.4f})")

    # Probability Calibration
    print("\nCalibrating predicted probabilities with Sigmoid Platt Scaling...")
    if "Random Forest" in best_name:
        base_estimator = rf_clf
        X_train_best = X_train_raw
        X_test_best = X_test_raw
        use_scaled = False
    elif "HistGradientBoosting" in best_name:
        base_estimator = hgb_clf
        X_train_best = X_train_raw
        X_test_best = X_test_raw
        use_scaled = False
    else:
        base_estimator = log_reg
        X_train_best = X_train_scaled
        X_test_best = X_test_scaled
        use_scaled = True

    calibrated_clf = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv="prefit")
    calibrated_clf.fit(X_train_best, y_train)

    prob_calibrated = calibrated_clf.predict_proba(X_test_best)[:, 1]
    pred_calibrated = (prob_calibrated >= 0.50).astype(int)
    metrics_calibrated = evaluate_model_performance(y_test, pred_calibrated, prob_calibrated, f"Calibrated {best_name}")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Model':<32} | {'Recall':<7} | {'Precision':<9} | {'F1':<7} | {'ROC-AUC':<8} | {'PR-AUC':<7} | {'Brier':<7}")
    print("-" * 80)
    for m in [metrics_lr, metrics_rf, metrics_hgb, metrics_calibrated]:
        print(f"{m['model_name']:<32} | {m['recall']:<7.4f} | {m['precision']:<9.4f} | {m['f1_score']:<7.4f} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<7.4f} | {m['brier_score']:<7.4f}")
    print("=" * 80)

    # Feature Importance extraction
    if hasattr(rf_clf, "feature_importances_"):
        raw_importances = rf_clf.feature_importances_
        feature_importance_dict = {
            feat: round(float(imp), 4) for feat, imp in sorted(zip(feature_names, raw_importances), key=lambda x: x[1], reverse=True)
        }
    else:
        feature_importance_dict = {}

    # Save artifacts
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)

        package = {
            "calibrated_model": calibrated_clf,
            "base_model": base_estimator,
            "best_model_name": best_name,
            "rf_model": rf_clf,
            "hgb_model": hgb_clf,
            "lr_model": log_reg,
            "scaler": scaler,
            "use_scaled": use_scaled,
            "feature_names": feature_names,
            "feature_importances": feature_importance_dict,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "metrics": metrics_calibrated,
            "all_metrics": {
                "logistic_regression": metrics_lr,
                "random_forest": metrics_rf,
                "hist_gradient_boosting": metrics_hgb,
                "calibrated_production_model": metrics_calibrated
            }
        }
        joblib.dump(package, MODEL_PIPELINE_PATH)
        print(f"\nModel pipeline artifact saved to: {MODEL_PIPELINE_PATH}")

        # Save metrics JSON
        summary_results = {
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_period": "2018-01-01 to 2022-12-31",
            "test_period": "2023-01-01 to 2024-12-31",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "models_comparison": {
                "logistic_regression": metrics_lr,
                "random_forest": metrics_rf,
                "hist_gradient_boosting": metrics_hgb,
                "calibrated_production_model": metrics_calibrated
            },
            "top_10_feature_importances": dict(list(feature_importance_dict.items())[:10])
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(summary_results, f, indent=4)
        print(f"Metrics summary saved to: {METRICS_PATH}")

    return {
        "models": {
            "lr": log_reg,
            "rf": rf_clf,
            "hgb": hgb_clf,
            "calibrated": calibrated_clf
        },
        "metrics": {
            "lr": metrics_lr,
            "rf": metrics_rf,
            "hgb": metrics_hgb,
            "calibrated": metrics_calibrated
        },
        "test_data": (X_test_best, y_test, prob_calibrated)
    }


if __name__ == "__main__":
    train_and_compare_models()
