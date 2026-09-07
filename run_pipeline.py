#!/usr/bin/env python3
"""
Master Execution Script for Telco Customer Churn ML Pipeline
"""

import sys
import time
from pathlib import Path

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train import train_and_export_pipeline
from src.predict import ChurnPredictor


def main():
    print("=" * 70)
    print("   🚀 TELCO CUSTOMER CHURN PREDICTION - END-TO-END ML PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Train and evaluate pipeline
    print("\n[Phase 1/2] Training, Benchmarking & Hyperparameter Tuning...")
    metadata = train_and_export_pipeline()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f"🏆 Champion Model: {metadata['model_name']}")
    print(f"🎯 Calibrated Threshold: {metadata['optimal_threshold']}")
    print("\n📊 Holdout Test Set Performance:")
    tm = metadata["test_metrics"]
    print(f"   • Accuracy:  {tm['accuracy']:.4f}")
    print(f"   • ROC-AUC:   {tm['roc_auc']:.4f}")
    print(f"   • Recall:    {tm['recall']:.4f}")
    print(f"   • Precision: {tm['precision']:.4f}")
    print(f"   • F1-Score:  {tm['f1']:.4f}")

    # 2. Test Single Customer Inference
    print("\n[Phase 2/2] Validating Inference Engine with Sample Personas...")
    predictor = ChurnPredictor()

    high_risk_persona = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 179.00
    }

    low_risk_persona = {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.20,
        "TotalCharges": 4173.00
    }

    print("\n--- Persona 1: High-Risk Profile (Month-to-month, Fiber optic, No Support, Electronic check) ---")
    res1 = predictor.predict_single(high_risk_persona)
    print(f"Churn Probability: {res1['churn_percentage']}% | Risk Tier: {res1['risk_level']} | Prediction: {res1['churn_prediction']}")
    print("Risk Factors:", res1["risk_factors"])
    print("Recommended Retention Action:", res1["retention_actions"][0]["title"] if res1["retention_actions"] else "None")

    print("\n--- Persona 2: Low-Risk Profile (2-Year Contract, 65m tenure, Tech Support, Auto-Pay) ---")
    res2 = predictor.predict_single(low_risk_persona)
    print(f"Churn Probability: {res2['churn_percentage']}% | Risk Tier: {res2['risk_level']} | Prediction: {res2['churn_prediction']}")
    print("Risk Factors:", res2["risk_factors"])

    print("\n" + "=" * 70)
    print("   ✨ PIPELINE READY FOR PRODUCTION & INTERACTIVE DASHBOARD")
    print("=" * 70)


if __name__ == "__main__":
    main()
