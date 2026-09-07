#!/usr/bin/env python3
"""
CLI Prediction Tool for Telco Customer Churn
Usage:
    python3 predict_cli.py --interactive
    python3 predict_cli.py --file path/to/customers.csv --output scored_output.csv
    python3 predict_cli.py --demo
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import ChurnPredictor


def run_demo():
    print("=" * 65)
    print("🎯 RUNNING DEMO CHURN PREDICTIONS")
    print("=" * 65)
    predictor = ChurnPredictor()

    sample_cases = [
        {
            "name": "High-Risk Customer (Month-to-month, Fiber optic, No Support)",
            "data": {
                "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.50, "TotalCharges": 179.00
            }
        },
        {
            "name": "Low-Risk Customer (2-Year Contract, 65m tenure, Tech Support, Auto-Pay)",
            "data": {
                "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
                "tenure": 65, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
                "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
                "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
                "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 64.20, "TotalCharges": 4173.00
            }
        }
    ]

    for case in sample_cases:
        print(f"\n--- {case['name']} ---")
        res = predictor.predict_single(case["data"])
        print(f"• Churn Probability : {res['churn_percentage']}%")
        print(f"• Prediction        : {res['churn_prediction']}")
        print(f"• Risk Tier         : {res['risk_level']}")
        print(f"• Top Risk Drivers  : {', '.join(res['risk_factors'])}")
        if res['retention_actions']:
            print(f"• Recommended Action: {res['retention_actions'][0]['title']} -> {res['retention_actions'][0]['action']}")


def run_batch_file(input_file: str, output_file: str):
    print(f"📂 Loading {input_file}...")
    df = pd.read_csv(input_file)
    predictor = ChurnPredictor()
    print(f"⚙️ Running inference on {len(df)} records...")
    scored = predictor.predict_dataframe(df)
    scored.to_csv(output_file, index=False)
    churn_count = (scored['Churn_Prediction'] == 'Yes').sum()
    print(f"✅ Saved scored results to {output_file}")
    print(f"📊 Summary: {churn_count} / {len(df)} predicted churners ({churn_count/len(df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Telco Customer Churn Prediction CLI")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions on representative personas")
    parser.add_argument("--file", type=str, help="Path to input CSV file for batch prediction")
    parser.add_argument("--output", type=str, default="scored_customers.csv", help="Output path for scored CSV")

    args = parser.parse_args()

    if args.file:
        run_batch_file(args.file, args.output)
    else:
        run_demo()


if __name__ == "__main__":
    main()
