"""
Inference Engine and Retention Strategy Module for Telco Customer Churn
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np

from src.data_loader import clean_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


class ChurnPredictor:
    """
    Inference and Explainability engine for customer churn prediction.
    """

    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self._load_assets()

    def _load_assets(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Please run train.py first.")
        
        self.pipeline = joblib.load(self.model_path)
        
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.threshold = self.metadata.get("optimal_threshold", 0.5)
        else:
            self.metadata = {}
            self.threshold = 0.5

    def _generate_risk_factors(self, cust: Dict[str, Any]) -> List[str]:
        """Identifies specific risk drivers and positive retention anchors for this customer."""
        factors = []
        
        # 1. Contract Commitment
        contract = str(cust.get("Contract", "")).strip()
        if contract == "Month-to-month":
            factors.append("Month-to-month contract (Highest churn risk factor in dataset — 42.7% avg churn)")
        elif contract == "Two year":
            factors.append("2-Year contract commitment (Strongest retention anchor — only 2.8% avg churn)")
        elif contract == "One year":
            factors.append("1-Year contract commitment (Solid retention buffer — 11.3% avg churn)")
        
        # 2. Tenure Lifetime
        try:
            tenure = float(cust.get("tenure", 0))
            if tenure <= 6:
                factors.append("Early lifecycle risk (Tenure <= 6 months — 52.9% churn volatility)")
            elif tenure <= 12:
                factors.append("First-year customer lifecycle (Tenure <= 12 months)")
            elif tenure >= 48:
                factors.append("Long-term established loyalty (Tenure >= 48 months — under 10% churn)")
        except (ValueError, TypeError):
            pass

        # 3. Internet Service Type & Pricing
        inet = str(cust.get("InternetService", "")).strip()
        if inet == "Fiber optic":
            factors.append("Fiber optic internet plan (High price sensitivity & competitor switching — 41.9% churn)")
        elif inet == "No":
            factors.append("No internet service (Basic phone-only plan — very low 7.4% churn profile)")
        elif inet == "DSL":
            factors.append("DSL internet plan (Moderate baseline churn — 19.0%)")

        try:
            monthly = float(cust.get("MonthlyCharges", 0))
            if monthly > 85:
                factors.append(f"High monthly bill burden (${monthly:.2f}/mo)")
            elif monthly < 30:
                factors.append(f"Low budget tier plan (${monthly:.2f}/mo)")
            
            # Recent price jump detection
            total = float(cust.get("TotalCharges", 0))
            tenure = float(cust.get("tenure", 0))
            if tenure > 1:
                avg_hist = total / tenure
                if (monthly - avg_hist) > 15:
                    factors.append(f"Recent monthly price hike (+${(monthly - avg_hist):.2f}/mo above historical avg)")
        except (ValueError, TypeError):
            pass

        # 4. Addon Services (only relevant if customer has active internet)
        if inet != "No":
            tech_sup = str(cust.get("TechSupport", "")).strip()
            if tech_sup == "No":
                factors.append("No Tech Support add-on subscribed")

            sec = str(cust.get("OnlineSecurity", "")).strip()
            if sec == "No":
                factors.append("No Online Security protection active")

        # 5. Payment Method & Billing
        pay = str(cust.get("PaymentMethod", "")).strip()
        if "electronic check" in pay.lower():
            factors.append("Electronic Check payment method (High churn correlation — 45.3% avg churn)")
        elif "automatic" in pay.lower():
            factors.append("Automated payment enabled (Bank/Card Auto-Pay reduces friction)")

        paperless = str(cust.get("PaperlessBilling", "")).strip()
        if paperless == "Yes" and contract == "Month-to-month":
            factors.append("Paperless billing active on month-to-month")

        if not factors:
            factors.append("Standard profile baseline indicators")

        return factors[:4]

    def _generate_retention_recommendations(self, cust: Dict[str, Any], proba: float) -> List[Dict[str, str]]:
        """Produces actionable, personalized retention recommendations."""
        recs = []

        contract = str(cust.get("Contract", "")).strip()
        pay = str(cust.get("PaymentMethod", "")).strip()
        tech_sup = str(cust.get("TechSupport", "")).strip()
        sec = str(cust.get("OnlineSecurity", "")).strip()
        inet = str(cust.get("InternetService", "")).strip()

        if contract == "Month-to-month":
            recs.append({
                "title": "Lock-In Long Term Contract",
                "action": "Offer a 1-year or 2-year contract with a 15% discount for the first 3 months.",
                "impact": "Reduces churn probability by ~40%"
            })

        if "automatic" not in pay.lower():
            recs.append({
                "title": "Incentivize Auto-Pay Enrollment",
                "action": "Provide a $5/month billing credit upon switching to Bank Transfer or Credit Card Auto-Pay.",
                "impact": "Reduces churn probability by ~25%"
            })

        if inet != "No" and (tech_sup == "No" or sec == "No"):
            recs.append({
                "title": "Bundle Free Security & Support",
                "action": "Offer 6 months of complimentary Tech Support & Online Security add-ons.",
                "impact": "Improves customer satisfaction & sticky usage"
            })

        if proba > 0.60:
            recs.append({
                "title": "Dedicated Customer Success Outreach",
                "action": "Schedule a proactive check-in call from senior retention specialists within 48 hours.",
                "impact": "High-priority immediate intervention"
            })

        if not recs or proba <= 0.30:
            recs.append({
                "title": "Loyalty Appreciation & Cross-Sell",
                "action": "Send a customer appreciation reward (bonus speed boost or loyalty gift).",
                "impact": "Reinforces long-term brand loyalty & NPS"
            })

        return recs[:3]

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single customer dictionary.
        """
        df = pd.DataFrame([customer_data])
        cleaned_df = clean_data(df, is_training=False)

        if "customerID" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=["customerID"])
        if "Churn" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=["Churn"])

        proba = float(self.pipeline.predict_proba(cleaned_df)[0, 1])
        prediction = "Yes" if proba >= self.threshold else "No"

        if proba < 0.30:
            risk_level = "Low"
            risk_color = "#10b981" # Green
        elif proba < 0.60:
            risk_level = "Medium"
            risk_color = "#f59e0b" # Amber
        else:
            risk_level = "High"
            risk_color = "#ef4444" # Red

        risk_factors = self._generate_risk_factors(customer_data)
        retention_actions = self._generate_retention_recommendations(customer_data, proba)

        return {
            "churn_prediction": prediction,
            "churn_probability": round(proba, 4),
            "churn_percentage": round(proba * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "decision_threshold": self.threshold,
            "risk_factors": risk_factors,
            "retention_actions": retention_actions,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch inference on a pandas DataFrame.
        """
        input_df = df.copy()
        cleaned_df = clean_data(input_df, is_training=False)
        
        feature_df = cleaned_df.copy()
        if "customerID" in feature_df.columns:
            feature_df = feature_df.drop(columns=["customerID"])
        if "Churn" in feature_df.columns:
            feature_df = feature_df.drop(columns=["Churn"])

        probas = self.pipeline.predict_proba(feature_df)[:, 1]
        preds = ["Yes" if p >= self.threshold else "No" for p in probas]
        risk_tiers = [
            "Low" if p < 0.30 else ("Medium" if p < 0.60 else "High") for p in probas
        ]

        result_df = input_df.copy()
        result_df["Churn_Probability"] = np.round(probas, 4)
        result_df["Churn_Risk_Pct"] = np.round(probas * 100, 1)
        result_df["Churn_Prediction"] = preds
        result_df["Risk_Tier"] = risk_tiers

        return result_df
