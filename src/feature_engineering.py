"""
Feature Engineering Module for Telco Customer Churn
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Telco Churn feature engineering.
    """

    def __init__(self, include_tenure_cohort: bool = True):
        self.include_tenure_cohort = include_tenure_cohort

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Total Add-on Services Count
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]
        
        present_services = [c for c in service_cols if c in df.columns]
        if present_services:
            df["TotalServicesCount"] = df[present_services].apply(
                lambda row: sum(1 for v in row if str(v).strip().lower() == "yes"), axis=1
            )
            
            # Security & Support focused add-ons count
            sec_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"] if c in df.columns]
            df["SecurityServicesCount"] = df[sec_cols].apply(
                lambda row: sum(1 for v in row if str(v).strip().lower() == "yes"), axis=1
            )

            # Streaming services count
            stream_cols = [c for c in ["StreamingTV", "StreamingMovies"] if c in df.columns]
            df["StreamingCount"] = df[stream_cols].apply(
                lambda row: sum(1 for v in row if str(v).strip().lower() == "yes"), axis=1
            )
        else:
            df["TotalServicesCount"] = 0
            df["SecurityServicesCount"] = 0
            df["StreamingCount"] = 0

        # 2. Tenure Cohort
        if self.include_tenure_cohort and "tenure" in df.columns:
            def categorize_tenure(t):
                if t <= 12:
                    return "0-12m"
                elif t <= 24:
                    return "12-24m"
                elif t <= 48:
                    return "24-48m"
                elif t <= 60:
                    return "48-60m"
                else:
                    return ">60m"
            df["TenureCohort"] = df["tenure"].apply(categorize_tenure)

        # 3. Monthly To Total Charges Ratio
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)
            df["AvgMonthlyChargeHistorical"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / np.maximum(df["tenure"], 1),
                df["MonthlyCharges"]
            )
            df["ChargeDiscrepancy"] = df["MonthlyCharges"] - df["AvgMonthlyChargeHistorical"]

        # 4. AutoPayment Indicator
        if "PaymentMethod" in df.columns:
            df["AutoPayment"] = df["PaymentMethod"].astype(str).str.lower().apply(
                lambda x: "Yes" if "automatic" in x else "No"
            )

        # 5. Short-term Contract Indicator
        if "Contract" in df.columns:
            df["IsMonthToMonth"] = df["Contract"].apply(
                lambda x: "Yes" if str(x).strip() == "Month-to-month" else "No"
            )

        # 6. Fiber Optic Risk Flag
        if "InternetService" in df.columns:
            df["IsFiberOptic"] = df["InternetService"].apply(
                lambda x: "Yes" if str(x).strip() == "Fiber optic" else "No"
            )

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Helper functional wrapper for TelcoFeatureEngineer"""
    fe = TelcoFeatureEngineer()
    return fe.transform(df)
