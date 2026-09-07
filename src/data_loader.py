"""
Data Loader and Initial Cleaning Module for Telco Customer Churn
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads raw CSV data from disk."""
    path = Path(filepath) if filepath else DEFAULT_DATA_PATH
    if not path.exists():
        fallback_path = Path(__file__).resolve().parent.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(f"Dataset not found at {path} or fallback {fallback_path}")
    
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans raw DataFrame:
    - Trims whitespace from strings
    - Converts TotalCharges to numeric, imputes blanks (tenure == 0) with 0.0
    - Converts SeniorCitizen from 0/1 to 'No'/'Yes' for uniform categorical processing
    - Maps target Churn to binary (1/0) if present and is_training is True
    """
    df = df.copy()

    # TotalCharges conversion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For tenure == 0, TotalCharges is typically 0
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen to string category for uniform handling
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"}).fillna("No")

    # Enforce domain consistency for No Internet and No Phone Service
    internet_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    if "InternetService" in df.columns:
        no_internet_mask = df["InternetService"].astype(str).str.strip().str.lower() == "no"
        for col in internet_addons:
            if col in df.columns:
                df.loc[no_internet_mask, col] = "No internet service"

    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        no_phone_mask = df["PhoneService"].astype(str).str.strip().str.lower() == "no"
        df.loc[no_phone_mask, "MultipleLines"] = "No phone service"

    # Clean Churn column if present
    if "Churn" in df.columns and is_training:
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

    return df


def get_train_test_data(
    filepath: Optional[str | Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_id: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads, cleans, and splits data into train/test sets with stratification on target Churn.
    """
    raw_df = load_raw_data(filepath)
    cleaned_df = clean_data(raw_df, is_training=True)

    if "Churn" not in cleaned_df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    # Separate target
    y = cleaned_df["Churn"]
    X = cleaned_df.drop(columns=["Churn"])

    if drop_id and "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
