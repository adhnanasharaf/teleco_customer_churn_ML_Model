"""
Preprocessing Pipeline Module for Telco Customer Churn
"""

from typing import List, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.feature_engineering import TelcoFeatureEngineer


NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServicesCount",
    "SecurityServicesCount",
    "StreamingCount",
    "MonthlyToTotalRatio",
    "AvgMonthlyChargeHistorical",
    "ChargeDiscrepancy",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureCohort",
    "AutoPayment",
    "IsMonthToMonth",
    "IsFiberOptic",
]


def build_preprocessor() -> Pipeline:
    """
    Constructs a full preprocessor pipeline:
    1. TelcoFeatureEngineer
    2. ColumnTransformer (Numerical scaling + Categorical One-Hot Encoding)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    full_preprocessor = Pipeline([
        ("feature_engineer", TelcoFeatureEngineer()),
        ("column_transformer", column_transformer),
    ])

    return full_preprocessor


def get_feature_names_after_preprocessing(preprocessor: Pipeline, sample_df: pd.DataFrame) -> List[str]:
    """
    Extracts output feature names from the fitted preprocessor pipeline.
    """
    # Fit preprocessor if not already fitted
    fe_df = preprocessor.named_steps["feature_engineer"].transform(sample_df)
    ct = preprocessor.named_steps["column_transformer"]
    
    cat_encoder = ct.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    
    all_features = NUMERICAL_FEATURES + encoded_cat_features
    return all_features
