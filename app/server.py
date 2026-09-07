"""
FastAPI Server for Telco Customer Churn Interactive Dashboard and API
"""

import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import ChurnPredictor

app = FastAPI(
    title="Telco Customer Churn Intelligence System",
    description="Machine Learning API for predicting customer churn risk, explaining drivers, and generating retention strategies.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "app" / "static"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"


class CustomerProfile(BaseModel):
    gender: str = Field(default="Female", example="Female")
    SeniorCitizen: str = Field(default="No", example="No")
    Partner: str = Field(default="No", example="No")
    Dependents: str = Field(default="No", example="No")
    tenure: float = Field(default=2.0, ge=0, le=100, example=2)
    PhoneService: str = Field(default="Yes", example="Yes")
    MultipleLines: str = Field(default="No", example="No")
    InternetService: str = Field(default="Fiber optic", example="Fiber optic")
    OnlineSecurity: str = Field(default="No", example="No")
    OnlineBackup: str = Field(default="No", example="No")
    DeviceProtection: str = Field(default="No", example="No")
    TechSupport: str = Field(default="No", example="No")
    StreamingTV: str = Field(default="Yes", example="Yes")
    StreamingMovies: str = Field(default="Yes", example="Yes")
    Contract: str = Field(default="Month-to-month", example="Month-to-month")
    PaperlessBilling: str = Field(default="Yes", example="Yes")
    PaymentMethod: str = Field(default="Electronic check", example="Electronic check")
    MonthlyCharges: float = Field(default=89.5, ge=0, example=89.5)
    TotalCharges: float = Field(default=179.0, ge=0, example=179.0)


# Lazy predictor loader
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = ChurnPredictor()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="ML Model not trained yet. Please run 'python3 run_pipeline.py' first."
            )
    return _predictor


@app.get("/api/model-info")
def get_model_info():
    """Returns model metadata, test metrics, and benchmark results."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found. Train model first.")
    with open(meta_path, "r") as f:
        return json.load(f)


@app.get("/api/sample-customer/{persona}")
def get_sample_customer(persona: str):
    """Returns pre-configured customer personas for quick UI testing."""
    samples = {
        "high_risk": {
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
        },
        "moderate_risk": {
            "gender": "Male",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 18,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.00,
            "TotalCharges": 1170.00
        },
        "low_risk": {
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
    }
    if persona not in samples:
        raise HTTPException(status_code=404, detail="Persona not found. Choose: high_risk, moderate_risk, low_risk")
    return samples[persona]


@app.post("/api/predict")
def predict_churn(profile: CustomerProfile):
    """Predicts churn probability, risk level, and retention strategies for a customer."""
    predictor = get_predictor()
    result = predictor.predict_single(profile.model_dump())
    return result


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Uploads CSV of customer profiles and returns scored churn probabilities."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    predictor = get_predictor()
    try:
        scored_df = predictor.predict_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error on batch: {str(e)}")

    # Summary metrics
    total_records = len(scored_df)
    churn_count = int((scored_df["Churn_Prediction"] == "Yes").sum())
    avg_risk = float(scored_df["Churn_Probability"].mean() * 100)
    high_risk_count = int((scored_df["Risk_Tier"] == "High").sum())
    medium_risk_count = int((scored_df["Risk_Tier"] == "Medium").sum())
    low_risk_count = int((scored_df["Risk_Tier"] == "Low").sum())

    # Return top 50 rows for preview + summary
    preview_data = scored_df.head(50).to_dict(orient="records")

    return {
        "total_records": total_records,
        "predicted_churners": churn_count,
        "churn_rate_pct": round((churn_count / total_records) * 100, 1) if total_records else 0,
        "avg_churn_risk_pct": round(avg_risk, 1),
        "risk_breakdown": {
            "high": high_risk_count,
            "medium": medium_risk_count,
            "low": low_risk_count,
        },
        "preview": preview_data,
        "columns": scored_df.columns.tolist()
    }


@app.get("/api/plots/{plot_name}")
def get_plot_image(plot_name: str):
    """Returns saved plot images."""
    safe_name = Path(plot_name).name
    img_path = PLOTS_DIR / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(img_path)


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check probe for container orchestrators and cloud platforms."""
    model_ready = (MODELS_DIR / "best_churn_model.joblib").exists()
    return {
        "status": "healthy" if model_ready else "degraded",
        "service": "telco-churn-intelligence-engine",
        "version": "1.0.0",
        "model_loaded": model_ready
    }


# Mount static assets directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    # Dynamically bind to cloud-injected PORT (Render/Hugging Face default) or 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)
