"""
Telco Customer Churn Intelligence System
Production Gradio Application for Hugging Face Spaces & Local Deployment
"""

import os
import sys
import warnings
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

# Suppress pickle version mismatch warnings
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 1. Compatibility hook for Hugging Face ZeroGPU runtime
try:
    import spaces  # type: ignore
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

# 2. Compatibility shim: Ensure HfFolder is present if huggingface_hub >= 0.24 is loaded
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @classmethod
            def get_token(cls):
                return getattr(huggingface_hub, "get_token", lambda: None)()

            @classmethod
            def save_token(cls, token: str):
                if hasattr(huggingface_hub, "login"):
                    huggingface_hub.login(token=token)

            @classmethod
            def delete_token(cls):
                if hasattr(huggingface_hub, "logout"):
                    huggingface_hub.logout()

        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# 3. Patch Gradio 5 client schema parsing bug (handles boolean additionalProperties)
try:
    import gradio_client.utils
    _orig_json_schema = gradio_client.utils._json_schema_to_python_type
    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool) or not isinstance(schema, dict):
            return "Any"
        try:
            return _orig_json_schema(schema, defs)
        except Exception:
            return "Any"
    gradio_client.utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

import gradio as gr
from src.predict import ChurnPredictor

# Initialize the predictor
predictor = ChurnPredictor()

# Load model metadata for analytics tab
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
PLOTS_DIR = BASE_DIR / "plots"

# Quick Persona definitions
PERSONAS = {
    "high_risk": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50, "TotalCharges": 179.00
    },
    "moderate_risk": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 18, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.00, "TotalCharges": 1170.00
    },
    "low_risk": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 65, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.20, "TotalCharges": 4173.00
    }
}


def load_persona_data(persona_key: str):
    p = PERSONAS[persona_key]
    return (
        p["gender"], p["SeniorCitizen"], p["Partner"], p["Dependents"], p["tenure"],
        p["PhoneService"], p["MultipleLines"], p["InternetService"], p["OnlineSecurity"],
        p["OnlineBackup"], p["DeviceProtection"], p["TechSupport"], p["StreamingTV"],
        p["StreamingMovies"], p["Contract"], p["PaperlessBilling"], p["PaymentMethod"],
        p["MonthlyCharges"], p["TotalCharges"]
    )


def build_prob_meter_html(prob: float) -> str:
    pct = prob * 100
    if pct >= 50:
        color = "#ef4444"
        text_color = "#fca5a5"
    elif pct >= 30:
        color = "#f59e0b"
        text_color = "#fcd34d"
    else:
        color = "#10b981"
        text_color = "#6ee7b7"

    return f"""
    <div style="background: #1e293b; padding: 18px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #cbd5e1; font-size: 14px;">Estimated Churn Probability</span>
            <span style="font-weight: 800; color: {text_color}; font-size: 22px;">{pct:.1f}%</span>
        </div>
        <div style="background: #0f172a; border-radius: 999px; height: 16px; overflow: hidden; position: relative; border: 1px solid #475569;">
            <div style="background: {color}; height: 100%; width: {min(pct, 100):.1f}%; border-radius: 999px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: #94a3b8;">
            <span>🟢 0% (Loyal)</span>
            <span>Decision Threshold (35.0%)</span>
            <span>🔴 100% (Definite Churn)</span>
        </div>
    </div>
    """


# 4. Single Prediction Function (Decorated for ZeroGPU)
@gpu_decorator
def predict_single_customer(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):
    try:
        cust = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }
        res = predictor.predict_single(cust)
        prob = res["churn_probability"]
        prob_pct = f"{prob * 100:.1f}%"
        tier = res["risk_tier"]
        prediction = res["churn_prediction"]
        
        # Color coding for risk badge
        if tier == "High":
            badge_html = f"""
            <div style="background: rgba(239,68,68,0.18); border: 1px solid #ef4444; border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 12px;">
                <span style="font-size: 22px; font-weight: 800; color: #ef4444;">🚨 HIGH CHURN RISK ({prob_pct})</span>
                <p style="color: #cbd5e1; margin-top: 4px; font-size: 13px;">Customer is strongly inclined to cancel. Immediate retention intervention required.</p>
            </div>
            """
        elif tier == "Medium":
            badge_html = f"""
            <div style="background: rgba(245,158,11,0.18); border: 1px solid #f59e0b; border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 12px;">
                <span style="font-size: 22px; font-weight: 800; color: #f59e0b;">⚠️ MODERATE CHURN RISK ({prob_pct})</span>
                <p style="color: #cbd5e1; margin-top: 4px; font-size: 13px;">Customer shows warning indicators. Targeted incentive offers recommended.</p>
            </div>
            """
        else:
            badge_html = f"""
            <div style="background: rgba(16,185,129,0.18); border: 1px solid #10b981; border-radius: 10px; padding: 16px; text-align: center; margin-bottom: 12px;">
                <span style="font-size: 22px; font-weight: 800; color: #10b981;">🛡️ LOW CHURN RISK ({prob_pct})</span>
                <p style="color: #cbd5e1; margin-top: 4px; font-size: 13px;">Customer is highly loyal and stable. Suitable for loyalty cross-selling.</p>
            </div>
            """

        meter_html = build_prob_meter_html(prob)

        drivers_md = "### 🔍 Key Risk Drivers & Anchors\n"
        for factor in res.get("top_risk_factors", []):
            drivers_md += f"* **{factor}**\n"

        strat_md = "### 💡 Recommended Retention Action Plan\n"
        for strat in res.get("retention_strategies", []):
            strat_md += f"* {strat}\n"

        return badge_html, meter_html, drivers_md, strat_md
    except Exception as e:
        error_html = f"<div style='color: red; padding: 10px;'>Error during inference: {str(e)}</div>"
        return error_html, "", f"Error: {str(e)}", "N/A"


# 5. Batch CSV Prediction Function (Decorated for ZeroGPU)
@gpu_decorator
def predict_batch_csv(file_obj):
    if file_obj is None:
        return "Please upload a valid CSV file.", None, None
    
    try:
        df = pd.read_csv(file_obj.name)
        scored_df = predictor.predict_dataframe(df)
        
        total = len(scored_df)
        churners = int((scored_df["Churn_Prediction"] == "Yes").sum())
        churn_rate = (churners / total * 100) if total else 0
        avg_risk = scored_df["Churn_Probability"].mean() * 100
        
        high_cnt = int((scored_df["Risk_Tier"] == "High").sum())
        med_cnt = int((scored_df["Risk_Tier"] == "Medium").sum())
        low_cnt = int((scored_df["Risk_Tier"] == "Low").sum())
        
        summary_md = f"""
        ### 📊 Batch Evaluation Summary
        * **Total Customers Analyzed:** `{total:,}`
        * **Predicted Churners:** `{churners:,}` (`{churn_rate:.1f}%` overall churn rate)
        * **Average Churn Risk:** `{avg_risk:.1f}%`
        * **Risk Distribution:** 🔴 **High:** `{high_cnt}` | 🟡 **Medium:** `{med_cnt}` | 🟢 **Low:** `{low_cnt}`
        """
        
        # Save scored CSV to temp file for download
        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix="_churn_scored.csv")
        scored_df.to_csv(tmp_output.name, index=False)
        
        preview = scored_df.head(50)
        return summary_md, preview, tmp_output.name
    except Exception as e:
        return f"❌ Error processing batch file: {str(e)}", None, None


# Custom CSS for modern UI
custom_css = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    padding: 24px 16px 12px 16px;
    background: linear-gradient(135deg, rgba(37,99,235,0.12) 0%, rgba(99,102,241,0.06) 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(59,130,246,0.2);
}
.persona-btn {
    font-size: 13px !important;
    padding: 6px 12px !important;
}
"""

with gr.Blocks(title="Telco Churn Intelligence Platform", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # ⚡ Telco Customer Churn Intelligence System
            ### **Enterprise Machine Learning Engine for Proactive Retention & Churn Risk Scoring**
            """
        )

    with gr.Tabs():
        
        # ==========================================
        # TAB 1: SINGLE CUSTOMER ASSESSMENT
        # ==========================================
        with gr.TabItem("👤 Single Customer Risk Assessment"):
            gr.Markdown("### ⚡ Quick-Load Persona Templates")
            with gr.Row():
                btn_high = gr.Button("🚨 Load High-Risk Newcomer", elem_classes=["persona-btn"], variant="secondary")
                btn_med = gr.Button("⚠️ Load Moderate-Risk DSL User", elem_classes=["persona-btn"], variant="secondary")
                btn_low = gr.Button("🛡️ Loyal Low-Risk Advocate", elem_classes=["persona-btn"], variant="secondary")

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("#### 📋 Customer Account & Demographics")
                    with gr.Row():
                        gender = gr.Dropdown(["Female", "Male"], value="Female", label="Gender")
                        senior_citizen = gr.Dropdown(["No", "Yes"], value="No", label="Senior Citizen")
                        partner = gr.Dropdown(["No", "Yes"], value="No", label="Partner")
                        dependents = gr.Dropdown(["No", "Yes"], value="No", label="Dependents")
                    
                    with gr.Row():
                        tenure = gr.Slider(minimum=0, maximum=72, value=2, step=1, label="Tenure (Months with Company)")
                        monthly_charges = gr.Number(value=89.50, label="Monthly Charges ($)")
                        total_charges = gr.Number(value=179.00, label="Total Lifetime Charges ($)")

                    with gr.Row():
                        contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract Type")
                        internet_service = gr.Dropdown(["Fiber optic", "DSL", "No"], value="Fiber optic", label="Internet Service")
                        payment_method = gr.Dropdown([
                            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                        ], value="Electronic check", label="Payment Method")

                    with gr.Accordion("🛠️ Subscribed Services & Value-Add Add-ons", open=True):
                        with gr.Row():
                            phone_service = gr.Dropdown(["Yes", "No"], value="Yes", label="Phone Service")
                            multiple_lines = gr.Dropdown(["No", "Yes", "No phone service"], value="No", label="Multiple Lines")
                            paperless_billing = gr.Dropdown(["Yes", "No"], value="Yes", label="Paperless Billing")
                        with gr.Row():
                            online_security = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Online Security")
                            online_backup = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Online Backup")
                            device_protection = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Device Protection")
                        with gr.Row():
                            tech_support = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Tech Support")
                            streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="Streaming TV")
                            streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="Streaming Movies")

                    predict_btn = gr.Button("⚡ Run Real-Time Churn Analysis", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("#### 🎯 Prediction Results & Retention Strategy")
                    out_badge = gr.HTML(label="Risk Badge")
                    out_meter = gr.HTML(label="Probability Meter")
                    out_drivers = gr.Markdown()
                    out_strategy = gr.Markdown()

            # Wire Persona Buttons
            all_inputs = [
                gender, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service, online_security,
                online_backup, device_protection, tech_support, streaming_tv,
                streaming_movies, contract, paperless_billing, payment_method,
                monthly_charges, total_charges
            ]
            
            btn_high.click(fn=lambda: load_persona_data("high_risk"), inputs=[], outputs=all_inputs)
            btn_med.click(fn=lambda: load_persona_data("moderate_risk"), inputs=[], outputs=all_inputs)
            btn_low.click(fn=lambda: load_persona_data("low_risk"), inputs=[], outputs=all_inputs)

            # Wire Prediction Button
            predict_btn.click(
                fn=predict_single_customer,
                inputs=all_inputs,
                outputs=[out_badge, out_meter, out_drivers, out_strategy]
            )

        # ==========================================
        # TAB 2: BATCH CSV INFERENCE
        # ==========================================
        with gr.TabItem("📁 Batch CSV Scoring"):
            gr.Markdown("### 📤 Upload Customer Batch Dataset for Bulk Scoring")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Telco Customers CSV", file_types=[".csv"])
                    batch_btn = gr.Button("🚀 Score Entire Batch", variant="primary")
                with gr.Column(scale=2):
                    batch_summary = gr.Markdown("Upload a CSV file and click 'Score Entire Batch' to view results.")
                    download_file = gr.File(label="📥 Download Scored CSV with Risk Probabilities")

            batch_preview = gr.Dataframe(label="Scored Customers Table (Top 50 Preview)", interactive=False)

            batch_btn.click(
                fn=predict_batch_csv,
                inputs=[file_input],
                outputs=[batch_summary, batch_preview, download_file]
            )

        # ==========================================
        # TAB 3: MODEL ANALYTICS & EDA PLOTS
        # ==========================================
        with gr.TabItem("📊 Model Performance & EDA Analytics"):
            gr.Markdown(
                """
                ### 🏆 Tuned Machine Learning Model Metrics
                * **Primary Model Architecture:** Tuned Random Forest Classifier with Cost-Sensitive Optimization
                * **Holdout ROC-AUC Score:** `0.8414`
                * **Churn Recall Rate:** `78.2%` (Maximized to capture high-risk cancellations)
                * **Decision Threshold:** `0.35` (Calibrated for financial ROI maximization)
                """
            )
            
            with gr.Row():
                cm_img = PLOTS_DIR / "confusion_matrix.png"
                roc_img = PLOTS_DIR / "roc_curve.png"
                if cm_img.exists():
                    gr.Image(value=str(cm_img), label="Confusion Matrix (Holdout Set)")
                if roc_img.exists():
                    gr.Image(value=str(roc_img), label="ROC-AUC Curve & Optimal Threshold")

            with gr.Row():
                fi_img = PLOTS_DIR / "feature_importance.png"
                mc_img = PLOTS_DIR / "model_comparison.png"
                if fi_img.exists():
                    gr.Image(value=str(fi_img), label="Top Churn Drivers (Feature Importance)")
                if mc_img.exists():
                    gr.Image(value=str(mc_img), label="Model Benchmark Comparison")

        # ==========================================
        # TAB 4: API REFERENCE & ARCHITECTURE
        # ==========================================
        with gr.TabItem("📑 System Architecture & API"):
            gr.Markdown(
                """
                ### 🏗️ Production System Architecture
                
                ```
                Raw Customer Data (CSV / REST API)
                             │
                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 🔄 Data Pipeline & Preprocessing Engine                 │
                │  - Categorical One-Hot Encoding                        │
                │  - Missing Value & Tenure Transformation              │
                │  - Standard Scaler on Financial Features              │
                └────────────────────────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 🧠 Tuned ML Model (Random Forest / Scikit-Learn)       │
                │  - Probability Scoring & Threshold Calibration         │
                └────────────────────────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 💡 Decision Engine & Explainability Layer              │
                │  - Churn Risk Tier: High (🔴), Medium (🟡), Low (🟢)   │
                │  - Risk Drivers Identification                         │
                │  - Prescriptive Retention Action Generation            │
                └────────────────────────────────────────────────────────┘
                ```

                ### 💻 Programmatic API Usage via Python `gradio_client`
                
                ```python
                from gradio_client import Client

                client = Client("adhnan001/telco-churn-engine")
                result = client.predict(
                    gender="Female",
                    senior_citizen="No",
                    partner="No",
                    dependents="No",
                    tenure=2,
                    phone_service="Yes",
                    multiple_lines="No",
                    internet_service="Fiber optic",
                    online_security="No",
                    online_backup="No",
                    device_protection="No",
                    tech_support="No",
                    streaming_tv="Yes",
                    streaming_movies="Yes",
                    contract="Month-to-month",
                    paperless_billing="Yes",
                    payment_method="Electronic check",
                    monthly_charges=89.5,
                    total_charges=179.0,
                    api_name="/predict_single_customer"
                )
                print(result)
                ```
                """
            )


# 6. Launch the Application with Gradio Queue
if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_api=False
    )
