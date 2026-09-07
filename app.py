"""
Production Entry Point for Telco Customer Churn Intelligence System
Integrates FastAPI backend with Gradio UI and Static Interactive Dashboard.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Set up base directories
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 1. Compatibility shim: Ensure HfFolder is present if huggingface_hub >= 0.24 is loaded
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

# 2. Compatibility shim: Fix Starlette TemplateResponse signature differences with Gradio
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse

    def _patched_template_response(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get("context", {})
            request = kwargs.get("request", context.get("request") if isinstance(context, dict) else None)
            new_args = (request, name) + args[2:]
            kwargs["context"] = context
            return _orig_template_response(self, *new_args, **kwargs)
        return _orig_template_response(self, *args, **kwargs)

    starlette.templating.Jinja2Templates.TemplateResponse = _patched_template_response
except Exception:
    pass

import gradio as gr

# 3. Import the FastAPI backend application
server_path = BASE_DIR / "app" / "server.py"
spec = importlib.util.spec_from_file_location("server_module", server_path)
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
fastapi_app = server_module.app

# 4. Gradio Prediction Helper
def gradio_predict(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):
    try:
        predictor = server_module.get_predictor()
        customer_data = {
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
        res = predictor.predict_single(customer_data)
        
        prob_pct = f"{res['churn_probability'] * 100:.1f}%"
        tier = res['risk_tier']
        prediction = res['churn_prediction']
        
        drivers_md = "\n".join([f"- {d}" for d in res.get("top_risk_factors", [])]) or "No major risk factors detected."
        strategies_md = "\n".join([f"- {s}" for s in res.get("retention_strategies", [])]) or "Customer is satisfied and engaged."
        
        return prediction, prob_pct, tier, drivers_md, strategies_md
    except Exception as e:
        return "Error", f"Error: {str(e)}", "N/A", str(e), "N/A"


# 5. Build Gradio Interface
custom_css = """
.nav-links-bar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 12px 0 20px 0;
}
.nav-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: #2563eb;
    color: white !important;
    text-decoration: none !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s ease;
}
.nav-link-btn:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
}
.nav-link-btn.secondary {
    background: #0f172a;
    border: 1px solid #334155;
}
.nav-link-btn.secondary:hover {
    background: #1e293b;
}
"""

with gr.Blocks(title="Telco Churn Intelligence Platform", css=custom_css) as demo:
    gr.Markdown(
        """
        # ⚡ Telco Customer Churn Intelligence System
        **Production-grade Predictive Churn Modeling, Risk Assessment & Automated Retention Engine**
        
        <div class="nav-links-bar">
            <a class="nav-link-btn" href="/static/index.html" target="_blank">🌐 Open Full Interactive Dashboard</a>
            <a class="nav-link-btn secondary" href="/docs" target="_blank">📑 FastAPI Swagger Docs</a>
            <a class="nav-link-btn secondary" href="/health" target="_blank">🩺 Health Check</a>
        </div>
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("👤 Quick Churn Assessment"):
            gr.Markdown("### Input Customer Profile for Instant Churn Risk Scoring")
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gender = gr.Dropdown(["Female", "Male"], value="Female", label="Gender")
                        senior_citizen = gr.Dropdown(["No", "Yes"], value="No", label="Senior Citizen")
                        partner = gr.Dropdown(["No", "Yes"], value="No", label="Partner")
                        dependents = gr.Dropdown(["No", "Yes"], value="No", label="Dependents")
                    
                    with gr.Row():
                        tenure = gr.Slider(minimum=0, maximum=72, value=2, step=1, label="Tenure (Months)")
                        monthly_charges = gr.Number(value=89.5, label="Monthly Charges ($)")
                        total_charges = gr.Number(value=179.0, label="Total Charges ($)")
                    
                    with gr.Row():
                        contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract Type")
                        internet_service = gr.Dropdown(["Fiber optic", "DSL", "No"], value="Fiber optic", label="Internet Service")
                        payment_method = gr.Dropdown([
                            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                        ], value="Electronic check", label="Payment Method")
                    
                    with gr.Accordion("Additional Services & Add-ons", open=False):
                        with gr.Row():
                            phone_service = gr.Dropdown(["Yes", "No"], value="Yes", label="Phone Service")
                            multiple_lines = gr.Dropdown(["No", "Yes", "No phone service"], value="No", label="Multiple Lines")
                            paperless_billing = gr.Dropdown(["Yes", "No"], value="Yes", label="Paperless Billing")
                        with gr.Row():
                            online_security = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Online Security")
                            online_backup = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Online Backup")
                            device_protection = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Device Protection")
                            tech_support = gr.Dropdown(["No", "Yes", "No internet service"], value="No", label="Tech Support")
                            streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="Streaming TV")
                            streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="Streaming Movies")

                    predict_btn = gr.Button("⚡ Analyze Customer Churn Risk", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Risk Assessment Results")
                    out_pred = gr.Textbox(label="Prediction", interactive=False)
                    out_prob = gr.Textbox(label="Churn Probability", interactive=False)
                    out_tier = gr.Textbox(label="Risk Tier", interactive=False)
                    out_drivers = gr.Markdown(label="Key Risk Drivers")
                    out_strategy = gr.Markdown(label="Prescriptive Retention Action Plan")

            predict_btn.click(
                fn=gradio_predict,
                inputs=[
                    gender, senior_citizen, partner, dependents, tenure,
                    phone_service, multiple_lines, internet_service, online_security,
                    online_backup, device_protection, tech_support, streaming_tv,
                    streaming_movies, contract, paperless_billing, payment_method,
                    monthly_charges, total_charges
                ],
                outputs=[out_pred, out_prob, out_tier, out_drivers, out_strategy]
            )

        with gr.TabItem("🌐 Embedded Full Dashboard"):
            gr.Markdown(
                """
                ### Full Analytics & Batch Scoring Dashboard
                You can also open the dashboard in a dedicated tab: [Open in New Window](/static/index.html)
                """
            )
            gr.HTML('<iframe src="/static/index.html" style="width: 100%; height: 850px; border: 1px solid #e2e8f0; border-radius: 12px;"></iframe>')

        with gr.TabItem("📑 API Documentation & Health"):
            gr.Markdown(
                """
                ### API & Services Overview
                
                The backend service exposes standard RESTful endpoints for enterprise system integration:
                
                * 📑 **Interactive OpenAPI / Swagger UI**: [`/docs`](/docs)
                * 📑 **Alternative ReDoc UI**: [`/redoc`](/redoc)
                * 🩺 **Health Check Probe**: [`/health`](/health)
                * 📊 **Model Metadata & Metrics**: [`/api/model-info`](/api/model-info)
                * ⚡ **Single Inference Endpoint**: `POST /api/predict`
                * 📁 **Batch CSV Inference Endpoint**: `POST /api/predict-batch`
                """
            )

# 6. Mount Gradio onto the FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # On Hugging Face Spaces, user apps MUST bind to 7860 (HF uses other ports like 7861 internally)
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_REPO_NAME") or os.environ.get("SYSTEM") == "spaces":
        port = 7860
    else:
        port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Telco Churn Intelligence Platform starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
