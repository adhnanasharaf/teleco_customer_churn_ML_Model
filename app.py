"""
Hugging Face Spaces Entrypoint (Gradio SDK + FastAPI Engine)
Mounts Gradio Blocks portal interface onto the existing FastAPI backend.
"""

import sys
from pathlib import Path
import importlib.util
import gradio as gr

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load FastAPI server module safely without colliding with root app.py name
server_path = BASE_DIR / "app" / "server.py"
spec = importlib.util.spec_from_file_location("server_api_module", server_path)
server_api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_api_module)
fastapi_app = server_api_module.app

# Define a clean Gradio Blocks portal interface
with gr.Blocks(title="Telco Churn Intelligence System") as demo:
    gr.Markdown(
        """
        # 📊 Telco Customer Churn Intelligence System
        
        Welcome to the **Telco Churn Intelligence Engine (TCIE)** on Hugging Face Spaces!
        
        The machine learning inference engine and full REST API services are active:
        
        ### 🚀 Quick Access Links:
        * 🌐 **[Open Full Interactive Web Dashboard](/static/index.html)** *(Single Customer Scoring, Batch CSV Processor, Model Diagnostic Visuals)*
        * 📑 **[Open FastAPI Interactive Swagger Documentation](/docs)** *(API Explorer & Schema Models)*
        * 🩺 **[Check System Health & Model Status](/health)**
        
        ---
        *Powered by Scikit-Learn, FastAPI, and Gradio.*
        """
    )

# Mount Gradio onto the existing FastAPI application at root
app = gr.mount_gradio_app(app=fastapi_app, blocks=demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
