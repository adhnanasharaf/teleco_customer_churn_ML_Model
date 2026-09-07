import importlib.util
from pathlib import Path
import sys

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

# 2. Compatibility shim: Fix Starlette >= 0.36 TemplateResponse argument order with Gradio
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

# 3. Compatibility hook for Hugging Face ZeroGPU environments
try:
    import spaces
    @spaces.GPU
    def _zero_gpu_inference_hook():
        return True
except Exception:
    pass

import gradio as gr
from fastapi.staticfiles import StaticFiles

# 3. Dynamically import the existing FastAPI app from app/server.py
server_path = Path(__file__).resolve().parent / "app" / "server.py"
spec = importlib.util.spec_from_file_location("server_module", server_path)
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
fastapi_app = server_module.app

# 4. Build the Gradio interface
with gr.Blocks(title="Telco Churn Intelligence System") as demo:
    gr.Markdown(
        """
        # 📊 Telco Customer Churn Intelligence System
        
        The model engine and API services are running:
        
        * 🌐 **[Open Interactive Full Dashboard](/static/index.html)**
        * 📑 **[FastAPI Swagger Documentation](/docs)**
        * 🩺 **[System Health Check](/health)**
        """
    )

# 5. Mount existing FastAPI routes and static assets onto the Gradio app
app = demo.app

# Mount the FastAPI router/endpoints onto Gradio's underlying Starlette/FastAPI app
for route in fastapi_app.routes:
    if route not in app.routes:
        app.routes.append(route)

# Ensure static files are accessible
static_dir = Path(__file__).resolve().parent / "app" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="custom_static")

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=7860)
