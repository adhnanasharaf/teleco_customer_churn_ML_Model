# ==============================================================================
# Production Dockerfile for Hugging Face Spaces (Docker SDK) & Cloud Deployments
# Meets all Hugging Face Spaces security standards (UID 1000, Port 7860)
# ==============================================================================

FROM python:3.11-slim

# Install system utilities & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set user home and path variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside user's home
WORKDIR $HOME/app

# Copy requirements and install dependencies in user namespace
COPY --chown=user:user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --user --upgrade -r requirements.txt

# Copy all application files and pre-trained artifacts with non-root ownership
COPY --chown=user:user . $HOME/app

# Ensure directories exist and permissions are granted
RUN mkdir -p $HOME/app/models $HOME/app/plots $HOME/app/data

# Expose official Hugging Face Spaces port
EXPOSE 7860

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI ASGI server on port 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
