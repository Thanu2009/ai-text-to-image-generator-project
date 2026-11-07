# ==============================================================================
# Production Dockerfile for AI Text-to-Image Generator
# Single Source of Truth: brain.md
# ==============================================================================

# Use official, stable Python 3.10 slim image for minimal footprint and security
FROM python:3.10-slim

# Set environment variables:
# - PYTHONUNBUFFERED=1 ensures real-time log output in docker logs
# - PYTHONDONTWRITEBYTECODE=1 avoids writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_PORT=5000 \
    FLASK_ENV=production

# Install essential system dependencies:
# - libgl1 and libglib2.0-0 for OpenCV / Pillow image decoding
# - curl for Docker container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency definition first for optimal Docker layer caching
COPY requirements-cloud.txt /app/requirements.txt

# Install Python packages inside the container
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code into container
COPY . /app/

# Ensure generated image storage directory exists with proper permissions
RUN mkdir -p /app/static/generated

# Expose Flask application port
EXPOSE 5000

# Docker Healthcheck: queries /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
   CMD curl -f "http://localhost:${PORT:-5000}/health" || exit 1

# Default execution command: Launch Flask server binding to 0.0.0.0
CMD ["python", "app.py"]
