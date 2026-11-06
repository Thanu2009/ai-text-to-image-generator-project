"""
Configuration Module for AI Text-to-Image Generator.
Loads environment variables and sets system-wide defaults.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if it exists
load_dotenv(BASE_DIR / ".env")

class Config:
    """Application configuration parameters."""
    SECRET_KEY = os.getenv("SECRET_KEY", "bca_t2i_secret_key_default_98234")
    PORT = int(os.getenv("FLASK_PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

    # Backend generation mode: 'auto', 'diffusers', or 'hf_api'
    GENERATION_MODE = os.getenv("GENERATION_MODE", "auto").lower()

    # Hugging Face API Token (optional for local diffusers, required for HF Inference API)
    HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

    # Model ID to load or query
    MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0").strip()

    # Image generation parameters
    NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", 30))
    GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
    IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", 512))
    IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", 512))

    # Limits and paths
    MAX_PROMPT_LENGTH = 500
    OUTPUT_DIR = BASE_DIR / "static" / "generated"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_secrets(text: str) -> str:
    """
    Remove sensitive credentials such as Hugging Face tokens or API keys
    from error messages and logs before returning to frontend or logging.
    """
    if not text:
        return ""
    
    sanitized = str(text)
    
    # Redact explicit HF_TOKEN if configured
    if Config.HF_TOKEN:
        sanitized = sanitized.replace(Config.HF_TOKEN, "[REDACTED_HF_TOKEN]")
    
    # Redact Hugging Face token patterns like hf_xxxxxxxxxxxxxxxxxxxxxxxx
    sanitized = re.sub(r"hf_[a-zA-Z0-9]{20,}", "[REDACTED_HF_TOKEN]", sanitized)

    # Redact Bearer tokens in headers
    sanitized = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer [REDACTED_TOKEN]", sanitized, flags=re.IGNORECASE)

    # Redact typical API key URL queries
    sanitized = re.sub(r"([?&]key=)[^&\s]+", r"\1[REDACTED_KEY]", sanitized)

    return sanitized
