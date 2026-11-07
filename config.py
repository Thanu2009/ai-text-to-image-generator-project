
"""
Configuration Module for AI Text-to-Image Generator.
Loads environment variables and sets system-wide defaults.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Project base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration parameters."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-development-key")

    # Generation settings
    GENERATION_MODE = os.getenv("GENERATION_MODE", "cloudflare").strip().lower()

    # Cloudflare Workers AI settings
    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    CLOUDFLARE_MODEL_ID = os.getenv(
        "CLOUDFLARE_MODEL_ID",
        "@cf/black-forest-labs/flux-1-schnell"
    ).strip()

    MODEL_ID = CLOUDFLARE_MODEL_ID

    # Image settings
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2048"))
    OUTPUT_DIR = BASE_DIR / "static" / "generated"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Server settings
    PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"


def sanitize_secrets(value):
    """Redact credentials from error messages and logs."""

    if value is None:
        return value

    text = str(value)

    secrets = [
        Config.CLOUDFLARE_API_TOKEN,
        Config.CLOUDFLARE_ACCOUNT_ID,
        os.getenv("HF_TOKEN", ""),
        os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
    ]

    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")

    text = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s,'\"]+",
        r"\1[REDACTED]",
        text
    )

    text = re.sub(
        r"(?i)(api[_-]?key|token|secret)(\s*[=:]\s*)[^\s,'\"]+",
        r"\1\2[REDACTED]",
        text
    )

    return text
