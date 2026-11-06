"""
Flask Application for AI Text-to-Image Generator.
Handles HTTP endpoints, request validation, error formatting, and static file delivery.

Single Source of Truth: brain.md
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from config import Config, sanitize_secrets
from model_service import TextToImageModelService, ModelServiceError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")

# Initialize Flask application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # Max 2MB request payload

# Initialize AI Model Service
try:
    model_service = TextToImageModelService()
except Exception as init_err:
    logger.exception("Failed during initial model_service construction.")
    model_service = None
    _init_error = sanitize_secrets(str(init_err))
else:
    _init_error = None


@app.route("/", methods=["GET"])
def index():
    """
    Renders the modern dark AI UI homepage.
    Passes current system status, model information, and active device to template.
    """
    status_info = model_service.get_status_info() if model_service else {
        "configured_mode": Config.GENERATION_MODE,
        "model_id": Config.MODEL_ID,
        "device": "Unavailable",
        "torch_available": False,
        "diffusers_available": False,
        "is_local_pipeline_loaded": False,
        "has_hf_token": bool(Config.HF_TOKEN),
        "pipeline_load_error": _init_error
    }
    return render_template("index.html", status=status_info, current_year=datetime.now().year)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for Docker container monitoring and uptime validation.
    Returns HTTP 200 when container is responsive.
    """
    return jsonify({
        "status": "healthy",
        "service": "AI Text-to-Image Generator",
        "timestamp": datetime.now().isoformat(),
        "model_service_ready": model_service is not None
    }), 200


@app.route("/status", methods=["GET"])
def get_status():
    """Returns detailed machine learning and hardware acceleration diagnostics."""
    if not model_service:
        return jsonify({
            "status": "error",
            "message": "Model service failed to initialize.",
            "details": _init_error
        }), 500

    return jsonify({
        "status": "success",
        "diagnostics": model_service.get_status_info(),
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/generate", methods=["POST"])
def generate():
    """
    Primary endpoint for real AI text-to-image synthesis.
    Accepts JSON: {"prompt": "user description..."}
    Returns JSON with generated image URL, exact prompt, timing, and model metadata.
    """
    if not model_service:
        return jsonify({
            "status": "error",
            "error_type": "ServiceInitializationError",
            "message": "AI Model service is not initialized.",
            "details": _init_error or "Check server logs for dependency or configuration issues."
        }), 500

    # Ensure request is JSON
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error_type": "InvalidContentTypeError",
            "message": "Request Content-Type must be 'application/json'.",
            "details": "Send JSON payload with a 'prompt' string field."
        }), 400

    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({
            "status": "error",
            "error_type": "EmptyPromptError",
            "message": "No prompt field provided in request body.",
            "details": "Please enter a descriptive prompt to generate an image."
        }), 400

    raw_prompt = data.get("prompt")

    try:
        # Run real AI generation pipeline
        result = model_service.generate_image(raw_prompt)
        return jsonify(result), 200

    except ModelServiceError as model_err:
        # Log full traceback in terminal
        logger.exception(f"ModelServiceError during generation: {model_err.message}")

        # Map error types to appropriate HTTP status codes
        status_code = 500
        if model_err.error_type in ("EmptyPromptError", "InvalidPromptTypeError", "PromptLengthExceededError"):
            status_code = 400
        elif model_err.error_type == "MissingAuthenticationError":
            status_code = 401
        elif model_err.error_type == "ModelLoadingUpstreamError":
            status_code = 503
        elif model_err.error_type in ("ConnectionError", "NetworkTimeoutError"):
            status_code = 504

        return jsonify({
            "status": "error",
            "error_type": model_err.error_type,
            "message": model_err.message,
            "details": model_err.details,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), status_code

    except Exception as exc:
        sanitized_msg = sanitize_secrets(str(exc))
        logger.exception("Unexpected server exception during image generation.")
        return jsonify({
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": f"An unexpected server error occurred: {sanitized_msg}",
            "details": "Refer to the Python server console for full traceback.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_image(filename: str):
    """
    Secure file download endpoint for generated images.
    Prevents directory traversal and delivers PNG attachment.
    """
    # Sanitize filename: strip path components
    safe_filename = os.path.basename(filename)
    file_path = Config.OUTPUT_DIR / safe_filename

    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Download requested for missing file: {safe_filename}")
        abort(404, description=f"Requested image '{safe_filename}' not found.")

    return send_from_directory(
        directory=str(Config.OUTPUT_DIR),
        path=safe_filename,
        as_attachment=True,
        mimetype="image/png"
    )


@app.errorhandler(404)
def handle_404(err):
    """Handle 404 Not Found cleanly for both API and browser calls."""
    if request.path.startswith("/generate") or request.path.startswith("/status") or request.path.startswith("/health"):
        return jsonify({
            "status": "error",
            "error_type": "NotFoundError",
            "message": "The requested API endpoint was not found."
        }), 404
    return render_template("index.html", status=model_service.get_status_info() if model_service else {}), 404


@app.errorhandler(500)
def handle_500(err):
    """Handle 500 Internal Server Error cleanly."""
    logger.exception("Internal server error encountered.")
    return jsonify({
        "status": "error",
        "error_type": "InternalServerError",
        "message": "An internal server error occurred."
    }), 500


if __name__ == "__main__":
    logger.info(
        f"Starting AI Text-to-Image Generator on 0.0.0.0:{Config.PORT} "
        f"[Mode: {Config.GENERATION_MODE}, Debug: {Config.DEBUG}]"
    )
    # Binding to 0.0.0.0 is mandatory for Docker container port forwarding
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
