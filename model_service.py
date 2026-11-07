
"""
AI Text-to-Image Generation Service.
Uses Cloudflare Workers AI with FLUX.1 schnell.

Strictly prompt-driven:
No mock data, placeholder images, or simulated outputs.
"""

import io
import os
import time
import uuid
import base64
import logging

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from PIL import Image

from config import Config, sanitize_secrets


logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Custom exception containing structured diagnostic details."""

    def __init__(
        self,
        message: str,
        error_type: str = "ModelServiceError",
        details: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or ""


class TextToImageModelService:
    """
    Generates real images using Cloudflare Workers AI.
    Model: FLUX.1 schnell
    """

    def __init__(self):

        self.mode = os.getenv(
            "GENERATION_MODE",
            "cloudflare"
        ).strip().lower()

        self.cloudflare_account_id = os.getenv(
            "CLOUDFLARE_ACCOUNT_ID", ""
        ).strip()

        self.cloudflare_api_token = os.getenv(
            "CLOUDFLARE_API_TOKEN", ""
        ).strip()

        self.model_id = os.getenv(
            "CLOUDFLARE_MODEL_ID",
            "@cf/black-forest-labs/flux-1-schnell"
        ).strip()

        self.device = "Cloudflare Workers AI"

        self.local_pipeline = None
        self.is_pipeline_loaded = False
        self.pipeline_load_error = None

        logger.info(
            f"Initializing image service "
            f"[Mode: {self.mode}, Model: {self.model_id}]"
        )

    def validate_prompt(self, prompt: Any) -> str:
        """Validate the user's image description."""

        if prompt is None:
            raise ModelServiceError(
                "Prompt was not provided.",
                error_type="EmptyPromptError"
            )

        if not isinstance(prompt, str):
            raise ModelServiceError(
                "Prompt must be a string.",
                error_type="InvalidPromptTypeError"
            )

        trimmed_prompt = prompt.strip()

        if not trimmed_prompt:
            raise ModelServiceError(
                "Prompt cannot be empty.",
                error_type="EmptyPromptError"
            )

        max_length = min(
            getattr(Config, "MAX_PROMPT_LENGTH", 2048),
            2048
        )

        if len(trimmed_prompt) > max_length:
            raise ModelServiceError(
                f"Prompt exceeds {max_length} characters.",
                error_type="PromptLengthExceededError"
            )

        return trimmed_prompt

    def generate_image(self, prompt_text: str) -> Dict[str, Any]:
        """
        Generate an image and save it in static/generated.
        """

        validated_prompt = self.validate_prompt(prompt_text)

        if self.mode != "cloudflare":
            raise ModelServiceError(
                "Invalid generation mode. Set GENERATION_MODE=cloudflare in .env.",
                error_type="ConfigurationError"
            )

        start_time = time.time()

        logger.info(
            f"Starting Cloudflare image generation for prompt: "
            f"{validated_prompt}"
        )

        generated_image, execution_backend = (
            self._generate_with_cloudflare(validated_prompt)
        )

        if not isinstance(generated_image, Image.Image):
            raise ModelServiceError(
                "Cloudflare did not return a valid image.",
                error_type="GenerationOutputError"
            )

        generation_duration = round(
            time.time() - start_time, 2
        )

        timestamp_str = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        unique_token = uuid.uuid4().hex[:8]

        filename = (
            f"gen_{timestamp_str}_{unique_token}.png"
        )

        output_directory = Path(Config.OUTPUT_DIR)
        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        filepath = output_directory / filename

        try:
            generated_image.save(
                filepath,
                format="PNG",
                optimize=True
            )

            logger.info(
                f"Generated image saved successfully: {filepath}"
            )

        except Exception as save_error:
            logger.exception("Image saving failed.")

            raise ModelServiceError(
                f"Failed to save generated image: "
                f"{sanitize_secrets(str(save_error))}",
                error_type="DiskWriteError"
            )

        image_url = f"/static/generated/{filename}"

        return {
            "status": "success",
            "image_url": image_url,
            "filename": filename,
            "prompt": validated_prompt,
            "model": self.model_id,
            "execution_backend": execution_backend,
            "generation_time_seconds": generation_duration,
            "dimensions": (
                f"{generated_image.width}x"
                f"{generated_image.height}"
            ),
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

    def _generate_with_cloudflare(self, prompt: str):
        """
        Send the prompt to Cloudflare Workers AI
        and convert the Base64 response into a PIL image.
        """

        if not self.cloudflare_account_id:
            raise ModelServiceError(
                "Cloudflare Account ID is missing from .env.",
                error_type="MissingConfigurationError"
            )

        if not self.cloudflare_api_token:
            raise ModelServiceError(
                "Cloudflare API token is missing from .env.",
                error_type="MissingAuthenticationError"
            )

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.cloudflare_account_id}/ai/run/"
            f"{self.model_id}"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.cloudflare_api_token}"
            ),
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "steps": 4
        }

        try:
            logger.info(
                f"Calling Cloudflare Workers AI model: {self.model_id}"
            )

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(15, 180)
            )

            if not response.ok:

                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "errors", error_data
                    )
                except ValueError:
                    error_message = response.text[:1000]

                raise ModelServiceError(
                    f"Cloudflare API returned HTTP "
                    f"{response.status_code}: "
                    f"{sanitize_secrets(str(error_message))}",
                    error_type="CloudflareAPIError",
                    details=(
                        f"HTTP status: {response.status_code}"
                    )
                )

            try:
                response_data = response.json()

            except ValueError:
                raise ModelServiceError(
                    "Cloudflare returned an invalid JSON response.",
                    error_type="InvalidAPIResponseError"
                )

            result = response_data.get(
                "result", response_data
            )

            if not isinstance(result, dict):
                raise ModelServiceError(
                    "Unexpected Cloudflare response structure.",
                    error_type="InvalidAPIResponseError"
                )

            image_base64 = result.get("image")

            if not image_base64:
                raise ModelServiceError(
                    "Cloudflare response did not contain image data.",
                    error_type="MissingImageDataError"
                )

            try:
                image_bytes = base64.b64decode(
                    image_base64,
                    validate=True
                )

                image_buffer = io.BytesIO(image_bytes)

                with Image.open(image_buffer) as opened_image:
                    opened_image.load()
                    generated_image = opened_image.convert("RGB")

            except Exception as image_error:
                raise ModelServiceError(
                    f"Could not decode generated image: "
                    f"{sanitize_secrets(str(image_error))}",
                    error_type="ImageDecodeError"
                )

            logger.info(
                "Cloudflare successfully generated an image."
            )

            backend = (
                f"Cloudflare Workers AI [{self.model_id}]"
            )

            return generated_image, backend

        except ModelServiceError:
            raise

        except requests.Timeout:
            logger.exception("Cloudflare request timed out.")

            raise ModelServiceError(
                "Cloudflare image generation request timed out.",
                error_type="CloudflareTimeoutError"
            )

        except requests.RequestException as request_error:
            logger.exception("Cloudflare connection failed.")

            raise ModelServiceError(
                f"Cloudflare connection failed: "
                f"{sanitize_secrets(str(request_error))}",
                error_type="CloudflareConnectionError"
            )

        except Exception as error:
            logger.exception(
                "Unexpected Cloudflare generation error."
            )

            raise ModelServiceError(
                f"Unexpected image generation error: "
                f"{sanitize_secrets(str(error))}",
                error_type="CloudflareGenerationError"
            )

    def get_status_info(self) -> Dict[str, Any]:
        """Return the current service configuration."""

        return {
            "configured_mode": self.mode,
            "model_id": self.model_id,
            "device": self.device,
            "torch_available": False,
            "diffusers_available": False,
            "is_local_pipeline_loaded": False,
            "has_hf_token": False,
            "has_cloudflare_token": bool(
                self.cloudflare_api_token
            ),
            "cloudflare_model_id": self.model_id,
            "pipeline_load_error": None
        }
