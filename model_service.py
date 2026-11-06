"""
AI Text-to-Image Generation Service.
Implements real text-to-image AI pipeline using PyTorch, Hugging Face Diffusers,
and Hugging Face Inference Client with Pillow image processing.

Strictly prompt-driven: No mock data, no placeholder images, no simulated outputs.
"""

import io
import os
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from config import Config, sanitize_secrets

logger = logging.getLogger(__name__)

# Check optional imports gracefully for inspection and initialization
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import diffusers
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    diffusers = None
    DIFFUSERS_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HF_HUB_AVAILABLE = False

import requests


class ModelServiceError(Exception):
    """Custom exception containing structured diagnostic details."""
    def __init__(self, message: str, error_type: str = "ModelServiceError", details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or ""


class TextToImageModelService:
    """
    Manages real text-to-image generation via Hugging Face Diffusers (local PyTorch pipeline)
    or Hugging Face Inference API.
    """

    def __init__(self):
        self.mode = Config.GENERATION_MODE
        self.model_id = Config.MODEL_ID
        self.device = self._detect_device()
        self.local_pipeline = None
        self.is_pipeline_loaded = False
        self.pipeline_load_error = None

        logger.info(
            f"Initializing TextToImageModelService [Mode: {self.mode}, "
            f"Model: {self.model_id}, Device: {self.device}]"
        )

        # Preload local diffusers pipeline if explicitly configured
        if self.mode == "diffusers":
            self._load_local_pipeline()

    def _detect_device(self) -> str:
        """Detect available hardware accelerator for PyTorch."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"cuda ({gpu_name})"
        return "cpu"

    def _load_local_pipeline(self):
        """
        Loads the pretrained Diffusers pipeline into memory/GPU.
        Raises an exception if dependencies or weights fail to load.
        """
        if not TORCH_AVAILABLE:
            err = "PyTorch is not installed. Please install torch with: pip install torch"
            self.pipeline_load_error = err
            logger.error(err)
            raise ModelServiceError(err, error_type="DependencyMissingError")

        if not DIFFUSERS_AVAILABLE:
            err = "Diffusers is not installed. Please install diffusers and transformers: pip install diffusers transformers"
            self.pipeline_load_error = err
            logger.error(err)
            raise ModelServiceError(err, error_type="DependencyMissingError")

        try:
            logger.info(f"Loading local diffusers model '{self.model_id}' on {self.device}...")
            
            # Determine PyTorch dtype based on device
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

            # Load the text-to-image pipeline
            self.local_pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                token=Config.HF_TOKEN if Config.HF_TOKEN else None,
                safety_checker=None  # Can be adjusted based on requirements
            )

            # Move to device
            if "cuda" in self.device:
                self.local_pipeline.to("cuda")
                # Optional memory optimization
                if hasattr(self.local_pipeline, "enable_attention_slicing"):
                    self.local_pipeline.enable_attention_slicing()

            self.is_pipeline_loaded = True
            self.pipeline_load_error = None
            logger.info("Local Diffusers pipeline successfully loaded into memory.")

        except Exception as exc:
            sanitized_msg = sanitize_secrets(str(exc))
            self.pipeline_load_error = sanitized_msg
            logger.exception("Failed to load local diffusers pipeline.")
            raise ModelServiceError(
                f"Failed to load local Diffusers model '{self.model_id}': {sanitized_msg}",
                error_type=exc.__class__.__name__,
                details="Verify that the model ID exists on Hugging Face, dependencies are installed, and sufficient VRAM/RAM is available."
            )

    def validate_prompt(self, prompt: Any) -> str:
        """
        Strict validation for user-provided prompts.
        Ensures prompt is non-empty, contains valid characters, and is within bounds.
        """
        if prompt is None:
            raise ModelServiceError(
                "Prompt was not provided in the request.",
                error_type="EmptyPromptError",
                details="Please enter a descriptive text prompt to generate an image."
            )

        if not isinstance(prompt, str):
            raise ModelServiceError(
                f"Invalid prompt type: expected string, received {type(prompt).__name__}.",
                error_type="InvalidPromptTypeError"
            )

        trimmed_prompt = prompt.strip()
        if not trimmed_prompt:
            raise ModelServiceError(
                "Prompt cannot be empty or only whitespace.",
                error_type="EmptyPromptError",
                details="Please enter words describing the scene, style, or subject you wish to generate."
            )

        if len(trimmed_prompt) > Config.MAX_PROMPT_LENGTH:
            raise ModelServiceError(
                f"Prompt exceeds maximum allowed length of {Config.MAX_PROMPT_LENGTH} characters "
                f"(current length: {len(trimmed_prompt)}).",
                error_type="PromptLengthExceededError"
            )

        return trimmed_prompt

    def generate_image(self, prompt_text: str) -> Dict[str, Any]:
        """
        Executes real AI text-to-image generation from the exact user prompt.
        
        Returns:
            Dictionary containing image paths, execution timing, metadata, and prompt details.
        
        Raises:
            ModelServiceError with actual underlying error details on failure.
        """
        # Strict prompt validation
        validated_prompt = self.validate_prompt(prompt_text)

        start_time = time.time()
        generated_pil_image: Optional[Image.Image] = None
        execution_backend = ""

        logger.info(f"Initiating real AI image generation for prompt: '{validated_prompt}'")

        # Determine generation execution route
        active_mode = self.mode

        if active_mode == "diffusers":
            generated_pil_image, execution_backend = self._generate_with_local_diffusers(validated_prompt)
        elif active_mode == "hf_api":
            generated_pil_image, execution_backend = self._generate_with_hf_api(validated_prompt)
        elif active_mode == "auto":
            # Auto mode: Use local diffusers if pipeline is loaded, otherwise use HF API
            if self.is_pipeline_loaded and self.local_pipeline is not None:
                generated_pil_image, execution_backend = self._generate_with_local_diffusers(validated_prompt)
            elif Config.HF_TOKEN:
                generated_pil_image, execution_backend = self._generate_with_hf_api(validated_prompt)
            elif TORCH_AVAILABLE and DIFFUSERS_AVAILABLE:
                # Try loading local pipeline on the fly
                try:
                    self._load_local_pipeline()
                    generated_pil_image, execution_backend = self._generate_with_local_diffusers(validated_prompt)
                except Exception:
                    # Fallback to HF API attempt
                    generated_pil_image, execution_backend = self._generate_with_hf_api(validated_prompt)
            else:
                generated_pil_image, execution_backend = self._generate_with_hf_api(validated_prompt)
        else:
            raise ModelServiceError(
                f"Unknown GENERATION_MODE: '{active_mode}'. Expected 'auto', 'diffusers', or 'hf_api'.",
                error_type="ConfigurationError"
            )

        if generated_pil_image is None or not isinstance(generated_pil_image, Image.Image):
            raise ModelServiceError(
                "Model execution completed but did not yield a valid PIL Image.",
                error_type="GenerationOutputError"
            )

        # Process and save the generated image
        generation_duration = round(time.time() - start_time, 2)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:8]
        filename = f"gen_{timestamp_str}_{unique_token}.png"
        filepath = Config.OUTPUT_DIR / filename

        try:
            # Save real generated image with Pillow
            generated_pil_image.save(filepath, format="PNG", optimize=True)
            logger.info(f"Successfully generated and saved real image to {filepath} in {generation_duration}s")
        except Exception as save_err:
            logger.exception("Failed to save generated image to disk.")
            raise ModelServiceError(
                f"Failed to write generated image to storage: {sanitize_secrets(str(save_err))}",
                error_type="DiskWriteError"
            )

        # Relative URL for browser consumption
        image_url = f"/static/generated/{filename}"

        return {
            "status": "success",
            "image_url": image_url,
            "filename": filename,
            "prompt": validated_prompt,
            "model": self.model_id,
            "execution_backend": execution_backend,
            "generation_time_seconds": generation_duration,
            "dimensions": f"{generated_pil_image.width}x{generated_pil_image.height}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _generate_with_local_diffusers(self, prompt: str) -> (Image.Image, str):
        """Generates image using local PyTorch Diffusers pipeline."""
        if not self.is_pipeline_loaded or self.local_pipeline is None:
            self._load_local_pipeline()

        try:
            logger.info(f"Running PyTorch Diffusers inference on {self.device}...")
            
            # Execute real model inference step
            result = self.local_pipeline(
                prompt=prompt,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                width=Config.IMAGE_WIDTH,
                height=Config.IMAGE_HEIGHT
            )

            if not hasattr(result, "images") or len(result.images) == 0:
                raise ModelServiceError("Diffusers pipeline returned empty image list.", error_type="ModelOutputError")

            image = result.images[0]
            backend_info = f"PyTorch Diffusers Local [{self.device}]"
            return image, backend_info

        except Exception as exc:
            sanitized = sanitize_secrets(str(exc))
            logger.exception("Exception during local PyTorch diffusers generation.")
            raise ModelServiceError(
                f"PyTorch Diffusers inference failed: {sanitized}",
                error_type=exc.__class__.__name__,
                details="Check memory constraints (VRAM/RAM) or reduce NUM_INFERENCE_STEPS / image dimensions."
            )

    def _generate_with_hf_api(self, prompt: str) -> (Image.Image, str):
        """
        Generates image using Hugging Face Inference API / Serverless Endpoint.
        Sends prompt to the real cloud-hosted diffusion model and converts returned bytes to PIL Image.
        """
        api_token = Config.HF_TOKEN
        if not api_token:
            raise ModelServiceError(
                "Hugging Face API token (HF_TOKEN) is not set in .env. "
                "To use the Hugging Face Inference API, provide a free user token from huggingface.co/settings/tokens, "
                "or install PyTorch and Diffusers to run the model locally.",
                error_type="MissingAuthenticationError",
                details="Set HF_TOKEN=your_token in .env or configure GENERATION_MODE=diffusers with PyTorch."
            )

        logger.info(f"Calling Hugging Face Inference API for model '{self.model_id}'...")

       
        # Use the modern Hugging Face Inference Providers client
        if not HF_HUB_AVAILABLE:
            raise ModelServiceError(
                "huggingface_hub is not installed correctly.",
                error_type="MissingDependencyError"
            )

        try:
            logger.info(
                f"Calling Hugging Face Inference Providers "
                f"for model '{self.model_id}'..."
            )

            client = InferenceClient(
                provider="auto",
                token=api_token
            )

            image = client.text_to_image(
                prompt=prompt,
                model=self.model_id,
                guidance_scale=Config.GUIDANCE_SCALE,
                num_inference_steps=Config.NUM_INFERENCE_STEPS
            )

            if isinstance(image, Image.Image):
                return image, f"Hugging Face Cloud API [{self.model_id}]"

            raise ModelServiceError(
                "Hugging Face returned an unexpected image response.",
                error_type="ImageDecodeError"
            )

        except ModelServiceError:
            raise

        except Exception as hf_err:
            sanitized_error = sanitize_secrets(str(hf_err))

            logger.exception(
                "Hugging Face Inference Providers request failed."
            )

            raise ModelServiceError(
                f"Hugging Face image generation failed: {sanitized_error}",
                error_type="HuggingFaceAPIError",
                details=sanitized_error
            )


    def get_status_info(self) -> Dict[str, Any]:
        """Returns the current operational status of the service."""
        return {
            "configured_mode": self.mode,
            "model_id": self.model_id,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "is_local_pipeline_loaded": self.is_pipeline_loaded,
            "has_hf_token": bool(Config.HF_TOKEN),
            "pipeline_load_error": self.pipeline_load_error
        }
