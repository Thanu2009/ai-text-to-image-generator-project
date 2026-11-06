"""
Unit and Integration Test Suite for AI Text-to-Image Generator.
Validates routing, validation, error handling, Docker health endpoints, and security.

Single Source of Truth: brain.md
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from config import sanitize_secrets, Config
from model_service import TextToImageModelService, ModelServiceError


class TextToImageTestCase(unittest.TestCase):
    """Test suite for Flask routes, input validation, and service logic."""

    def setUp(self):
        """Configure test client."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_route(self):
        """Test GET / renders HTML UI successfully."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("AI Text-to-Image", html)
        self.assertIn("Docker Containerized", html)

    def test_health_route_for_docker(self):
        """Test GET /health returns 200 OK for Docker HEALTHCHECK."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "AI Text-to-Image Generator")

    def test_status_route(self):
        """Test GET /status returns system diagnostics."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("diagnostics", data)
        self.assertIn("configured_mode", data["diagnostics"])

    def test_empty_prompt_rejected(self):
        """Test POST /generate rejects empty prompt with HTTP 400."""
        response = self.client.post("/generate", json={"prompt": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error_type"), "EmptyPromptError")

    def test_whitespace_prompt_rejected(self):
        """Test POST /generate rejects whitespace-only prompt with HTTP 400."""
        response = self.client.post("/generate", json={"prompt": "     \n\t   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error_type"), "EmptyPromptError")

    def test_missing_prompt_field(self):
        """Test POST /generate rejects payload missing prompt key."""
        response = self.client.post("/generate", json={"other_key": "some text"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error_type"), "EmptyPromptError")

    def test_non_json_content_type_rejected(self):
        """Test POST /generate rejects non-JSON Content-Type with HTTP 400."""
        response = self.client.post("/generate", data="plain text prompt")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error_type"), "InvalidContentTypeError")

    def test_prompt_length_exceeded(self):
        """Test POST /generate rejects prompts exceeding 500 characters."""
        long_prompt = "a" * 501
        response = self.client.post("/generate", json={"prompt": long_prompt})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error_type"), "PromptLengthExceededError")

    def test_secret_sanitization(self):
        """Test that sanitize_secrets redacts Hugging Face tokens and keys."""
        sample_error = "Authentication failed with token hf_abc1234567890abcdef1234567890 and Bearer secret_12345"
        sanitized = sanitize_secrets(sample_error)
        self.assertNotIn("hf_abc1234567890abcdef1234567890", sanitized)
        self.assertIn("[REDACTED_HF_TOKEN]", sanitized)
        self.assertIn("Bearer [REDACTED_TOKEN]", sanitized)

    def test_download_missing_image_404(self):
        """Test GET /download/<filename> returns 404 for non-existent image."""
        response = self.client.get("/download/non_existent_image_12345.png")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
