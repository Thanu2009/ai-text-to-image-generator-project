# AI Text-to-Image Generator — Project Brain (`brain.md`)

> **Single Source of Truth & Authoritative Project Reference**  
> *Version:* 2.0.0 (Mandatory Docker Containerized Architecture)  
> *Created:* 2026-11-04 | *Updated:* 2026-11-04  
> *Target Academic Level:* BCA Final Year Major Project / Demonstration  
> *Repository Root:* `ai_text_to_image_generator/`

---

## 1. Project Overview

### 1.1 Project Name & Purpose
- **Project Name:** AI Text-to-Image Generator
- **Purpose:** A complete, prompt-driven deep learning web application that translates natural language text prompts into high-resolution, contextually accurate visual images using real text-to-image machine learning models.
- **Mandatory Execution Paradigm:** **Docker strictly required.** All execution, testing, and deployment must run inside isolated Docker containers. No host Python packages required.

### 1.2 Problem Statement
Traditional computer vision and graphic design require extensive manual modeling or stock photography searches. Generative AI allows users to describe concepts in plain English and synthesize original visual imagery on demand. However, educational and production implementations frequently struggle with:
1. Fake/mock integrations that simulate AI rather than running real neural network inference.
2. Silent failures and opaque error messages when GPU VRAM or cloud quotas expire.
3. Complex host environment dependencies (CUDA versions, Python package conflicts, OS-specific build errors).

This project resolves these issues by packaging a real deep learning pipeline (PyTorch + Hugging Face Diffusers + Pillow) inside an immutable, production-grade **Docker** container with zero mock data and robust diagnostic reporting.

### 1.3 Project Objectives
- **100% Real AI Execution:** Guarantee that every single output image is synthesized by a real neural network (Latent Diffusion / Transformer) from the user's exact prompt.
- **Zero Mock / Predefined Data:** Strictly prohibit hardcoded prompts, mock image folders, or cached responses.
- **Docker-First Containerization:** Complete application runtime, dependencies, PyTorch tensors, and web services isolated in Docker via `docker compose`.
- **Dual Pipeline Architecture:** Support both local PyTorch Diffusers inference (with CPU/CUDA acceleration) and Hugging Face Inference API (for low-spec client machines).
- **Comprehensive Error Diagnostics:** Deliver transparent error classes, full stack trace logging, and helpful troubleshooting advice without exposing private API keys or tokens.
- **Modern Dark UI:** Provide a responsive, glassmorphic interface with real-time generation timers and instant downloads.

### 1.4 Target Users
- **BCA Project Examiners & Evaluators:** Reviewing real-world machine learning code, REST API architecture, prompt engineering pipelines, and containerization best practices.
- **Students & AI Learners:** Studying the interaction between PyTorch, Diffusers, latent spaces, Docker, and web frameworks.
- **End Users & Creators:** Generating images for rapid conceptualization and design prototyping.

### 1.5 Technology Stack
| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Container Engine** | Docker / Docker Compose | `>=20.10 / v2` | Containerization, volume management, health checks, environment isolation |
| **Container Base** | `python:3.10-slim` | Debian-based | Lightweight, secure base with `curl`, `libgl1`, and `libglib2.0-0` |
| **Backend Framework** | Python / Flask | `>=3.0.0` | Routing, REST API endpoints, request validation, error formatting |
| **Deep Learning Engine** | PyTorch (`torch`) | `>=2.1.0` | Tensor computation, GPU acceleration (`cuda`), autograd execution |
| **Generative Pipeline** | Hugging Face Diffusers | `>=0.27.0` | Stable Diffusion / SDXL latent diffusion pipeline management |
| **Cloud Inference Client** | Hugging Face Hub / REST | `>=0.21.0` | Serverless cloud model inference via Hugging Face API |
| **Image Processing** | Pillow (`PIL`) | `>=10.0.0` | Binary image decoding, color verification, file saving to disk |
| **Environment & Config** | `python-dotenv` | `>=1.0.0` | Environment variable management and `.env` parsing |
| **Frontend Markup** | HTML5 (Semantic) | Modern | Accessible layout, prompt input, status displays, modal preview |
| **Frontend Styling** | CSS3 (Custom Glassmorphism) | Modern | Dark AI aesthetic, CSS variables, flex/grid layouts, responsive media queries |
| **Frontend Scripting** | Vanilla JavaScript (ES6+) | Modern | Asynchronous `fetch` calls, validation, elapsed timer, DOM updates |

---

## 2. Docker Architecture & Container Configuration

### 2.1 Container Specifications
- **Base Image:** `python:3.10-slim`
- **System Dependencies:** `curl` (container health probes), `libgl1` (OpenGL runtime for Pillow/graphics), `libglib2.0-0` (GLib runtimes).
- **Exposed Port:** `5000` (mapped to host `5000:5000`).
- **Network Mode:** Bridge network (`t2i-network`).
- **Process Binding:** `0.0.0.0:5000` inside container to allow host traffic forwarding.
- **Container Name:** `ai_text_to_image_app`.
- **Restart Policy:** `unless-stopped`.

### 2.2 Volume Mounts & Persistence
1. **Generated Images Mount (Host Bind Mount):**
   - Host path: `./static/generated`
   - Container path: `/app/static/generated`
   - **Purpose:** Every PNG synthesized by the AI model inside the container is saved directly to the host disk, surviving container restarts, image rebuilds, and deletions.
2. **Model Weights Cache (Named Docker Volume):**
   - Volume name: `hf_cache`
   - Container path: `/root/.cache/huggingface`
   - **Purpose:** Prevents re-downloading multi-gigabyte neural network weights whenever the container restarts.

### 2.3 Healthcheck Configuration
- **Endpoint:** `GET /health` (implemented in `app.py`).
- **Docker Parameters:**
  - `interval`: 30 seconds
  - `timeout`: 10 seconds
  - `start_period`: 20 seconds
  - `retries`: 3
  - `test`: `["CMD", "curl", "-f", "http://localhost:5000/health"]`

### 2.4 GPU Acceleration Configuration
- By default, the container executes on standard multi-core CPU.
- When an NVIDIA GPU and NVIDIA Container Toolkit are installed, GPU acceleration is unlocked in `docker-compose.yml`:
  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  ```

### 2.5 Mandatory Docker Workflow Commands
```bash
# 1. Build container image
docker compose build

# 2. Start container in detached background mode
docker compose up -d

# 3. Stream real-time container logs and traces
docker compose logs -f

# 4. Execute automated test suite inside container
docker compose exec ai-generator python -m unittest discover tests -v

# 5. Stop container services safely
docker compose down
```

---

## 3. Project Architecture & File Structure

### 3.1 Complete Folder and File Structure
```text
ai_text_to_image_generator/
├── Dockerfile                   # Production Python 3.10 slim container definition
├── docker-compose.yml           # Docker Compose service, volumes, network, & healthcheck
├── .dockerignore                # Excludes local artifacts and secrets from container image
├── brain.md                     # Single Source of Truth & Authoritative Project Guide
├── app.py                       # Flask server (routes, error handling, Docker 0.0.0.0 binding)
├── config.py                    # Environment loader, parameters, and secret sanitization
├── model_service.py             # Core AI engine (Diffusers, PyTorch, HF API, Pillow)
├── requirements.txt             # Pinned Python package dependencies
├── .env.example                 # Template environment configuration
├── .env                         # Your active configuration (contains HF_TOKEN)
├── README.md                    # Installation, usage guide, and BCA Viva Q&A documentation
├── templates/
│   └── index.html               # Main user interface template
├── static/
│   ├── css/
│   │   └── style.css            # Dark AI theme styling, animations, and responsive layout
│   ├── js/
│   │   └── main.js              # Client-side validation, AJAX requests, loading state, DOM management
│   └── generated/               # Output directory storing real generated PNG images
└── tests/
    └── test_app.py              # Automated test suite (validation, error trapping, routes, security)
```

### 3.2 Purpose of Every Important File
- **`Dockerfile`:** Packages Python 3.10, system libraries (`curl`, `libgl1`), dependencies from `requirements.txt`, application code, and health check into a container image.
- **`docker-compose.yml`:** Defines the `ai-generator` service, environment variable passthrough, persistent volume mappings, and container orchestration.
- **`.dockerignore`:** Prevents local development caches (`__pycache__`, `.env`, git history, local test results) from bloating the Docker image.
- **`brain.md`:** The comprehensive reference document governing architecture, coding rules, features, Docker workflows, and change management.
- **`config.py`:** Reads `.env`, sets defaults for server port, model IDs, inference parameters, and provides `sanitize_secrets()` to strip credentials from errors.
- **`model_service.py`:** Encapsulates `TextToImageModelService`. Validates prompt text, loads the local PyTorch `AutoPipelineForText2Image` or queries the Hugging Face Inference API, converts raw outputs into `PIL.Image`, and saves `.png` files.
- **`app.py`:** Initializes the Flask application, exposes endpoints (`/`, `/health`, `/generate`, `/status`, `/download/<filename>`), binds to `0.0.0.0`, handles HTTP error codes, and logs full tracebacks.
- **`requirements.txt`:** Specifies dependencies (`flask`, `torch`, `diffusers`, `transformers`, `accelerate`, `Pillow`, `huggingface_hub`, `requests`, `python-dotenv`).
- **`.env.example`:** Documentation for configuration variables (`GENERATION_MODE`, `HF_TOKEN`, `MODEL_ID`, `NUM_INFERENCE_STEPS`, `GUIDANCE_SCALE`).
- **`templates/index.html`:** Semantic HTML5 interface containing prompt textarea, status badges, Docker environment indicator, generation progress spinner with elapsed timer, and image display card.
- **`static/css/style.css`:** Implements dark AI glassmorphism styling (`#0a0e17` dark base, `#8b5cf6` violet accents, `#06b6d4` cyan glows) with responsive layouts.
- **`static/js/main.js`:** Handles form submission, character counter, timer countdown/countup, async communication with `/generate`, error alert rendering, and download triggering.
- **`tests/test_app.py`:** Unit and integration tests covering prompt validation (empty, whitespace, length), `/health` Docker endpoint, error status mapping, and credential sanitization.

### 3.3 Flask Routes and Responsibilities
| HTTP Method | Route | Purpose | Expected Payload | Return Type |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the main UI with initial model and system configuration | None | `text/html` |
| `GET` | `/health` | Dedicated health check endpoint for Docker container probes | None | `application/json` (HTTP 200) |
| `POST` | `/generate` | Accepts prompt, runs real AI model, saves image, returns URL & metadata | `{"prompt": "..."}` | `application/json` |
| `GET` | `/status` | Diagnostic route returning active backend mode, device, and PyTorch status | None | `application/json` |
| `GET` | `/download/<filename>` | Secure file download endpoint with `as_attachment=True` | URL param `filename` | `image/png` |

---

## 4. AI/ML Model Documentation

### 4.1 Model Name and Source
- **Default Recommended Model:** `stabilityai/stable-diffusion-xl-base-1.0` (Hugging Face)
- **Supported Alternative Lightweight Model:** `segmind/tiny-sd` (547M params, optimized for low-latency CPU demonstrations)
- **Supported Classic Standard Model:** `runwayml/stable-diffusion-v1-5` (1.0B params, balanced VRAM footprint)
- **Supported Modern DiT Model:** `black-forest-labs/FLUX.1-schnell` (Next-gen 12B Diffusion Transformer)

### 4.2 Model Architecture and Type
The underlying machine learning architecture is a **Latent Diffusion Model (LDM)**:
1. **Text Encoder (CLIP ViT-L/14 or OpenCLIP):** Translates words and tokens into high-dimensional semantic vector embeddings (\(768\) or \(1024\) dimensions).
2. **Latent Space (VAE Encoder/Decoder):** Operates in a compressed latent space \(128 \times 128 \times 4\) representing an 8x spatial reduction.
3. **Denoising Core (U-Net with Cross-Attention or Diffusion Transformer):** Iteratively predicts and subtracts noise conditioned on the text embeddings via cross-attention layers.
4. **Variational Autoencoder (VAE) Decoder:** Converts the final denoised latent representation back into human-perceptible RGB pixel arrays.

### 4.3 Image Generation Parameters
- `NUM_INFERENCE_STEPS`: `30` (balanced between quality and inference time)
- `GUIDANCE_SCALE`: `7.5` (Classifier-Free Guidance weight balancing prompt adherence vs. image diversity)
- `IMAGE_WIDTH`: `512` (scalable to `1024` for SDXL)
- `IMAGE_HEIGHT`: `512` (scalable to `1024` for SDXL)

---

## 5. Strict Project Rules

1. **Zero Mock Data:**
   - Never use fake or simulated image generators.
   - Never use placeholder image services.
   - Never display pre-existing static images masquerading as generated results.
2. **Exact Prompt Driven:**
   - Every generated image must be synthesized solely from the exact prompt text entered by the user.
3. **No Silent Substitutions:**
   - If model generation fails, do not return a fallback stock image. Return the actual error message.
   - Never display a cached image from a previous run as the output of a new generation.
4. **Strict Empty Prompt Rejection:**
   - If prompt is empty, null, or whitespace-only, abort immediately and return HTTP 400 with `EmptyPromptError`.
5. **Authentic Diagnostic Errors:**
   - Display real Python and API error classes.
   - Print complete tracebacks to the server terminal using `logging.exception()`.
6. **Credential Privacy & Security:**
   - Never expose Hugging Face tokens, passwords, or system secrets in UI error cards, client responses, or version control.
   - Always run error strings through `sanitize_secrets()` before transmitting to clients.
7. **Mandatory Docker Execution:**
   - Do not instruct users to execute `python app.py` on their host machines. All workflows run through Docker.

---

## 6. Coding Standards

### 6.1 Python & Flask Conventions
- Adhere strictly to **PEP 8** style guidelines.
- Use explicit type annotations (`def generate_image(prompt: str) -> Dict[str, Any]:`).
- Include docstrings on all classes, public methods, and utility functions.
- Use `pathlib.Path` for cross-platform filesystem path resolution.
- Use structured logging via `logging.getLogger("app")`.

### 6.2 Frontend Conventions
- Semantic HTML5 elements (`<header>`, `<main>`, `<section>`, `<article>`, `<textarea>`, `<button>`).
- Vanilla JavaScript written in modern ES6+ syntax.
- Custom dark glassmorphism CSS without bloated third-party CSS frameworks.

### 6.3 Security & Secret Redaction
- Exclude `.env` from Docker images via `.dockerignore`.
- Scrub tokens using regex patterns in `sanitize_secrets()`.

---

## 7. Testing and Validation

### 7.1 Automated Test Matrix (`tests/test_app.py`)
Executed inside container via `docker compose exec ai-generator python -m unittest discover tests -v`:
| Test Case | Method | Description | Expected Result |
| :--- | :--- | :--- | :--- |
| `test_homepage_route` | `GET /` | Verifies the main UI loads successfully | HTTP 200, HTML content returned |
| `test_health_route_for_docker` | `GET /health` | Verifies Docker healthcheck probe | HTTP 200, `status: "healthy"` |
| `test_status_route` | `GET /status` | Verifies status endpoint outputs valid JSON | HTTP 200, contains diagnostics |
| `test_empty_prompt_rejected` | `POST /generate` | Submits empty string `""` | HTTP 400, `error_type: "EmptyPromptError"` |
| `test_whitespace_prompt_rejected` | `POST /generate` | Submits whitespace string `"   "` | HTTP 400, `error_type: "EmptyPromptError"` |
| `test_missing_prompt_field` | `POST /generate` | Submits JSON without prompt key | HTTP 400, `error_type: "EmptyPromptError"` |
| `test_non_json_content_type_rejected` | `POST /generate` | Submits raw non-JSON text | HTTP 400, `error_type: "InvalidContentTypeError"` |
| `test_prompt_length_exceeded` | `POST /generate` | Submits string `> 500` characters | HTTP 400, `error_type: "PromptLengthExceededError"` |
| `test_secret_sanitization` | Unit | Passes string with `hf_...` token to `sanitize_secrets()` | Returns string with `[REDACTED_HF_TOKEN]` |
| `test_download_missing_image_404` | `GET /download/fake.png` | Attempts to download missing file | HTTP 404 |

---

## 8. Change Management Protocol

Whenever a change, feature request, bug fix, or modification is requested, follow this strict protocol:

1. **Step 1 — Read `brain.md`:** Review before creating or modifying any codebase files to ensure alignment with existing patterns and Docker requirements.
2. **Step 2 — Impact Analysis:** Identify which specific files need changes (`Dockerfile`, `docker-compose.yml`, `app.py`, `model_service.py`, `config.py`, templates, CSS, JS, or tests).
3. **Step 3 — Compatibility Check:** Verify that the proposed modification adheres to all Strict Project Rules (Section 5) and Docker Architecture (Section 2).
4. **Step 4 — Proposal Summary:** Briefly explain to the user what will be modified and why.
5. **Step 5 — Minimal Implementation:** Implement only the requested change cleanly without introducing unrelated refactors or unwanted bloat.
6. **Step 6 — Preserve Functionality:** Ensure existing working endpoints, UI interactions, and Docker configurations remain unbroken.
7. **Step 7 — Test & Verify:** Run unit/integration tests inside Docker and verify that the prompt-to-image pipeline functions properly.
8. **Step 8 — Update `brain.md`:** If new routes, container settings, configuration keys, or dependencies are introduced, update `brain.md` immediately.
9. **Step 9 — Final Report:** Report all modified files, executed tests, verification results, and any outstanding considerations to the user.

---

## 9. Current Project Status

### 9.1 Implemented & Verified Components
- [x] **`Dockerfile`:** Production Python 3.10 slim container definition with `curl`, `libgl1`, `libglib2.0-0`, and `HEALTHCHECK`.
- [x] **`docker-compose.yml`:** Complete orchestration with persistent volumes (`./static/generated` and `hf_cache`), port forwarding `5000:5000`, network bridge, and health monitoring.
- [x] **`.dockerignore`:** Excludes local caches, secrets, and git metadata from Docker images.
- [x] **`app.py`:** Complete Flask application with `0.0.0.0` binding, `/health`, `/status`, `/generate`, `/download/<filename>`, and structured error handling.
- [x] **`config.py`:** Configuration loader supporting `.env`, fallback defaults, paths, and `sanitize_secrets()` regex redaction.
- [x] **`model_service.py`:** Complete `TextToImageModelService` implementation featuring:
  - Strict prompt validation (empty, whitespace, length).
  - Local PyTorch Diffusers pipeline execution with device detection (`cuda`/`cpu`).
  - Hugging Face Cloud Inference API execution with upstream status handling.
  - Pillow image processing and unique filename disk persistence.
  - Custom `ModelServiceError` with structured diagnostics.
- [x] **`templates/index.html`:** Modern dark AI glassmorphic interface with Docker containerized badge, live status bar, character counter, loading spinner with elapsed timer, and collapsible BCA Viva guide.
- [x] **`static/css/style.css`:** Custom dark AI CSS3 styling with ambient glow effects, responsive grid, and button animations.
- [x] **`static/js/main.js`:** Vanilla JavaScript controller managing input validation, asynchronous fetch calls, real-time stage progression, timer, and error display.
- [x] **`tests/test_app.py`:** Automated test suite implementing 10 test cases covering routing, validation, Docker health probes, and security.
- [x] **`requirements.txt`:** Core production dependencies defined.
- [x] **`.env.example`:** Complete template for environment variables and model selection.
- [x] **`README.md`:** Comprehensive college demonstration manual, Docker quickstart workflow, troubleshooting table, and top 10 BCA Viva Q&A.
- [x] **`brain.md`:** Complete authoritative documentation (v2.0.0).

### 9.2 Pending Components
- **None.** The entire system, frontend, backend, AI engine, tests, documentation, and Docker configuration are complete and ready for execution.

### 9.3 Known Environmental Notes & Constraints
- Docker must be installed and running on the host machine to execute container commands.
- For local Diffusers GPU execution, the host requires an NVIDIA GPU with the NVIDIA Container Toolkit installed. For systems without a GPU, `GENERATION_MODE=hf_api` or `MODEL_ID=segmind/tiny-sd` is recommended for high-speed demonstration.
