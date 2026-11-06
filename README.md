# AI Text-to-Image Generator (Dockerized)

🎓 **BCA Final Year Major Project** | 🐳 **Production Docker Containerized** | 🧠 **Real Machine Learning Pipeline**

A complete, prompt-driven deep learning web application that converts user natural language descriptions into high-resolution visual imagery using real **Latent Diffusion Models**, PyTorch, and Hugging Face Diffusers. 

> **Strict Project Principle:** 100% real neural network inference. Zero mock data, zero placeholder images, zero hardcoded prompts, and zero silent substitutions.

---

## 🌟 Key Highlights

- 🐳 **Mandatory Docker Containerization:** Zero host Python dependencies required. The complete application, dependencies, PyTorch runtime, and Flask server are isolated inside an optimized Docker container.
- 🧠 **Real Deep Learning Pipeline:** Powered by real pre-trained Latent Diffusion Models (`stabilityai/stable-diffusion-xl-base-1.0`, `runwayml/stable-diffusion-v1-5`, or `segmind/tiny-sd`).
- ⚡ **Dual Execution Engines:**
  - **Local Diffusers Mode (`diffusers`):** Executes PyTorch diffusion loops locally with CUDA GPU acceleration or multi-core CPU.
  - **Cloud Inference Mode (`hf_api`):** Uses Hugging Face Inference API for high-speed cloud GPU synthesis on low-spec hardware.
  - **Auto Mode (`auto`):** Automatically selects the best available engine.
- 💾 **Docker Volume Persistence:** Generated images are saved into a host-mounted Docker volume (`./static/generated`) so outputs persist across container restarts. Model weights are cached in `hf_cache`.
- 🩺 **Automated Healthchecks:** Features an internal `/health` endpoint monitored every 30 seconds by the Docker daemon.
- 🔒 **Sanitized Security:** All Hugging Face tokens and API keys are automatically redacted from error traces and logs using regular expression scrubbers.

---

## 📁 Project Structure

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
├── README.md                    # This comprehensive documentation and demonstration manual
├── templates/
│   └── index.html               # Semantic HTML5 UI with dark glassmorphic design
├── static/
│   ├── css/
│   │   └── style.css            # Custom CSS3 dark AI aesthetic with glowing elements
│   ├── js/
│   │   └── main.js              # Vanilla JS frontend controller (AJAX, timer, DOM updates)
│   └── generated/               # Docker-persisted directory storing real AI output PNGs
└── tests/
    └── test_app.py              # Automated unit and integration test suite
```

---

## 🐳 Docker Installation & Setup

All execution, development, and testing for this project runs through Docker. **You do not need to install Python or PyTorch on your host computer.**

### 1. Prerequisites (Install Docker)
- **Windows / macOS:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
  - Ensure Docker Desktop is running (green whale icon in system tray).
  - On Windows, ensure WSL 2 backend is enabled in Docker Desktop settings.
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo systemctl enable --now docker
  ```

Verify your installation:
```bash
docker --version
docker compose version
```

---

## 🚀 Quickstart Guide (Docker Workflow)

### Step 1: Clone / Navigate to Project
```bash
cd ai_text_to_image_generator
```

### Step 2: Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

Open `.env` in any text editor and configure your settings:
```env
# Generation mode: 'auto', 'hf_api', or 'diffusers'
GENERATION_MODE=auto

# (Recommended for fast cloud GPU synthesis) Your free Hugging Face User Token:
# Get one in 30 seconds at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_actual_token_here

# Selected model ID
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
```

### Step 3: Build the Docker Image
Build the container image with all Python dependencies:
```bash
docker compose build
```

### Step 4: Start the Application Container
Launch the container in the background:
```bash
docker compose up -d
```

### Step 5: View Live Application Logs
Monitor the real-time server output, model loading status, and generation traces:
```bash
docker compose logs -f
```

### Step 6: Open the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

### Step 7: Stop the Container
When finished, stop the container safely:
```bash
docker compose down
```

---

## 🧪 Running Automated Tests in Docker

Run the full suite of unit, validation, security, and integration tests directly inside the Docker container:

```bash
docker compose exec ai-generator python -m unittest discover tests -v
```

This verifies:
- Homepage route (`GET /`)
- Container health route (`GET /health`)
- System status route (`GET /status`)
- Rejection of empty and whitespace-only prompts (`HTTP 400`)
- Rejection of prompts exceeding 500 characters (`HTTP 400`)
- Secret and API token sanitization
- Path traversal protection on downloads (`HTTP 404`)

---

## ⚡ Enabling NVIDIA GPU Acceleration (Optional)

If your host machine has an NVIDIA GPU (e.g. RTX 3060, RTX 4070) and you want to run the local PyTorch Diffusers pipeline inside Docker with CUDA acceleration:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Open `docker-compose.yml` and uncomment the GPU reservation section:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
3. Restart the container:
   ```bash
   docker compose down
   docker compose up -d
   ```

---

## 🛠️ Docker Troubleshooting Guide

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `Cannot connect to the Docker daemon` | Docker Desktop is closed | Launch Docker Desktop and wait for it to report "Engine running". |
| `port is already allocated: 5000` | Another process is using port 5000 | Change `FLASK_PORT=5001` in `.env` and `5001:5000` in `docker-compose.yml`. |
| `MissingAuthenticationError` | `HF_TOKEN` is blank in `.env` when using `hf_api` mode | Get a free Hugging Face token at `huggingface.co/settings/tokens` and add it to `.env`. |
| `ModelLoadingUpstreamError (503)` | Cloud model is cold-starting | Hugging Face free tier wakes models in 15–30 seconds. Wait 20 seconds and click Generate again. |
| Container status `unhealthy` | Server failed to boot | Run `docker compose logs` to inspect the exact Python traceback. |
| Model download takes too long | Large SDXL weights (6GB) on slow internet | Switch `.env` to `MODEL_ID=segmind/tiny-sd` (500MB) or use `GENERATION_MODE=hf_api`. |

---

## 🎓 BCA Demonstration & Viva Voce Guide

### 1. How Text-to-Image Diffusion Models Work
```
┌─────────────────┐       Tokenization       ┌────────────────────────┐
│  User Text      │  ─────────────────────►  │  CLIP / OpenCLIP       │
│  "A snow leopard│                          │  Text Encoder (768-dim)│
└─────────────────┘                          └───────────┬────────────┘
                                                         │ Semantic Conditioning
                                                         ▼
┌─────────────────┐    Forward Diffusion     ┌────────────────────────┐
│ Latent Gaussian │  ─────────────────────►  │ U-Net / DiT Core       │
│ Noise Tensor    │                          │ (30 Denoising Steps)   │
└─────────────────┘                          └───────────┬────────────┘
                                                         │ Denoised Latent Code
                                                         ▼
                                             ┌────────────────────────┐
                                             │ VAE Decoder            │
                                             │ (Latent -> RGB Pixels) │
                                             └───────────┬────────────┘
                                                         │
                                                         ▼
                                             ┌────────────────────────┐
                                             │ Generated PNG Image    │
                                             │ (Pillow File Save)     │
                                             └────────────────────────┘
```

### 2. Top 10 BCA Viva Questions & Answers

#### Q1: What is a Latent Diffusion Model (LDM)?
**Answer:** An LDM is a generative deep learning architecture that performs the diffusion (noise generation and removal) process within a compressed, lower-dimensional latent space constructed by a Variational Autoencoder (VAE), rather than on high-dimensional raw pixel space. This reduces computational complexity by a factor of 8x to 16x.

#### Q2: What is the role of the CLIP model in this project?
**Answer:** CLIP (Contrastive Language-Image Pre-Training) acts as the **Text Encoder**. It transforms raw English prompt strings into continuous high-dimensional semantic vector embeddings. These embeddings are fed into the U-Net through cross-attention mechanisms to condition and guide image synthesis.

#### Q3: What is Classifier-Free Guidance (CFG / `guidance_scale`)?
**Answer:** CFG is an inference-time technique that balances how strictly the generated image adheres to the text prompt versus how creative/diverse the generation is. A standard guidance scale of `7.5` amplifies the difference between prompt-conditioned predictions and unconditional predictions.

#### Q4: What is the Variational Autoencoder (VAE) used for?
**Answer:** The VAE consists of an **Encoder** and a **Decoder**. During training, the encoder compresses images into latent space. During text-to-image inference, the **Decoder** takes the final denoised latent tensor and translates it back into visual RGB pixel values.

#### Q5: Why is Docker used instead of running Python locally?
**Answer:** Machine learning projects rely on complex, OS-specific binary packages (PyTorch, CUDA libraries, libgl). Docker encapsulates the exact Python version, system libraries, and dependencies into an immutable image, ensuring zero configuration discrepancies across different computers.

#### Q6: How do we prevent generated images from being lost when the container stops?
**Answer:** We configure a Docker bind mount volume in `docker-compose.yml`: `./static/generated:/app/static/generated`. Any file written inside `/app/static/generated` is instantly written to the host's hard drive and persists indefinitely.

#### Q7: How does this project prevent mock data from being displayed?
**Answer:** The backend has zero predefined image mappings. The `model_service.py` receives the exact user prompt string, validates it, and passes it directly to either the PyTorch `AutoPipelineForText2Image` or the Hugging Face `InferenceClient`. If the operation fails, an actual exception is caught and displayed.

#### Q8: What does the `/health` endpoint do?
**Answer:** It provides a lightweight HTTP probe for container orchestration. Docker queries `http://localhost:5000/health` every 30 seconds. If the application crashes or freezes, Docker marks the container as `unhealthy`.

#### Q9: How are sensitive API keys kept secure?
**Answer:** API keys are never hardcoded in source files or frontend scripts. They are passed via `.env` files (which are excluded from Docker images via `.dockerignore`). Furthermore, the `config.py` module runs all error strings through `sanitize_secrets()` to redact tokens before logging or sending responses.

#### Q10: What is the difference between PyTorch Diffusers and the Hugging Face Inference API?
**Answer:** PyTorch Diffusers runs the neural network locally on the host's CPU/GPU, requiring model weights to be downloaded onto the machine. The Hugging Face Inference API sends the prompt over HTTPS to Hugging Face's cloud GPU clusters, which perform the neural computation and stream back the image bytes.

---

## 📜 License & Credits

- Built for BCA Major Project Demonstration (2026).
- Powered by [PyTorch](https://pytorch.org/), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), and [Flask](https://palletsprojects.com/p/flask/).
