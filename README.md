# 🌐 Real-Time Call Translation

**Live two-way translation for phone calls** - English ↔ Hindi with ultra-low latency.

## 🚀 Features

- **Real-time ASR** using faster-whisper (CTranslate2 backend)
- **Smart VAD** with Silero for natural speech boundaries
- **Noise suppression** for clean audio in any environment
- **GPU-accelerated** for lowest latency (NVIDIA CUDA)
- **WebSocket streaming** for real-time communication
- **Translation** via GPT-4.1-mini for accuracy
- **TTS** via OpenAI for natural speech output

## 🎮 Hardware Requirements

### For NVIDIA GPU (Recommended)
- **GPU**: NVIDIA RTX 2060 or better (6GB+ VRAM)
- **Recommended**: RTX 3080/4080/4090 for best performance
- **RAM**: 16GB+
- **Driver**: NVIDIA Driver 525+ with CUDA 12.1

### For CPU Only
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB+
- **Note**: ~3-5x higher latency than GPU

### ☁️ Cloud GPU (vast.ai)
- **Recommended**: RTX 4090 (~$0.40-0.80/hr)
- **Minimum**: RTX 3080 or better
- See [vast.ai Deployment Guide](#-vastai-deployment-guide) below

---

## 📦 Quick Start (Local)

### Prerequisites

1. **Docker & Docker Compose**
2. **NVIDIA Container Toolkit** (for GPU support)
3. **OpenAI API Key**

### Install NVIDIA Container Toolkit (Linux)

```bash
# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate
```

2. **Create environment file**
```bash
echo "OPENAI_API_KEY=your-api-key-here" > ml-python/.env
```

3. **Start with GPU** (NVIDIA)
```bash
docker compose up --build
```

4. **Start with CPU** (No GPU)
```bash
docker compose -f docker-compose.cpu.yml up --build
```

5. **Open the app**
```
http://localhost:5173
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required for translation & TTS |
| `WHISPER_MODEL` | `large-v3` | ASR model (`large-v3`, `distil-large-v3`, `medium`) |
| `WHISPER_COMPUTE` | `float16` | Compute type (`float16`, `int8`, `int8_float16`) |
| `BATCH_SIZE` | `4` | Parallel decoding workers |
| `CUDA_DEVICE` | `0` | GPU device index |

### Model Selection Guide

| Model | VRAM | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `large-v3` | ~3GB | Baseline | Best | Production with RTX 3080+ |
| `distil-large-v3` | ~2GB | 2x faster | Very Good | Balanced performance |
| `medium` | ~1.5GB | 3x faster | Good | Lower VRAM GPUs |

### Compute Type Guide

| Type | Speed | Accuracy | GPU Support |
|------|-------|----------|-------------|
| `float16` | Fast | Best | RTX 20xx+ (Tensor Cores) |
| `int8_float16` | Faster | Very Good | RTX 20xx+ |
| `int8` | Fastest | Good | All CUDA GPUs |

## 📊 Performance Benchmarks

### RTX 4090 (24GB VRAM)
- ASR Latency: ~200ms
- End-to-end: ~500ms
- Concurrent speakers: 8+

### RTX 3080 (10GB VRAM)
- ASR Latency: ~300ms
- End-to-end: ~700ms
- Concurrent speakers: 4

### CPU (Intel i9-13900K)
- ASR Latency: ~1.5s
- End-to-end: ~2.5s
- Concurrent speakers: 1-2

## 🏗️ Architecture

```
┌─────────────┐     WebSocket      ┌─────────────┐      HTTP       ┌─────────────┐
│   Frontend  │ ◄──────────────────► │  Go Backend │ ◄──────────────► │  ML Worker  │
│   (React)   │    Audio + Text     │  (WebSocket)│   ASR/Translate │  (FastAPI)  │
│   :5173     │                     │    :8000    │       /TTS       │    :9001    │
└─────────────┘                     └─────────────┘                 └──────┬──────┘
                                                                          │
                                                                    ┌─────▼─────┐
                                                                    │  NVIDIA   │
                                                                    │    GPU    │
                                                                    │  (CUDA)   │
                                                                    └───────────┘
```

---

## ☁️ vast.ai Deployment Guide

Deploy on **vast.ai** to get RTX 4090 performance at ~$0.40-0.80/hour!

### Why vast.ai?

| Feature | vast.ai | AWS/GCP | Local |
|---------|---------|---------|-------|
| RTX 4090 Cost | $0.40-0.80/hr | $3-5/hr | $2000+ upfront |
| Setup Time | 5 minutes | 30+ minutes | Hours |
| Maintenance | None | Complex | Required |
| Scaling | Instant | Minutes | Buy more hardware |

### Prerequisites

1. **vast.ai Account**: Sign up at [vast.ai](https://vast.ai)
2. **OpenAI API Key**: For translation and TTS
3. **vast.ai CLI** (optional): `pip install vastai`

---

### 🚀 Method 1: Web UI (Easiest)

#### Step 1: Find an RTX 4090 Instance

1. Go to [vast.ai Console](https://cloud.vast.ai/create/)
2. Set filters:
   - **GPU**: RTX 4090
   - **CUDA**: >= 12.0
   - **Disk**: >= 50 GB
   - **Reliability**: >= 95%
3. Sort by **$/hr** (cheapest first)
4. Click **RENT** on your preferred instance

#### Step 2: Configure Instance

- **Image**: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- **Disk Space**: 50 GB
- **Docker Options**: Enable SSH
- **On-start Script**:
```bash
apt-get update && apt-get install -y git curl
```

#### Step 3: Connect via SSH

After instance starts (~1-2 minutes):
```bash
# Copy SSH command from vast.ai console
ssh -p <PORT> root@<HOST>
```

#### Step 4: Deploy the Application

```bash
# Clone repository
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate

# Make script executable
chmod +x deploy-vastai-instance.sh

# Run setup (will prompt for OpenAI API key)
./deploy-vastai-instance.sh
```

#### Step 5: Configure Port Forwarding

1. In vast.ai console, go to your instance
2. Click **Open Ports** / **Port Mapping**
3. Add port **9001** (HTTP)
4. Copy the public URL

#### Step 6: Connect Local Frontend

On your **local machine**:
```bash
# Set environment variable to point to vast.ai
export ML_WORKER_URL=http://<VAST_AI_IP>:9001

# Start local backend + frontend
docker compose -f docker-compose.cpu.yml up backend frontend
```

Or update `backend-go/main.go`:
```go
// Change ML_WORKER_HOST to your vast.ai IP
```

---

### 🚀 Method 2: CLI Script (Automated)

#### Step 1: Install vast.ai CLI

```bash
pip install vastai
```

#### Step 2: Set API Key

```bash
# Get API key from https://cloud.vast.ai/account/
export VASTAI_API_KEY="your-api-key"
vastai set api-key $VASTAI_API_KEY
```

#### Step 3: Run Deployment Script

```bash
# Set OpenAI key
export OPENAI_API_KEY="your-openai-key"

# Run automated deployment
chmod +x deploy-vastai.sh
./deploy-vastai.sh
```

The script will:
1. Search for cheapest RTX 4090 instances
2. Create and start the instance
3. Provide SSH connection details

#### Step 4: Complete Setup on Instance

```bash
# SSH into instance (command provided by script)
ssh -p <PORT> root@<HOST>

# Run instance setup
cd realtime-call-translate
./deploy-vastai-instance.sh
```

---

### � Method 3: Docker Compose on vast.ai

For running the **entire stack** on vast.ai:

#### Step 1: Create Instance with Docker

1. Rent RTX 4090 instance on vast.ai
2. Use image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
3. Enable Docker mode

#### Step 2: Deploy Full Stack

```bash
# SSH into instance
ssh -p <PORT> root@<HOST>

# Install Docker Compose
apt-get update && apt-get install -y docker-compose-plugin

# Clone and deploy
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate

# Create .env file
echo "OPENAI_API_KEY=your-key" > ml-python/.env

# Deploy with vast.ai optimized config
docker compose -f docker-compose.vastai.yml up --build -d
```

#### Step 3: Access the App

Open ports in vast.ai:
- **5173**: Frontend
- **8000**: Backend
- **9001**: ML Worker (optional)

Access: `http://<VAST_AI_IP>:5173`

---

### 📊 vast.ai Performance & Cost

#### RTX 4090 Performance (24GB VRAM)
| Metric | Value |
|--------|-------|
| ASR Latency | ~150-200ms |
| End-to-End | ~400-600ms |
| Concurrent Speakers | 8-10 |
| Model | large-v3 (float16) |

#### Cost Estimates
| Usage | Hourly | Daily | Monthly |
|-------|--------|-------|---------|
| Light (4 hrs/day) | $0.50 | $2 | $60 |
| Medium (8 hrs/day) | $0.50 | $4 | $120 |
| Heavy (24/7) | $0.50 | $12 | $360 |

*Prices vary based on availability. RTX 4090 typically $0.40-0.80/hr*

---

### 🔧 vast.ai Troubleshooting

#### Instance won't start
```bash
# Check instance status
vastai show instances

# Destroy and recreate
vastai destroy instance <ID>
```

#### Port not accessible
1. Check vast.ai port forwarding settings
2. Ensure firewall allows the port
3. Try direct IP mode

#### GPU not detected in container
```bash
# Verify GPU access
nvidia-smi

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### High latency
1. Choose instance closer to your location
2. Check network speed: `speedtest-cli`
3. Use instance with higher inet_down rating

---

## �🐛 General Troubleshooting

### GPU not detected
```bash
# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
```

### Out of memory (OOM)
```bash
# Use smaller model
WHISPER_MODEL=distil-large-v3 docker compose up

# Or use int8 quantization
WHISPER_COMPUTE=int8 docker compose up
```

### High latency
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure `float16` compute type
- Use wired network connection
- Reduce audio buffer size

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CUDA-optimized Whisper
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection
- [OpenAI](https://openai.com) - Translation & TTS APIs
- [vast.ai](https://vast.ai) - Affordable cloud GPUs
