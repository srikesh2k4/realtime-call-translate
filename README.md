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
- **Recommended**: RTX 3080/4080 for best performance
- **RAM**: 16GB+
- **Driver**: NVIDIA Driver 525+ with CUDA 12.1

### For CPU Only
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB+
- **Note**: ~3-5x higher latency than GPU

## 📦 Quick Start

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

## 🐛 Troubleshooting

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
