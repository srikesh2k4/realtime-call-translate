# 🚀 vast.ai Quick Deploy Reference

## One-Line Deploy (After SSH into vast.ai)

```bash
git clone https://github.com/srikesh2k4/realtime-call-translate.git && cd realtime-call-translate && chmod +x deploy-vastai-instance.sh && ./deploy-vastai-instance.sh
```

## Manual Quick Setup

```bash
# 1. Install dependencies
apt-get update && apt-get install -y python3.11 python3-pip ffmpeg libsndfile1 git

# 2. Install PyTorch + ML libs
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper>=1.0.0 silero-vad>=5.0.0 fastapi uvicorn openai python-dotenv noisereduce soundfile numpy scipy librosa

# 3. Clone and setup
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate/ml-python

# 4. Set API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 5. Start server
python3 -m uvicorn worker:app --host 0.0.0.0 --port 9001
```

## Search Criteria for vast.ai

```
GPU: RTX 4090
CUDA: >= 12.0
Disk: >= 50 GB
Reliability: >= 95%
Internet Down: >= 200 Mbps
```

## Required Ports

| Port | Service |
|------|---------|
| 9001 | ML Worker API |
| 8000 | Go Backend (optional) |
| 5173 | Frontend (optional) |
| 22 | SSH |

## Test Endpoints

```bash
# Health check
curl http://localhost:9001/health

# Test ASR (with audio file)
curl -X POST http://localhost:9001/asr \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "...", "language": "en"}'
```

## Monitor

```bash
# GPU usage
nvidia-smi -l 1
# or
nvtop

# Service logs
journalctl -u ml-worker -f

# Memory
htop
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...      # Required
WHISPER_MODEL=large-v3     # large-v3, distil-large-v3, medium
WHISPER_COMPUTE=float16    # float16, int8_float16, int8
BATCH_SIZE=8               # Parallel workers
CUDA_DEVICE=0              # GPU index
```
