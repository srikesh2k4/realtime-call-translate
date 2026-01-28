#!/bin/bash
# ============================================
# 🚀 VAST.AI JUPYTER-BASED DEPLOYMENT (NO DOCKER)
# ============================================
# Run this INSIDE the vast.ai Jupyter terminal
# Compatible with Cloudflare Tunnel
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================"
echo "🚀 Real-Time Translation - vast.ai Setup"
echo "   (Jupyter-based, No Docker)"
echo "============================================"
echo -e "${NC}"

# ============================================
# STEP 1: System Dependencies
# ============================================
echo -e "${BLUE}📦 Installing system dependencies...${NC}"

apt-get update -qq
apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    wget \
    git \
    htop \
    nvtop \
    tmux \
    golang-go \
    build-essential \
    cmake

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 2>/dev/null || true
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true

echo -e "${GREEN}✓ System dependencies installed${NC}"

# ============================================
# STEP 2: GPU Check
# ============================================
echo ""
echo -e "${BLUE}🎮 Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo -e "${GREEN}✓ GPU detected with ${GPU_MEM}MB VRAM${NC}"

if [ "$GPU_MEM" -lt 16000 ]; then
    echo -e "${YELLOW}⚠️  Warning: GPU has less than 16GB VRAM. NLLB-3.3B may not fit!${NC}"
    echo -e "${YELLOW}   Consider using NLLB-1.3B instead.${NC}"
fi

# ============================================
# STEP 3: Python Environment
# ============================================
echo ""
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"

pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -q

# Install core ML dependencies
pip install -q \
    ctranslate2>=4.0.0 \
    faster-whisper>=1.0.0 \
    silero-vad>=5.1 \
    transformers>=4.37.0 \
    sentencepiece>=0.1.99 \
    protobuf>=4.25.0 \
    accelerate>=0.26.0 \
    optimum>=1.17.0 \
    noisereduce>=3.0.0 \
    soundfile>=0.12.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    librosa>=0.10.0 \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    openai>=1.12.0 \
    python-dotenv>=1.0.0

echo -e "${GREEN}✓ Python packages installed${NC}"

# ============================================
# STEP 4: Clone/Update Repository
# ============================================
echo ""
echo -e "${BLUE}📁 Setting up application...${NC}"

cd /workspace 2>/dev/null || cd /root || cd ~

# Clone if not exists, else pull latest
if [ ! -d "realtime-call-translate" ]; then
    git clone https://github.com/srikesh2k4/realtime-call-translate.git
    cd realtime-call-translate
else
    cd realtime-call-translate
    git fetch origin
    git checkout optimise 2>/dev/null || git checkout main
    git pull
fi

echo -e "${GREEN}✓ Repository ready${NC}"

# ============================================
# STEP 5: Configure Environment
# ============================================
echo ""
echo -e "${BLUE}🔧 Configuring environment...${NC}"

# Create .env file in ml-python
cat > ml-python/.env << 'ENVEOF'
# OpenAI API Key (OPTIONAL - only for TTS, translation is local)
OPENAI_API_KEY=

# ASR Settings
WHISPER_MODEL=large-v3-turbo
WHISPER_COMPUTE=float16

# Translation Settings (LOCAL NLLB)
NLLB_MODEL=facebook/nllb-200-3.3B
NLLB_COMPUTE=bfloat16
USE_BETTERTRANSFORMER=true
MAX_NEW_TOKENS=512

# Processing
BATCH_SIZE=8
CUDA_DEVICE=0
ENVEOF

# Prompt for OpenAI key (optional)
echo -e "${YELLOW}Enter OpenAI API key (press Enter to skip - only needed for TTS):${NC}"
read -r OPENAI_KEY
if [ -n "$OPENAI_KEY" ]; then
    sed -i "s/OPENAI_API_KEY=/OPENAI_API_KEY=$OPENAI_KEY/" ml-python/.env
fi

echo -e "${GREEN}✓ Environment configured${NC}"

# ============================================
# STEP 6: Download Models (This takes time!)
# ============================================
echo ""
echo -e "${BLUE}📥 Downloading ML models (this takes 5-10 minutes)...${NC}"

cd ml-python

python3 << 'PYEOF'
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1/3 Downloading Whisper large-v3-turbo...")
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
del model
print("    ✓ Whisper ready")

print("2/3 Downloading Silero VAD...")
from silero_vad import load_silero_vad
load_silero_vad()
print("    ✓ VAD ready")

print("3/3 Downloading NLLB-200-3.3B (this is ~7GB)...")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
del model, tokenizer
print("    ✓ NLLB ready")

import torch
torch.cuda.empty_cache()
print("\n✓ All models downloaded successfully!")
PYEOF

cd ..
echo -e "${GREEN}✓ Models downloaded${NC}"

# ============================================
# STEP 7: Build Go Backend
# ============================================
echo ""
echo -e "${BLUE}🔨 Building Go backend...${NC}"

cd backend-go
go mod download
go build -ldflags="-w -s" -o server main.go
cd ..

echo -e "${GREEN}✓ Go backend built${NC}"

# ============================================
# STEP 8: Install Cloudflare Tunnel
# ============================================
echo ""
echo -e "${BLUE}☁️  Installing Cloudflare Tunnel (cloudflared)...${NC}"

# Download cloudflared
curl -Lo /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared

echo -e "${GREEN}✓ Cloudflared installed${NC}"

# ============================================
# STEP 9: Create Start Scripts
# ============================================
echo ""
echo -e "${BLUE}📝 Creating start scripts...${NC}"

# ML Worker start script
cat > start-ml-worker.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")/ml-python"
source .env 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
python3 -m uvicorn worker:app --host 0.0.0.0 --port 9001
SCRIPT
chmod +x start-ml-worker.sh

# Go Backend start script
cat > start-backend.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")/backend-go"
export ML_WORKER_HOST=127.0.0.1
export ML_WORKER_PORT=9001
./server
SCRIPT
chmod +x start-backend.sh

# Cloudflare tunnel script
cat > start-cloudflare.sh << 'SCRIPT'
#!/bin/bash
# Start Cloudflare Quick Tunnel (no account needed)
echo "🌐 Starting Cloudflare Quick Tunnel..."
echo "   This will give you a public URL like: https://xxx.trycloudflare.com"
echo ""
cloudflared tunnel --url http://localhost:8000
SCRIPT
chmod +x start-cloudflare.sh

# Combined start script
cat > start-all.sh << 'SCRIPT'
#!/bin/bash
echo "🚀 Starting all services..."

# Start ML Worker in background
echo "Starting ML Worker on port 9001..."
./start-ml-worker.sh &
ML_PID=$!
sleep 30  # Wait for models to load

# Check if ML Worker is up
if curl -s http://localhost:9001/health > /dev/null; then
    echo "✓ ML Worker is running"
else
    echo "⚠️ ML Worker may still be loading models..."
fi

# Start Go Backend in background
echo "Starting Go Backend on port 8000..."
./start-backend.sh &
BACKEND_PID=$!
sleep 5

# Check if Backend is up
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is running"
fi

echo ""
echo "============================================"
echo "🎉 Services started!"
echo "============================================"
echo ""
echo "Local endpoints:"
echo "  ML Worker:  http://localhost:9001/health"
echo "  Backend:    http://localhost:8000/health"
echo ""
echo "To expose via Cloudflare, run in another terminal:"
echo "  ./start-cloudflare.sh"
echo ""
echo "Process IDs:"
echo "  ML Worker: $ML_PID"
echo "  Backend:   $BACKEND_PID"
echo ""
echo "To stop all: kill $ML_PID $BACKEND_PID"

# Keep script running
wait
SCRIPT
chmod +x start-all.sh

# Stop script
cat > stop-all.sh << 'SCRIPT'
#!/bin/bash
echo "Stopping all services..."
pkill -f "uvicorn worker:app" || true
pkill -f "./server" || true
pkill -f "cloudflared" || true
echo "✓ All services stopped"
SCRIPT
chmod +x stop-all.sh

echo -e "${GREEN}✓ Start scripts created${NC}"

# ============================================
# STEP 10: Print Instructions
# ============================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
echo "  ./start-all.sh              # Start ML Worker + Backend"
echo "  ./start-cloudflare.sh       # Expose via Cloudflare (run in separate terminal)"
echo ""
echo -e "${BLUE}Individual Services:${NC}"
echo "  ./start-ml-worker.sh        # ML Worker only (port 9001)"
echo "  ./start-backend.sh          # Go Backend only (port 8000)"
echo ""
echo -e "${BLUE}Stop Everything:${NC}"
echo "  ./stop-all.sh"
echo ""
echo -e "${BLUE}Health Checks:${NC}"
echo "  curl http://localhost:9001/health   # ML Worker"
echo "  curl http://localhost:8000/health   # Backend"
echo ""
echo -e "${BLUE}Monitor GPU:${NC}"
echo "  nvtop"
echo ""
echo -e "${YELLOW}⚠️  For Cloudflare Tunnel:${NC}"
echo "  1. Run ./start-cloudflare.sh in a separate terminal"
echo "  2. Copy the https://xxx.trycloudflare.com URL"
echo "  3. Use that URL in your frontend config"
echo ""
