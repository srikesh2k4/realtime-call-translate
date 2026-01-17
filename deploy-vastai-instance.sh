#!/bin/bash
# ============================================
# 🚀 VAST.AI INSTANCE SETUP SCRIPT
# ============================================
# Run this INSIDE the vast.ai instance after SSH
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
echo "🚀 Setting up ML Worker on vast.ai"
echo "============================================"
echo -e "${NC}"

# ============================================
# STEP 1: System Setup
# ============================================
echo -e "${BLUE}📦 Installing system dependencies...${NC}"

apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    htop \
    nvtop

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 || true
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 || true

echo -e "${GREEN}✓ System dependencies installed${NC}"

# ============================================
# STEP 2: GPU Check
# ============================================
echo ""
echo -e "${BLUE}🎮 Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo -e "${GREEN}✓ GPU detected${NC}"

# ============================================
# STEP 3: Python Environment
# ============================================
echo ""
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"

pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install CTranslate2 and faster-whisper
pip install ctranslate2>=4.0.0 faster-whisper>=1.0.0

echo -e "${GREEN}✓ PyTorch installed${NC}"

# ============================================
# STEP 4: Install Application
# ============================================
echo ""
echo -e "${BLUE}📁 Installing application...${NC}"

cd /root

# Clone if not already cloned
if [ ! -d "realtime-call-translate" ]; then
    git clone https://github.com/srikesh2k4/realtime-call-translate.git
fi

cd realtime-call-translate/ml-python

# Install requirements
pip install -r requirements.txt

echo -e "${GREEN}✓ Application installed${NC}"

# ============================================
# STEP 5: Configure Environment
# ============================================
echo ""
echo -e "${BLUE}🔧 Configuring environment...${NC}"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Enter your OpenAI API key:${NC}"
    read -s OPENAI_API_KEY
fi

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
WHISPER_MODEL=large-v3
WHISPER_COMPUTE=float16
BATCH_SIZE=8
CUDA_DEVICE=0
EOF

echo -e "${GREEN}✓ Environment configured${NC}"

# ============================================
# STEP 6: Download Models
# ============================================
echo ""
echo -e "${BLUE}📥 Pre-downloading models (this may take a few minutes)...${NC}"

python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper large-v3...')
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('✓ Model downloaded and loaded on GPU')
"

python3 -c "
from silero_vad import load_silero_vad
print('Downloading Silero VAD...')
load_silero_vad()
print('✓ VAD model downloaded')
"

echo -e "${GREEN}✓ Models downloaded${NC}"

# ============================================
# STEP 7: Start the Service
# ============================================
echo ""
echo -e "${BLUE}🚀 Starting ML Worker...${NC}"

# Create systemd service for auto-restart
cat > /etc/systemd/system/ml-worker.service << EOF
[Unit]
Description=Real-Time Translation ML Worker
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/realtime-call-translate/ml-python
Environment="OPENAI_API_KEY=$OPENAI_API_KEY"
Environment="WHISPER_MODEL=large-v3"
Environment="WHISPER_COMPUTE=float16"
Environment="BATCH_SIZE=8"
ExecStart=/usr/bin/python3 -m uvicorn worker:app --host 0.0.0.0 --port 9001 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ml-worker
systemctl start ml-worker

echo -e "${GREEN}✓ ML Worker service started${NC}"

# ============================================
# STEP 8: Get Connection Info
# ============================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)

echo -e "ML Worker URL: ${YELLOW}http://$PUBLIC_IP:9001${NC}"
echo -e "Health Check:  ${YELLOW}http://$PUBLIC_IP:9001/health${NC}"
echo ""
echo -e "${BLUE}Test the API:${NC}"
echo "curl http://$PUBLIC_IP:9001/health"
echo ""
echo -e "${BLUE}View logs:${NC}"
echo "journalctl -u ml-worker -f"
echo ""
echo -e "${BLUE}Monitor GPU:${NC}"
echo "nvtop"
echo ""
echo -e "${YELLOW}⚠️  Make sure to open port 9001 in vast.ai port forwarding!${NC}"
echo ""
echo -e "${BLUE}To connect your local frontend/backend:${NC}"
echo "export ML_WORKER_HOST=$PUBLIC_IP"
echo "export ML_WORKER_PORT=9001"
echo ""
