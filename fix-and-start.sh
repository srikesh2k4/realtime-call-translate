#!/bin/bash
# ============================================
# 🔧 FIX VAST.AI CUDA + RESTART ALL SERVICES
# ============================================
# Run this script to fix libcublas and restart everything

set -e

echo "🔧 Fixing CUDA library paths..."

# Find CUDA libraries
CUDA_PATHS=$(find /usr -name "libcublas*.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -z "$CUDA_PATHS" ]; then
    CUDA_PATHS="/usr/local/cuda/lib64"
fi

# Export CUDA paths
export LD_LIBRARY_PATH=$CUDA_PATHS:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "   ✓ LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Test CUDA
echo "🔍 Testing CUDA..."
python3 -c "import torch; print(f'   ✓ PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Kill any existing processes
echo "🛑 Stopping existing services..."
pkill -f "uvicorn worker:app" 2>/dev/null || true
pkill -f "./server" 2>/dev/null || true
sleep 2

# Start ML Worker
echo "🚀 Starting ML Worker..."
cd /workspace/realtime-call-translate/ml-python

# Pull latest code
git stash 2>/dev/null || true
git pull origin optimise 2>/dev/null || true

# Start ML worker in background
nohup python -m uvicorn worker:app --host 0.0.0.0 --port 9001 > /tmp/ml_worker.log 2>&1 &
ML_PID=$!
echo "   ✓ ML Worker started (PID: $ML_PID)"
echo "   → Logs: tail -f /tmp/ml_worker.log"

# Wait for ML worker to be ready
echo "⏳ Waiting for ML Worker to load models (this takes ~1-2 min)..."
for i in {1..120}; do
    if curl -s http://localhost:9001/health > /dev/null 2>&1; then
        echo "   ✓ ML Worker is ready!"
        break
    fi
    sleep 2
done

# Start Go Backend
echo "🚀 Starting Go Backend..."
cd /workspace/realtime-call-translate/backend-go

# Build if needed
if [ ! -f "./server" ]; then
    go build -o server main.go
fi

export ML_WORKER_HOST=127.0.0.1
export ML_WORKER_PORT=9001
nohup ./server > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "   ✓ Backend started (PID: $BACKEND_PID)"
echo "   → Logs: tail -f /tmp/backend.log"

sleep 3

# Health check
echo ""
echo "🔍 Health Checks..."
curl -s http://localhost:9001/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'   ML Worker: {d.get(\"status\", \"unknown\")} | Device: {d.get(\"device\", \"unknown\")} | Languages: {\", \".join(d.get(\"supported_languages\", []))}')" 2>/dev/null || echo "   ⚠️ ML Worker not responding yet"
curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'   Backend: {d.get(\"status\", \"unknown\")} | Connections: {d.get(\"connections\", 0)}')" 2>/dev/null || echo "   ⚠️ Backend not responding yet"

echo ""
echo "============================================"
echo "✅ Services Started!"
echo "============================================"
echo ""
echo "To expose via Cloudflare:"
echo "   cloudflared tunnel --url http://localhost:8000"
echo ""
echo "View logs:"
echo "   tail -f /tmp/ml_worker.log   # ML Worker"
echo "   tail -f /tmp/backend.log     # Backend"
echo ""
echo "Process IDs:"
echo "   ML Worker: $ML_PID"
echo "   Backend:   $BACKEND_PID"
echo ""
