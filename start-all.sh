#!/bin/bash
# ============================================
# 🚀 ONE-COMMAND STARTUP FOR VAST.AI
# ============================================
# Usage: bash start-all.sh

set -e

echo "============================================"
echo "🚀 LIVE CALL TRANSLATION - STARTING ALL"
echo "============================================"

# Kill any existing processes
echo "🛑 Stopping existing services..."
pkill -f "uvicorn worker:app" 2>/dev/null || true
pkill -f "./server" 2>/dev/null || true
sleep 2

# Set working directory
cd /workspace/realtime-call-translate

# Pull latest code
echo "📥 Pulling latest code..."
git fetch origin 2>/dev/null || true
git reset --hard origin/optimise 2>/dev/null || true

# Create .env for ML worker
echo "📝 Creating .env..."
cat > ml-python/.env << 'EOF'
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
WHISPER_MODEL=large-v3-turbo
EOF

# Start ML Worker in background
echo "🧠 Starting ML Worker..."
cd ml-python
nohup python -m uvicorn worker:app --host 0.0.0.0 --port 9001 > /tmp/ml_worker.log 2>&1 &
ML_PID=$!
echo "   → ML Worker PID: $ML_PID"
cd ..

# Wait for ML worker to initialize
echo "⏳ Waiting for ML Worker (30 seconds)..."
sleep 30

# Build and start Go backend
echo "⚙️ Building Go backend..."
cd backend-go
go build -o server main.go 2>/dev/null || true
export ML_WORKER_HOST=127.0.0.1
export ML_WORKER_PORT=9001
nohup ./server > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "   → Backend PID: $BACKEND_PID"
cd ..

sleep 3

echo ""
echo "============================================"
echo "✅ SERVICES STARTED!"
echo "============================================"
echo ""
echo "ML Worker:  http://localhost:9001 (PID: $ML_PID)"
echo "Backend:    http://localhost:8000 (PID: $BACKEND_PID)"
echo ""
echo "📋 Logs:"
echo "   tail -f /tmp/ml_worker.log"
echo "   tail -f /tmp/backend.log"
echo ""
echo "🌐 Now run Cloudflare tunnel:"
echo "   cloudflared tunnel --url http://localhost:8000"
echo ""
echo "Then update your local frontend/.env with the tunnel URL!"
echo "============================================"
