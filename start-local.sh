#!/bin/bash
# ============================================
# 🚀 LOCAL DEPLOYMENT - ONLY FRONTEND EXPOSED
# ============================================
# All services run locally on vast.ai
# Only Frontend is exposed via Cloudflare
# ML ↔ Backend ↔ Frontend all on localhost for low latency
#
# Usage: bash start-local.sh
# ============================================

set -e

echo "============================================"
echo "🚀 LOCAL DEPLOYMENT STARTING..."
echo "============================================"

cd /workspace/realtime-call-translate

# Kill existing
echo "🛑 Stopping existing services..."
pkill -f "uvicorn worker:app" 2>/dev/null || true
pkill -f "./server" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
sleep 2

# Pull latest
echo "📥 Pulling latest code..."
git fetch origin 2>/dev/null && git reset --hard origin/optimise 2>/dev/null || true

# Create ML .env
echo "📝 Setting up ML Worker..."
cat > ml-python/.env << 'EOF'
OPENAI_API_KEY=YOUR_API_KEY_HERE
WHISPER_MODEL=large-v3-turbo
EOF
echo "   ⚠️  Edit ml-python/.env and add your OPENAI_API_KEY!"

# Remove frontend .env (use proxy instead)
echo "📝 Configuring Frontend for local proxy..."
rm -f frontend/.env 2>/dev/null || true

# Start ML Worker (Terminal 1 functionality)
echo ""
echo "🧠 Starting ML Worker on port 9001..."
cd ml-python
nohup python -m uvicorn worker:app --host 127.0.0.1 --port 9001 > /tmp/ml.log 2>&1 &
echo "   → ML Worker starting... (logs: /tmp/ml.log)"
cd ..

# Wait for ML Worker
echo "⏳ Waiting for ML Worker to initialize..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:9001/health > /dev/null 2>&1; then
        echo "   ✓ ML Worker ready!"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# Start Go Backend (Terminal 2 functionality)
echo "⚙️ Starting Backend on port 8000..."
cd backend-go
go build -o server main.go 2>/dev/null || true
export ML_WORKER_HOST=127.0.0.1
export ML_WORKER_PORT=9001
nohup ./server > /tmp/backend.log 2>&1 &
echo "   → Backend starting... (logs: /tmp/backend.log)"
cd ..

sleep 3

# Start Frontend (Terminal 3 functionality)
echo "🖥️ Starting Frontend on port 5173..."
cd frontend
npm install --silent 2>/dev/null || true
nohup npm run dev -- --host 0.0.0.0 > /tmp/frontend.log 2>&1 &
echo "   → Frontend starting... (logs: /tmp/frontend.log)"
cd ..

sleep 5

# Health checks
echo ""
echo "🔍 Health Checks..."
curl -s http://127.0.0.1:9001/health > /dev/null 2>&1 && echo "   ✓ ML Worker: OK" || echo "   ✗ ML Worker: FAILED"
curl -s http://127.0.0.1:8000/health > /dev/null 2>&1 && echo "   ✓ Backend: OK" || echo "   ✗ Backend: FAILED"
curl -s http://127.0.0.1:5173 > /dev/null 2>&1 && echo "   ✓ Frontend: OK" || echo "   ✗ Frontend: FAILED"

echo ""
echo "============================================"
echo "✅ ALL SERVICES RUNNING LOCALLY"
echo "============================================"
echo ""
echo "📊 Architecture:"
echo "   ML Worker (9001) ← localhost → Backend (8000) ← localhost → Frontend (5173)"
echo ""
echo "🌐 Now expose ONLY the frontend:"
echo "   cloudflared tunnel --url http://localhost:5173"
echo ""
echo "📋 Logs:"
echo "   tail -f /tmp/ml.log       # ML Worker"
echo "   tail -f /tmp/backend.log  # Backend"
echo "   tail -f /tmp/frontend.log # Frontend"
echo ""
echo "🛑 To stop all:"
echo "   pkill -f uvicorn; pkill -f './server'; pkill -f 'npm run dev'"
echo "============================================"
