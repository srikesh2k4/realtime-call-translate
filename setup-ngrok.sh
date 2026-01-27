#!/bin/bash
# ============================================
# 🚀 NGROK SETUP SCRIPT
# ============================================
# Quick public access using ngrok (free tier)
# ============================================

set -e

echo "🌐 ngrok Setup for Live Translation"
echo "===================================="
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "📥 Installing ngrok..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install ngrok/ngrok/ngrok
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
        sudo apt update && sudo apt install ngrok
    else
        echo "❌ Unsupported OS. Please install ngrok manually:"
        echo "   https://ngrok.com/download"
        exit 1
    fi
fi

echo "✅ ngrok is installed"
echo ""

# Check for auth token
if ! ngrok config check &> /dev/null; then
    echo "🔐 Please enter your ngrok auth token:"
    echo "   Get it from: https://dashboard.ngrok.com/get-started/your-authtoken"
    read -p "Auth token: " NGROK_TOKEN
    ngrok config add-authtoken "$NGROK_TOKEN"
    echo "✅ Auth token saved"
fi

echo ""
echo "🚀 Starting services..."
docker compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

echo ""
echo "🌐 Starting ngrok tunnels..."
echo ""
echo "📱 Frontend URL will appear below..."
echo "🔌 WebSocket URL will appear below..."
echo ""

# Start ngrok for frontend (port 5173)
ngrok http 5173 --log=stdout > ngrok-frontend.log 2>&1 &
NGROK_FRONTEND_PID=$!

# Start ngrok for backend WebSocket (port 8000)
ngrok http 8000 --log=stdout > ngrok-backend.log 2>&1 &
NGROK_BACKEND_PID=$!

# Wait for ngrok to start
sleep 5

# Get public URLs
FRONTEND_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
BACKEND_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4 | sed 's/https/wss/g')

echo "✅ ngrok tunnels started!"
echo ""
echo "================================================"
echo "🌐 PUBLIC URLs:"
echo "================================================"
echo "📱 Frontend:  $FRONTEND_URL"
echo "🔌 WebSocket: $BACKEND_URL/ws"
echo "================================================"
echo ""
echo "💡 Open the frontend URL in your browser"
echo "⚠️  Note: Free ngrok resets URLs on restart"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping services...'; docker compose down; kill $NGROK_FRONTEND_PID $NGROK_BACKEND_PID 2>/dev/null; echo '✅ Stopped'; exit 0" INT

# Keep script running
tail -f /dev/null
