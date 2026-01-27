#!/bin/bash
# ============================================
# 🚀 NGROK QUICK START SCRIPT
# ============================================
# Expose your local translation service via ngrok
# Perfect for quick demos and testing
# ============================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================"
echo "🚀 Starting ngrok tunnels for Live Translation"
echo "============================================"
echo -e "${NC}"

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${YELLOW}ngrok not found. Installing...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install ngrok/ngrok/ngrok
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
            sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
            echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
            sudo tee /etc/apt/sources.list.d/ngrok.list && \
            sudo apt update && sudo apt install ngrok
    else
        echo -e "${YELLOW}Please install ngrok manually from: https://ngrok.com/download${NC}"
        exit 1
    fi
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo -e "${YELLOW}"
    echo "============================================"
    echo "⚠️  ngrok Authentication Required"
    echo "============================================"
    echo "1. Sign up at: https://dashboard.ngrok.com/signup"
    echo "2. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "3. Run: ngrok config add-authtoken <YOUR_TOKEN>"
    echo "============================================"
    echo -e "${NC}"
    exit 1
fi

# Start services if not running
if ! docker compose ps | grep -q "Up"; then
    echo -e "${GREEN}Starting Docker services...${NC}"
    docker compose up -d
    echo "Waiting for services to be healthy..."
    sleep 15
fi

# Kill any existing ngrok processes
pkill ngrok 2>/dev/null || true
sleep 2

echo -e "${GREEN}Starting ngrok tunnels...${NC}"

# Start ngrok with multiple tunnels
ngrok start --all --config ngrok.yml &

# Wait for ngrok to start
sleep 5

# Get tunnel URLs
echo -e "${GREEN}"
echo "============================================"
echo "✅ Tunnels Active!"
echo "============================================"
curl -s http://localhost:4040/api/tunnels | \
    python3 -c "import sys, json; data=json.load(sys.stdin); [print(f\"{t['name']:12} -> {t['public_url']}\") for t in data['tunnels']]" 2>/dev/null || \
    echo "Visit http://localhost:4040 to see your tunnel URLs"
echo "============================================"
echo -e "${NC}"

echo -e "${BLUE}📱 Share the frontend URL with users${NC}"
echo -e "${BLUE}🔧 ngrok dashboard: http://localhost:4040${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop tunnels${NC}"

# Keep script running
wait
