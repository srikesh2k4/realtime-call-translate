#!/bin/bash
# ============================================
# 🚀 CLOUDFLARE TUNNEL SETUP SCRIPT
# ============================================
# Sets up and runs Cloudflare Tunnel for public access
# ============================================

set -e

echo "🌐 Cloudflare Tunnel Setup for Live Translation"
echo "================================================"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing cloudflared..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install cloudflare/cloudflare/cloudflared
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared-linux-amd64.deb
        rm cloudflared-linux-amd64.deb
    else
        echo "❌ Unsupported OS. Please install cloudflared manually:"
        echo "   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"
        exit 1
    fi
fi

echo "✅ cloudflared is installed"
echo ""

# Check if already logged in
if [ ! -f ~/.cloudflared/cert.pem ]; then
    echo "🔐 Please authenticate with Cloudflare..."
    cloudflared tunnel login
fi

echo "✅ Authenticated with Cloudflare"
echo ""

# Check if tunnel exists
TUNNEL_NAME="live-translation"
TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}' || echo "")

if [ -z "$TUNNEL_ID" ]; then
    echo "🆕 Creating new tunnel: $TUNNEL_NAME"
    cloudflared tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
    echo "✅ Tunnel created: $TUNNEL_ID"
else
    echo "✅ Using existing tunnel: $TUNNEL_ID"
fi

echo ""
echo "📝 Update cloudflared-config.yml with:"
echo "   tunnel: $TUNNEL_ID"
echo "   credentials-file: /root/.cloudflared/$TUNNEL_ID.json"
echo ""

# Update config file
sed -i.bak "s/YOUR_TUNNEL_ID_HERE/$TUNNEL_ID/g" cloudflared-config.yml
echo "✅ Config file updated"

echo ""
echo "🌐 Configure DNS in Cloudflare Dashboard:"
echo "   1. Go to: https://dash.cloudflare.com/"
echo "   2. Select your domain"
echo "   3. Add CNAME records:"
echo "      - translate.yourdomain.com → $TUNNEL_ID.cfargotunnel.com"
echo "      - ws.yourdomain.com → $TUNNEL_ID.cfargotunnel.com"
echo "      - api.yourdomain.com → $TUNNEL_ID.cfargotunnel.com"
echo ""
echo "Press Enter once DNS is configured..."
read

echo ""
echo "🚀 Starting services with Docker Compose..."
docker compose up -d

echo ""
echo "🌐 Starting Cloudflare Tunnel..."
cloudflared tunnel --config cloudflared-config.yml run "$TUNNEL_NAME"
