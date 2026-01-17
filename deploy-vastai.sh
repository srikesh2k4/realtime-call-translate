#!/bin/bash
# ============================================
# 🚀 VAST.AI DEPLOYMENT SCRIPT
# ============================================
# This script helps deploy the ML Worker to vast.ai
# Run this on your LOCAL machine
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================"
echo "🚀 Real-Time Translation - vast.ai Deploy"
echo "============================================"
echo -e "${NC}"

# Check for vastai CLI
if ! command -v vastai &> /dev/null; then
    echo -e "${YELLOW}⚠️  vast.ai CLI not found. Installing...${NC}"
    pip install vastai
fi

# Check for API key
if [ -z "$VASTAI_API_KEY" ]; then
    echo -e "${YELLOW}Enter your vast.ai API key:${NC}"
    read -s VASTAI_API_KEY
    export VASTAI_API_KEY
fi

vastai set api-key $VASTAI_API_KEY

echo ""
echo -e "${GREEN}✓ vast.ai CLI configured${NC}"
echo ""

# ============================================
# STEP 1: Search for RTX 4090 instances
# ============================================
echo -e "${BLUE}📋 Searching for RTX 4090 instances...${NC}"
echo ""

# Search criteria for RTX 4090
# - GPU: RTX 4090 (24GB VRAM)
# - CUDA: 12.0+
# - Reliability: 95%+
# - Internet: 200+ Mbps

vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 cuda_vers>=12.0 reliability>=0.95 inet_down>=200' \
    --order 'dph_total' \
    --limit 10

echo ""
echo -e "${YELLOW}Select an instance ID from above (or press Enter for auto-select):${NC}"
read INSTANCE_ID

if [ -z "$INSTANCE_ID" ]; then
    # Auto-select cheapest reliable instance
    INSTANCE_ID=$(vastai search offers \
        'gpu_name=RTX_4090 num_gpus=1 cuda_vers>=12.0 reliability>=0.95 inet_down>=200' \
        --order 'dph_total' \
        --limit 1 \
        --raw | jq -r '.[0].id')
    echo -e "${GREEN}Auto-selected instance: $INSTANCE_ID${NC}"
fi

# ============================================
# STEP 2: Create the instance
# ============================================
echo ""
echo -e "${BLUE}🚀 Creating vast.ai instance...${NC}"

# Create instance with our Docker image
# Using CUDA 12.1 base image with SSH
vastai create instance $INSTANCE_ID \
    --image nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 \
    --disk 50 \
    --ssh \
    --direct \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --onstart-cmd "apt-get update && apt-get install -y git curl && curl -fsSL https://get.docker.com | sh"

echo ""
echo -e "${GREEN}✓ Instance created!${NC}"
echo ""

# Wait for instance to be ready
echo -e "${BLUE}⏳ Waiting for instance to start...${NC}"
sleep 30

# Get instance info
INSTANCE_INFO=$(vastai show instances --raw | jq -r '.[] | select(.id=='$INSTANCE_ID')')
SSH_HOST=$(echo $INSTANCE_INFO | jq -r '.ssh_host')
SSH_PORT=$(echo $INSTANCE_INFO | jq -r '.ssh_port')

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ INSTANCE READY!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "SSH Command: ${YELLOW}ssh -p $SSH_PORT root@$SSH_HOST${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. SSH into the instance"
echo "2. Clone your repo: git clone https://github.com/srikesh2k4/realtime-call-translate.git"
echo "3. Run: cd realtime-call-translate && ./deploy-vastai-instance.sh"
echo ""
