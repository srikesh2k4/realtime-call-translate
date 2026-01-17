#!/bin/bash
# ============================================
# vast.ai Startup Script
# ============================================

set -e

echo "🚀 Starting Real-Time Translation Worker on vast.ai"
echo "=================================================="

# Start SSH for remote access
echo "🔑 Starting SSH server..."
service ssh start

# Display GPU info
echo ""
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check CUDA
echo "🔧 CUDA Version:"
nvcc --version | grep "release"
echo ""

# Set environment defaults if not provided
export WHISPER_MODEL=${WHISPER_MODEL:-"large-v3"}
export WHISPER_COMPUTE=${WHISPER_COMPUTE:-"float16"}
export BATCH_SIZE=${BATCH_SIZE:-"4"}

echo "📋 Configuration:"
echo "   Model: $WHISPER_MODEL"
echo "   Compute: $WHISPER_COMPUTE"
echo "   Batch Size: $BATCH_SIZE"
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "/app/.env" ]; then
        echo "📁 Loading API key from .env file..."
        export $(cat /app/.env | xargs)
    else
        echo "⚠️  Warning: OPENAI_API_KEY not set. Translation/TTS won't work."
    fi
fi

# Pre-warm GPU
echo "🔥 Warming up GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    # Allocate and free memory to initialize CUDA context
    x = torch.randn(1000, 1000, device='cuda')
    del x
    torch.cuda.empty_cache()
    print(f'   ✓ GPU ready: {torch.cuda.get_device_name(0)}')
    print(f'   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('   ⚠️ No GPU detected!')
"

echo ""
echo "🌐 Starting ML Worker on port 9001..."
echo "=================================================="
echo ""

# Start the ML worker
exec python3 -m uvicorn worker:app \
    --host 0.0.0.0 \
    --port 9001 \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level info
