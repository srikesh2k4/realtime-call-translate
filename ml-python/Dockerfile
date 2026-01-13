# ============================================
# 🚀 NVIDIA CUDA OPTIMIZED ML WORKER
# ============================================
# Base: NVIDIA CUDA 12.1 with cuDNN 8 for PyTorch 2.x
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System deps for audio processing + CUDA
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# ============================================
# CUDA Environment Variables for optimal performance
# ============================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# CTranslate2 (faster-whisper backend) optimizations
ENV CT2_CUDA_ALLOW_FP16=1
ENV CT2_CUDA_CACHING_ALLOCATOR_CONFIG=max_split_size_mb:512

# PyTorch CUDA optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Enable TensorFloat-32 (TF32) for Ampere+ GPUs
ENV NVIDIA_TF32_OVERRIDE=1

# ============================================
# Install PyTorch with CUDA 12.1 support
# ============================================
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 2.x with CUDA 12.1 (optimized for RTX 30xx/40xx)
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================
# Install faster-whisper with CUDA support
# ============================================
# CTranslate2 with CUDA backend (faster-whisper dependency)
RUN pip3 install --no-cache-dir \
    ctranslate2>=4.0.0 \
    faster-whisper>=1.0.0

# ============================================
# Install other dependencies
# ============================================
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ============================================
# Pre-download models (faster cold start)
# ============================================
# Download Whisper large-v3 model
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='float32')" || true

# Download Silero VAD
RUN python3 -c "from silero_vad import load_silero_vad; load_silero_vad()" || true

# ============================================
# Copy application code
# ============================================
COPY worker.py .

# Expose port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Run server with single worker (GPU models aren't multiprocess-safe)
# Use --timeout-keep-alive for WebSocket connections
CMD ["python3", "-m", "uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1", "--timeout-keep-alive", "120"]
