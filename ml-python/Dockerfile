# ============================================
# ML WORKER - LOCAL TRANSLATION (No API)
# ============================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install faster-whisper and translation
RUN pip install --no-cache-dir \
    ctranslate2>=4.0.0 \
    faster-whisper>=1.0.0 \
    transformers>=4.37.0 \
    sentencepiece>=0.1.99 \
    accelerate>=0.26.0

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='float32')" || true
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; t=AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); m=AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')" || true
RUN python3 -c "from silero_vad import load_silero_vad; load_silero_vad()" || true

# Copy worker
COPY worker_v2.py worker.py

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:9001/health || exit 1

CMD ["python3", "-m", "uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1", "--timeout-keep-alive", "120"]
