FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker
COPY worker.py .

# Expose port
EXPOSE 9000

# Run server
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "9000"]
