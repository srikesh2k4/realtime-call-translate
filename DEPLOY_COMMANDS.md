# 🚀 vast.ai Jupyter Deployment - Complete Commands

## Terminal 1: ML Worker

```bash
# Clone and setup
cd /workspace
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate
git checkout optimise

# Install Python dependencies
pip install uvicorn fastapi transformers faster-whisper silero-vad torch numpy scipy soundfile noisereduce openai python-dotenv accelerate optimum sentencepiece

# Create .env
cd ml-python
echo 'OPENAI_API_KEY=YOUR_KEY_HERE' > .env

# Start ML Worker (wait for models to download ~10 min first time)
python -m uvicorn worker:app --host 0.0.0.0 --port 9001
```

---

## Terminal 2: Go Backend

```bash
# Install Go (if not installed)
wget -q https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Build and run
cd /workspace/realtime-call-translate/backend-go
sed -i 's/go 1.22/go 1.23/g' go.mod
go mod download
go build -o server main.go
export ML_WORKER_HOST=127.0.0.1
export ML_WORKER_PORT=9001
./server
```

---

## Terminal 3: Cloudflare Tunnel

```bash
# Install cloudflared (if not installed)
curl -Lo /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared

# Start tunnel
cloudflared tunnel --url http://localhost:8000
```

Copy the URL: `https://xxx.trycloudflare.com`

---

## Local Mac: Frontend

```bash
cd ~/live_call_translation/frontend

# Create .env with your Cloudflare URL
echo 'VITE_WS_URL=wss://YOUR-TUNNEL-URL.trycloudflare.com/ws' > .env

# Start frontend
npm run dev
```

Open: http://localhost:5173

---

## Health Checks

```bash
curl http://localhost:9001/health   # ML Worker
curl http://localhost:8000/health   # Backend
```
