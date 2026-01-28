# 🚀 vast.ai Jupyter Deployment (No Docker)

Deploy on vast.ai Jupyter instances with Cloudflare tunnel support.

## Quick Start

**Inside vast.ai Jupyter terminal:**
```bash
git clone https://github.com/srikesh2k4/realtime-call-translate.git
cd realtime-call-translate
git checkout optimise
chmod +x setup-vastai-jupyter.sh
./setup-vastai-jupyter.sh
```

## After Setup

**Start all services:**
```bash
./start-all.sh
```

**Expose via Cloudflare (in separate terminal):**
```bash
./start-cloudflare.sh
```
Copy the `https://xxx.trycloudflare.com` URL and use it in your frontend.

## GPU Requirements

| GPU | VRAM | Works? |
|-----|------|--------|
| RTX 4090 | 24GB | ✅ Best |
| RTX 3090 | 24GB | ✅ Good |
| A6000 | 48GB | ✅ Overkill |
| RTX 3080 | 10GB | ❌ Too small |

## Ports

| Port | Service |
|------|---------|
| 9001 | ML Worker |
| 8000 | Go Backend |

## Troubleshooting

**OOM Error:** Use `large-v3` instead of `large-v3-turbo` or reduce `BATCH_SIZE` in `.env`

**Slow startup:** First run downloads ~10GB of models. Subsequent runs are fast.

**Cloudflare not working:** Make sure port 8000 is running before starting cloudflared.
