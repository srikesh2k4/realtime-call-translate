
# ✅ FULL DEPLOYMENT & RUN COMMANDS (FINAL)

## 1️⃣ Go to project directory

```bash
cd /workspace/realtime-call-translate/ml-python
```

---

## 2️⃣ Create virtual environment (once)

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

You should see:

```
(venv)
```

---

## 3️⃣ Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn
```

Verify Gunicorn is installed:

```bash
python -m gunicorn --version
```

✅ Must show a version number.

---

## 4️⃣ Set required environment variables

(VERY IMPORTANT – otherwise server will exit)

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
```

Verify:

```bash
echo $OPENAI_API_KEY
```

---

## 5️⃣ (Optional but recommended) ML stability env vars

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## 6️⃣ Test server ONCE (foreground test)

This ensures there are **no import/startup errors**.

```bash
python -m gunicorn worker:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 \
  -b 0.0.0.0:9001
```

If it starts and listens → press **Ctrl+C** to stop.

---

## 7️⃣ Start server in background (CRASH-ONLY RESTART)

### 🔥 FINAL PRODUCTION COMMAND

```bash
nohup python -m gunicorn worker:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 \
  -b 0.0.0.0:9001 \
  > gunicorn.log 2>&1 &
```

You will see:

```
[1] <pid>
```

---

## 8️⃣ Verify server is running

### Check processes

```bash
ps aux | grep gunicorn | grep -v grep
```

You must see:

* 1 master
* 1 worker

### Check port

```bash
ss -lntp | grep 9001
```

Expected:

```
LISTEN 0.0.0.0:9001
```

### Check logs

```bash
tail -f gunicorn.log
```

Expected:

```
Starting gunicorn
Listening at: http://0.0.0.0:9001
Application startup complete
```

---

## 9️⃣ How crash-restart works (important)

| Event              | Result          |
| ------------------ | --------------- |
| Python exception   | Worker restarts |
| Segmentation fault | Worker restarts |
| Native crash       | Worker restarts |
| Normal running     | NO restart      |
| Idle               | NO restart      |

⚠️ This **does NOT auto-start after reboot** (environment limitation).

---

## 🔁 Stop the server

```bash
pkill gunicorn
```

---

## 🧪 Health test

```bash
curl http://127.0.0.1:9001
```

or test your API:

```bash
curl -X POST http://127.0.0.1:9001/process
```

---

## 📌 OPTIONAL: One-command startup script

Create file:

```bash
nano start_ml.sh
```

Paste:

```bash
#!/bin/bash
source venv/bin/activate
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
nohup python -m gunicorn worker:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 \
  -b 0.0.0.0:9001 \
  > gunicorn.log 2>&1 &
```

Make executable:

```bash
chmod +x start_ml.sh
```

Run anytime:

```bash
./start_ml.sh
```

---

# ✅ FINAL SUMMARY

✔ Server runs
✔ Auto-restarts **only on crash**
✔ No random restarts
✔ No connection refused (after startup)
✔ ML-safe configuration
✔ Production-grade for this environment

