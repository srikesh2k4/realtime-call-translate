import base64
import io
import os
import time
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from openai import OpenAI

# ================= SETUP =================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

SAMPLE_RATE = 16000
WINDOW_SECONDS = 6
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

# room -> {"chunks": list[np.ndarray], "last": timestamp}
buffers: dict[str, dict] = {}
buffers_lock = asyncio.Lock()

# ================= HELPERS =================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    """Convert float32 PCM → WAV"""
    buf = io.BytesIO()
    sf.write(
        buf,
        pcm,
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    buf.seek(0)
    return buf.read()


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm * pcm)))


def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": f"Translate {src} to {tgt}. Only output text."},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    return res.choices[0].message.content.strip()


def tts(text: str) -> str:
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    return base64.b64encode(audio.read()).decode("utf-8")


# ================= BACKGROUND CLEANUP =================

@app.on_event("startup")
async def cleanup_task():
    async def loop():
        while True:
            await asyncio.sleep(10)
            now = time.time()
            async with buffers_lock:
                for room in list(buffers.keys()):
                    if now - buffers[room]["last"] > 30:
                        del buffers[room]

    asyncio.create_task(loop())


# ================= HEAVY PROCESS (THREAD) =================

def process_audio(room: str, speaker_lang: str, pcm: np.ndarray):
    # Silence guard
    if rms_energy(pcm) < 0.002:
        return []

    wav = pcm_to_wav(pcm)

    # ASR
    tr = client.audio.transcriptions.create(
        file=("audio.wav", wav),
        model="gpt-4o-mini-transcribe",
        language=speaker_lang,
        temperature=0.0,
    )

    text = tr.text.strip()
    if not text:
        return []

    responses = []

    for target_lang in ("en", "hi"):
        translated = translate(text, speaker_lang, target_lang)
        audio_b64 = tts(translated)

        responses.append({
            "type": "final",
            "text": text,
            "translated": translated,
            "audio": audio_b64,
            "targetLang": target_lang,
        })

    return responses


# ================= API =================

@app.post("/process")
async def process(req: Request):
    room = req.headers.get("X-Room")
    speaker_lang = req.headers.get("X-Speaker-Lang")

    if not room or not speaker_lang:
        return []

    raw = await req.body()

    # Ensure float32 alignment
    if len(raw) % 4 != 0:
        return []

    pcm = np.frombuffer(raw, dtype=np.float32)

    if pcm.size == 0:
        return []

    async with buffers_lock:
        if room not in buffers:
            buffers[room] = {
                "chunks": [],
                "last": time.time(),
            }

        buffers[room]["chunks"].append(pcm)
        buffers[room]["last"] = time.time()

        total_samples = sum(len(b) for b in buffers[room]["chunks"])
        if total_samples < WINDOW_SAMPLES:
            return []

        audio = np.concatenate(buffers[room]["chunks"])[:WINDOW_SAMPLES]
        buffers[room]["chunks"].clear()

    # 🚀 Run ML safely outside event loop
    return await run_in_threadpool(
        process_audio,
        room,
        speaker_lang,
        audio,
    )
