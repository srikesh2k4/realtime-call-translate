import base64
import io
import os
import time
import asyncio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

# ================= SETUP =================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

SAMPLE_RATE = 16000
WINDOW_SECONDS = 6
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

# room -> {"chunks": [np.ndarray], "last": timestamp, "lock": asyncio.Lock}
buffers: dict[str, dict] = {}

# ================= HELPERS =================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(
        buf,
        pcm,
        SAMPLE_RATE,
        format="WAV",        # REQUIRED
        subtype="PCM_16",
    )
    buf.seek(0)
    return buf.read()


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm ** 2)))


def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"Translate from {src} to {tgt}. Output text only.",
            },
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


# ================= CLEANUP TASK =================

@app.on_event("startup")
async def cleanup_task():
    async def loop():
        while True:
            now = time.time()
            for room in list(buffers.keys()):
                if now - buffers[room]["last"] > 30:
                    del buffers[room]
            await asyncio.sleep(10)

    asyncio.create_task(loop())


# ================= ENDPOINT =================

@app.post("/process")
async def process(req: Request):
    room = req.headers.get("X-Room")
    speaker_lang = req.headers.get("X-Speaker-Lang")

    if not room or not speaker_lang:
        return []

    raw = await req.body()
    pcm = np.frombuffer(raw, dtype=np.float32)

    if pcm.size == 0:
        return []

    # silence guard
    if rms_energy(pcm) < 0.005:
        return []

    if room not in buffers:
        buffers[room] = {
            "chunks": [],
            "last": time.time(),
            "lock": asyncio.Lock(),
        }

    async with buffers[room]["lock"]:
        buffers[room]["chunks"].append(pcm)
        buffers[room]["last"] = time.time()

        total_samples = sum(len(b) for b in buffers[room]["chunks"])

        if total_samples < WINDOW_SAMPLES:
            return []

        audio = np.concatenate(buffers[room]["chunks"])[:WINDOW_SAMPLES]
        buffers[room]["chunks"].clear()

    # ===== PCM → WAV =====
    wav = pcm_to_wav(audio)

    # ===== ASR =====
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
