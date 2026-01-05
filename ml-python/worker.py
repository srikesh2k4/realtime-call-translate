# ================= HARD SAFETY (MUST BE FIRST) =================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ===============================================================

import io
import time
import base64
import asyncio
import numpy as np
import soundfile as sf
import torch
from threading import Lock
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from openai import OpenAI
from silero_vad import load_silero_vad, get_speech_timestamps

# ================= TORCH SAFETY =================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ================= SETUP =================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Neuro-Intent Aligned Speech Translation System (NIASTS)")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.6
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

DRIFT_THRESHOLD = 0.10
MIN_EMBEDDINGS = 4
MIN_SILENCE_SEC = 0.6

vad_model = load_silero_vad()

rooms = {}
rooms_lock = asyncio.Lock()

vad_lock = Lock()
process_lock = Lock()

# ================= AUDIO HELPERS =================
def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm * pcm)))

def detect_speech(pcm: np.ndarray) -> bool:
    audio = torch.from_numpy(pcm.copy())
    with vad_lock:
        ts = get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.65,
            min_speech_duration_ms=300,
            min_silence_duration_ms=300
        )
    return bool(ts)

# ================= SEMANTIC CORE =================
def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32)

def cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_stable(embeds: list[np.ndarray]) -> bool:
    if len(embeds) < MIN_EMBEDDINGS:
        return False
    drifts = [
        cosine_drift(embeds[i - 1], embeds[i])
        for i in range(1, len(embeds))
    ]
    return float(np.mean(drifts[-3:])) < DRIFT_THRESHOLD

# ================= STRICT VALIDATION =================
def linguistically_complete(text: str) -> bool:
    if len(text.split()) < 4:
        return False
    return any(p in text for p in [".", "?", "!", "।"])

# ================= STRICT TRANSLATION =================
def strict_translate(text: str, src: str, tgt: str) -> str:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a translation engine. "
                    "Translate EXACTLY. "
                    "Do NOT add, remove, explain, rephrase, or comment. "
                    "Output ONLY the translated text."
                )
            },
            {
                "role": "user",
                "content": f"Translate from {src} to {tgt}:\n{text}"
            }
        ],
        temperature=0.0,
        max_output_tokens=300
    )

    out = resp.output_text.strip()

    # HARD OUTPUT GUARDS
    if not out:
        raise ValueError("Empty translation")

    if len(out.split()) > len(text.split()) * 1.2:
        raise ValueError("Hallucination detected")

    forbidden = ["i think", "sorry", "hello", "you said", "as an ai"]
    if any(f in out.lower() for f in forbidden):
        raise ValueError("Conversational output detected")

    return out

# ================= TTS =================
def tts(text: str) -> str:
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )
    return base64.b64encode(audio.read()).decode()

# ================= CORE PIPELINE =================
def process_segment(room: str, lang: str, pcm: np.ndarray):
    with process_lock:

        if rms(pcm) < 0.004:
            return []

        has_voice = detect_speech(pcm)

        state = rooms[room]

        if has_voice:
            state["last_voice"] = time.time()

        if not has_voice:
            return []

        wav = pcm_to_wav(pcm)

        tr = client.audio.transcriptions.create(
            file=("audio.wav", wav),
            model="gpt-4o-mini-transcribe",
            language=lang,
            temperature=0.0
        )

        text = tr.text.strip()
        if len(text) < 4:
            return []

        state["texts"].append(text)
        state["embeddings"].append(embed(text))
        state["last"] = time.time()

        # ===== INTENT COMMIT CONDITIONS =====
        if not semantic_stable(state["embeddings"]):
            return []

        if not linguistically_complete(text):
            return []

        if time.time() - state["last_voice"] < MIN_SILENCE_SEC:
            return []

        # ===== COMMIT =====
        final_text = " ".join(state["texts"])

        outputs = []
        for tgt in ("en", "hi"):
            if tgt == lang:
                continue
            try:
                translated = strict_translate(final_text, lang, tgt)
                audio = tts(translated)
                outputs.append({
                    "type": "final",
                    "sourceText": final_text,
                    "translatedText": translated,
                    "audio": audio,
                    "sourceLang": lang,
                    "targetLang": tgt
                })
            except Exception:
                pass  # FAIL SAFE: SILENCE

        # RESET SESSION (semantic isolation)
        state["texts"].clear()
        state["embeddings"].clear()

        return outputs

# ================= API =================
@app.post("/process")
async def process(req: Request):
    room = req.headers.get("X-Session-Id")
    lang = req.headers.get("X-Speaker-Lang")

    if not room or not lang:
        return []

    raw = await req.body()
    pcm = np.frombuffer(raw, dtype=np.float32)
    if pcm.size == 0:
        return []

    pcm = pcm[:WINDOW_SAMPLES]

    async with rooms_lock:
        if room not in rooms:
            rooms[room] = {
                "texts": [],
                "embeddings": [],
                "last": time.time(),
                "last_voice": 0.0
            }

    return await run_in_threadpool(
        process_segment,
        room,
        lang,
        pcm
    )

# ================= CLEANUP =================
@app.on_event("startup")
async def cleanup():
    async def loop():
        while True:
            await asyncio.sleep(20)
            now = time.time()
            async with rooms_lock:
                for r in list(rooms):
                    if now - rooms[r]["last"] > 40:
                        del rooms[r]
    asyncio.create_task(loop())
