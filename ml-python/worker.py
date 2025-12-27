import base64
import io
import os
import time
import asyncio
import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from openai import OpenAI
from silero_vad import load_silero_vad, get_speech_timestamps

# ================= SETUP =================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

vad_model = load_silero_vad()

# room -> {"chunks": list[np.ndarray], "last": timestamp}
buffers: dict[str, dict] = {}
buffers_lock = asyncio.Lock()

# ================= AUDIO HELPERS =================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm * pcm)))


def has_real_speech(pcm: np.ndarray) -> bool:
    audio = torch.from_numpy(pcm)

    timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.65,                  # stricter
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )

    if not timestamps:
        return False

    # Require at least 20% voiced audio
    voiced = sum(t["end"] - t["start"] for t in timestamps)
    return (voiced / len(pcm)) > 0.20


# ================= TEXT SANITY =================

def is_valid_transcript(text: str) -> bool:
    text = text.strip()

    if len(text) < 5:
        return False

    # Reject filler words
    fillers = {
        "uh", "um", "hmm", "ah", "oh", "mm",
        "you", "thanks", "thank you"
    }
    if text.lower() in fillers:
        return False

    words = text.split()

    # Single-word hallucination
    if len(words) == 1:
        return False

    # Repetition hallucination
    if len(set(words)) <= 2 and len(words) > 4:
        return False

    # Garbage symbols
    if not any(c.isalpha() for c in text):
        return False

    return True


# ================= NLP =================

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


# ================= CLEANUP =================

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


# ================= CORE PROCESS =================

def process_audio(room: str, speaker_lang: str, pcm: np.ndarray):
    # 1. Hard noise gate
    if rms_energy(pcm) < 0.0045:
        return []

    # 2. Speech probability gate
    if not has_real_speech(pcm):
        return []

    wav = pcm_to_wav(pcm)

    # 3. ASR
    tr = client.audio.transcriptions.create(
        file=("audio.wav", wav),
        model="gpt-4o-mini-transcribe",
        language=speaker_lang,
        temperature=0.0,
    )

    text = tr.text.strip()

    # 4. Text sanity gate
    if not is_valid_transcript(text):
        return []

    responses = []

    for target_lang in ("en", "hi"):
        # 🔒 HARD FIX: skip same language
        if target_lang == speaker_lang:
            continue

        translated = translate(text, speaker_lang, target_lang)

        # Safety: translated text must differ
        if translated.strip().lower() == text.lower():
            continue

        audio_b64 = tts(translated)

        responses.append({
            "type": "final",
            "sourceText": text,
            "translatedText": translated,
            "audio": audio_b64,
            "sourceLang": speaker_lang,
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

    if len(raw) % 4 != 0:
        return []

    pcm = np.frombuffer(raw, dtype=np.float32)
    if pcm.size == 0:
        return []

    async with buffers_lock:
        if room not in buffers:
            buffers[room] = {"chunks": [], "last": time.time()}

        buffers[room]["chunks"].append(pcm)
        buffers[room]["last"] = time.time()

        total = sum(len(c) for c in buffers[room]["chunks"])
        if total < WINDOW_SAMPLES:
            return []

        audio = np.concatenate(buffers[room]["chunks"])[:WINDOW_SAMPLES]
        buffers[room]["chunks"].clear()

    return await run_in_threadpool(
        process_audio,
        room,
        speaker_lang,
        audio,
    )
