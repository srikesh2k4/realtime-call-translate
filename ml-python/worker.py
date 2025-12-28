import base64
import io
import os
import time
import asyncio
import threading
import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from openai import OpenAI
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ======================================================
# ENV
# ======================================================

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY missing"

client = OpenAI()  # IMPORTANT: do NOT pass args
app = FastAPI()

# ======================================================
# AUDIO CONFIG
# ======================================================

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

# ======================================================
# LOAD MODELS (ONCE)
# ======================================================

vad_model = load_silero_vad()

MODEL_ID = "google/madlad400-3b-mt"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()

# 🔒 GPU SAFETY LOCK (CRITICAL)
model_lock = threading.Lock()

# ======================================================
# HELPERS
# ======================================================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm * pcm)))


def has_speech(pcm: np.ndarray) -> bool:
    # IMPORTANT: copy() prevents segfault
    audio = torch.from_numpy(pcm.copy())

    timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.6,
        min_speech_duration_ms=400,
        min_silence_duration_ms=300,
    )

    if not timestamps:
        return False

    voiced = sum(t["end"] - t["start"] for t in timestamps)
    return voiced / len(pcm) > 0.25


def valid_text(text: str) -> bool:
    text = text.strip()
    if len(text) < 4:
        return False
    if not any(c.isalpha() for c in text):
        return False
    words = text.split()
    if len(words) == 1:
        return False
    return True


# ======================================================
# MADLAD TRANSLATION (LOW LATENCY)
# ======================================================

def translate(text: str, target_lang: str) -> str:
    prompt = f"<2{target_lang}> {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    with model_lock:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                num_beams=1,          # ⚡ fastest & stable
                early_stopping=True,
            )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ======================================================
# TTS
# ======================================================

def tts(text: str) -> str:
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    return base64.b64encode(audio.read()).decode("utf-8")


# ======================================================
# CORE PIPELINE
# ======================================================

def process_audio(pcm: np.ndarray, speaker_lang: str):
    # 1️⃣ Noise gate
    if rms_energy(pcm) < 0.004:
        return []

    # 2️⃣ Speech gate
    if not has_speech(pcm):
        return []

    # 3️⃣ ASR
    wav = pcm_to_wav(pcm)

    tr = client.audio.transcriptions.create(
        file=("audio.wav", wav),
        model="gpt-4o-mini-transcribe",
        language=speaker_lang,
        temperature=0.0,
    )

    text = tr.text.strip()
    if not valid_text(text):
        return []

    responses = []

    # 4️⃣ Translation (TWO WAY)
    for target_lang in ("en", "hi"):
        if target_lang == speaker_lang:
            continue

        translated = translate(text, target_lang)

        # 🔒 hallucination guard
        if not translated or translated.lower() == text.lower():
            continue

        responses.append({
            "type": "final",
            "sourceText": text,
            "translatedText": translated,
            "audio": tts(translated),
            "sourceLang": speaker_lang,
            "targetLang": target_lang,
        })

    return responses


# ======================================================
# API
# ======================================================

@app.post("/process")
async def process(req: Request):
    speaker_lang = req.headers.get("X-Speaker-Lang")
    if not speaker_lang:
        return []

    raw = await req.body()
    if len(raw) % 4 != 0:
        return []

    pcm = np.frombuffer(raw, dtype=np.float32)
    if pcm.size < WINDOW_SAMPLES:
        return []

    return await run_in_threadpool(
        process_audio,
        pcm,
        speaker_lang,
    )
