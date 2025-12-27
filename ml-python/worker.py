import base64
import io
import os
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

# room -> list[np.ndarray]
buffers = {}

# ================= HELPERS =================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    """
    Convert float32 PCM to WAV bytes.
    FIX: Explicitly specify format='WAV'
    """
    buf = io.BytesIO()
    sf.write(
        buf,
        pcm,
        SAMPLE_RATE,
        format="WAV",        # ✅ REQUIRED
        subtype="PCM_16",
    )
    buf.seek(0)
    return buf.read()


def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"Translate from {src} to {tgt}. No explanation.",
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


# ================= ENDPOINT =================

@app.post("/process")
async def process(req: Request):
    room = req.headers.get("X-Room")
    speaker_lang = req.headers.get("X-Speaker-Lang")

    raw = await req.body()
    pcm = np.frombuffer(raw, dtype=np.float32)

    if pcm.size == 0:
        return []

    buffers.setdefault(room, [])
    buffers[room].append(pcm)

    total_samples = sum(len(b) for b in buffers[room])

    if total_samples < WINDOW_SAMPLES:
        return []

    audio = np.concatenate(buffers[room])[:WINDOW_SAMPLES]
    buffers[room].clear()

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
