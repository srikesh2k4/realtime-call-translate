import base64
import io
import os
import numpy as np
import soundfile as sf
from collections import defaultdict

from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

# ================= ENV =================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ================= CONSTANTS =================

SAMPLE_RATE = 16000
WINDOW_SECONDS = 6
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

# ================= STATE =================
# room -> list[np.ndarray]
buffers = defaultdict(list)

# ================= HELPERS =================

def pcm_to_wav(pcm: np.ndarray) -> bytes:
    """Convert float32 PCM to WAV (16-bit)."""
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def translate(text: str, mode: int) -> str:
    direction = "English to Hindi" if mode == 0 else "Hindi to English"

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional interpreter. "
                    f"Translate strictly {direction}. "
                    f"No explanations. Preserve meaning."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )

    return res.choices[0].message.content.strip()


def tts(text: str) -> str:
    """
    Generate MP3 speech and return base64.
    IMPORTANT: DO NOT pass `format` (SDK does not support it)
    """
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )

    audio_bytes = audio.read()
    print("🔊 TTS bytes:", len(audio_bytes))

    return base64.b64encode(audio_bytes).decode("utf-8")


# ================= ENDPOINT =================

@app.post("/process")
async def process(req: Request):
    room = req.headers.get("X-Room", "default")
    mode = int(req.headers.get("X-Mode", "0"))

    try:
        # ===== READ AUDIO =====
        raw = await req.body()
        pcm = np.frombuffer(raw, dtype=np.float32)

        if pcm.size == 0:
            return {"type": "empty"}

        buffers[room].append(pcm)
        total_samples = sum(len(x) for x in buffers[room])

        print(f"🎙️ room={room} buffered={total_samples}")

        # ===== WAIT FOR FULL WINDOW =====
        if total_samples < WINDOW_SAMPLES:
            return {"type": "partial"}

        # ===== FINALIZE EXACT WINDOW =====
        audio = np.concatenate(buffers[room])[:WINDOW_SAMPLES]
        buffers[room].clear()

        print(f"📦 Final window ready | samples={len(audio)}")

        wav = pcm_to_wav(audio)

        # ===== ASR =====
        print("🧠 Transcribing")
        tr = client.audio.transcriptions.create(
            file=("audio.wav", wav),
            model="gpt-4o-mini-transcribe",
            language="en" if mode == 0 else "hi",
            temperature=0.0,
        )

        text = tr.text.strip()
        print("📝 ASR:", text)

        if not text:
            return {"type": "empty"}

        # ===== TRANSLATION =====
        translated = translate(text, mode)
        print("🌍 TRANSLATED:", translated)

        # ===== TTS =====
        audio_b64 = tts(translated)

        # ===== RETURN =====
        return {
            "type": "final",
            "text": text,
            "translated": translated,
            "audio": audio_b64,
        }

    except Exception as e:
        print("❌ ERROR:", e)
        buffers[room].clear()
        return {"type": "empty"}
