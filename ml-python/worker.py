import base64
import io
import os
import re
import time
import asyncio
import concurrent.futures
import numpy as np
import soundfile as sf
import torch

from collections import defaultdict
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

vad_model = load_silero_vad()

# Thread pool for parallel OpenAI API calls (translate + TTS concurrently)
api_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# Per-room context for Whisper prompts (last N transcripts)
room_context: dict[str, list[str]] = defaultdict(list)
MAX_CONTEXT = 5  # keep last 5 transcripts for prompt context

# ================= AUDIO HELPERS =================


def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm * pcm)))


def light_bandpass(pcm: np.ndarray) -> np.ndarray:
    """Lightweight FFT bandpass — only cuts sub-80Hz rumble and >4kHz hiss.
    The frontend already does heavy filtering, so this is minimal."""
    fft = np.fft.rfft(pcm)
    freqs = np.fft.rfftfreq(len(pcm), d=1.0 / SAMPLE_RATE)
    fft[freqs < 80] = 0
    fft[freqs > 4000] = 0
    return np.fft.irfft(fft, n=len(pcm)).astype(np.float32)


def extract_speech_segments(pcm: np.ndarray) -> tuple[np.ndarray, bool]:
    """Use Silero VAD to extract only speech segments from the audio.
    Returns (speech_pcm, has_speech) — combines the old has_real_speech
    and extract_speech_segments into ONE VAD pass to cut latency."""
    audio = torch.from_numpy(pcm)

    timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.45,
        min_speech_duration_ms=200,
        min_silence_duration_ms=300,
        speech_pad_ms=150,
    )

    if not timestamps:
        return np.array([], dtype=np.float32), False

    # Check voiced ratio
    voiced = sum(t["end"] - t["start"] for t in timestamps)
    if (voiced / len(pcm)) < 0.10:
        return np.array([], dtype=np.float32), False

    # Concatenate speech segments
    segments = []
    for ts in timestamps:
        start = max(0, ts["start"])
        end = min(len(pcm), ts["end"])
        segments.append(pcm[start:end])

    if not segments:
        return np.array([], dtype=np.float32), False

    return np.concatenate(segments), True


# ================= TEXT SANITY =================

# Common Whisper hallucination patterns
HALLUCINATION_PATTERNS = [
    r"^(thank you|thanks|bye|okay|yes|no|uh|um|hmm|ah|oh|mm)[\.\!\?]?$",
    r"^(please subscribe|like and subscribe|thanks for watching)",
    r"^(music|applause|laughter|silence)\s*$",
    r"^\[.*\]$",           # [Music], [Applause], etc.
    r"^\(.*\)$",           # (Music), (Applause), etc.
    r"^\.+$",              # Just periods
    r"^[\s\.\,\!\?]+$",   # Just punctuation
]

HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def is_valid_transcript(text: str) -> bool:
    text = text.strip()

    if len(text) < 3:
        return False

    # Check hallucination patterns
    for pat in HALLUCINATION_RE:
        if pat.match(text):
            return False

    words = text.split()

    # Single-word is usually hallucination for short audio
    if len(words) == 1 and len(text) < 10:
        return False

    # Repetition hallucination: same word/phrase repeated many times
    if len(words) > 4:
        unique_words = set(w.lower().strip(".,!?") for w in words)
        if len(unique_words) <= 2:
            return False

    # Check for excessive repetition of any single word (> 60% of text)
    if len(words) > 3:
        from collections import Counter
        word_counts = Counter(w.lower().strip(".,!?") for w in words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.6:
            return False

    # Garbage symbols
    if not any(c.isalpha() for c in text):
        return False

    return True


# ================= NLP =================

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    src_name = LANG_NAMES.get(src, src)
    tgt_name = LANG_NAMES.get(tgt, tgt)

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate {src_name} to {tgt_name}. "
                    f"Output ONLY the translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return res.choices[0].message.content.strip()


def tts(text: str, lang: str) -> str:
    voice = "alloy"
    if lang in ("hi", "te", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "ur"):
        voice = "shimmer"

    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        speed=1.15,  # slightly faster playback for lower perceived latency
        response_format="mp3",  # smaller than default, faster transfer
    )
    return base64.b64encode(audio.read()).decode("utf-8")


def translate_and_tts(text: str, src: str, tgt: str) -> dict | None:
    """Run translation then TTS for a single target language.
    Called in parallel for each target language."""
    translated = translate(text, src, tgt)

    if translated.strip().lower() == text.lower() and len(text) > 10:
        return None

    audio_b64 = tts(translated, tgt)

    return {
        "type": "final",
        "sourceText": text,
        "translatedText": translated,
        "audio": audio_b64,
        "sourceLang": src,
        "targetLang": tgt,
    }


# ================= CLEANUP =================

@app.on_event("startup")
async def cleanup_task():
    async def loop():
        while True:
            await asyncio.sleep(30)
            now = time.time()
            stale_rooms = [
                r for r, ctx in room_context.items()
                if not ctx  # empty context
            ]
            for r in stale_rooms:
                del room_context[r]
    asyncio.create_task(loop())


# ================= CORE PROCESS =================

def process_audio(room: str, speaker_lang: str, pcm: np.ndarray):
    t0 = time.perf_counter()
    duration = len(pcm) / SAMPLE_RATE
    print(f"[ML] Processing {duration:.1f}s audio from room={room} lang={speaker_lang}")

    # 1. Quick energy gate (skip silence immediately)
    if rms_energy(pcm) < 0.003:
        print("[ML] Rejected: too quiet")
        return []

    # 2. Lightweight bandpass (frontend already does heavy filtering)
    pcm = light_bandpass(pcm)

    # 3. Single VAD pass — extracts speech segments AND checks validity
    speech_pcm, has_speech = extract_speech_segments(pcm)
    if not has_speech or len(speech_pcm) < SAMPLE_RATE * 0.3:
        print("[ML] Rejected: no real speech detected")
        return []

    wav = pcm_to_wav(speech_pcm)

    # 4. Build context-aware prompt for Whisper
    context_prompt = ""
    if room in room_context and room_context[room]:
        recent = room_context[room][-3:]
        context_prompt = " ".join(recent)
        if len(context_prompt) > 200:
            context_prompt = context_prompt[-200:]

    # 5. ASR (fastest OpenAI transcription model)
    t_asr = time.perf_counter()
    asr_kwargs = {
        "file": ("audio.wav", wav),
        "model": "gpt-4o-mini-transcribe",
        "language": speaker_lang,
        "temperature": 0.0,
    }
    if context_prompt:
        asr_kwargs["prompt"] = context_prompt

    tr = client.audio.transcriptions.create(**asr_kwargs)
    text = tr.text.strip()
    print(f"[ML] ASR took {time.perf_counter() - t_asr:.2f}s → '{text}'")

    # 6. Text sanity gate
    if not is_valid_transcript(text):
        print(f"[ML] Rejected transcript: '{text}'")
        return []

    # 7. Update room context
    room_context[room].append(text)
    if len(room_context[room]) > MAX_CONTEXT:
        room_context[room] = room_context[room][-MAX_CONTEXT:]

    # 8. Translate + TTS in PARALLEL for each target language
    t_mt = time.perf_counter()
    target_langs = [lang for lang in ("en", "hi") if lang != speaker_lang]

    # Submit all translate+TTS jobs to thread pool concurrently
    futures = {
        api_executor.submit(translate_and_tts, text, speaker_lang, tgt): tgt
        for tgt in target_langs
    }

    responses = []
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result(timeout=30)
            if result:
                responses.append(result)
        except Exception as e:
            print(f"[ML] translate_and_tts error: {e}")

    total = time.perf_counter() - t0
    print(f"[ML] Translate+TTS took {time.perf_counter() - t_mt:.2f}s | Total pipeline: {total:.2f}s | {len(responses)} translation(s)")
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

    # Process the full audio buffer directly — the Go backend already
    # does smart VAD-based sentence accumulation, so we get complete
    # utterances here. No need for additional windowing.
    return await run_in_threadpool(
        process_audio,
        room,
        speaker_lang,
        pcm.copy(),  # copy to avoid buffer reference issues
    )
