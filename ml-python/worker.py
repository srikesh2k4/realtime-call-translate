import base64
import io
import os
import re
import time
import asyncio
import concurrent.futures
import threading
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

# Silero VAD TorchScript model is NOT thread-safe — concurrent calls
# corrupt internal state → "tensor does not have a device" crash.
# Serialize all VAD calls through this lock.
vad_lock = threading.Lock()

# Thread pool for parallel OpenAI API calls (translate + TTS concurrently)
api_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# Per-room dedup: track last transcript to prevent repetition
room_last_transcript: dict[str, str] = defaultdict(str)

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
    and extract_speech_segments into ONE VAD pass to cut latency.
    
    NOTE: Silero VAD TorchScript model is NOT thread-safe, so all calls
    are serialized through vad_lock."""
    audio = torch.from_numpy(pcm)

    with vad_lock:
        # Reset model state before each call to prevent cross-request leakage
        vad_model.reset_states()
        timestamps = get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,               # stricter voice confidence (was 0.45)
            min_speech_duration_ms=300,   # min 300ms speech segment (was 200ms)
            min_silence_duration_ms=400,  # need 400ms silence to split (was 300ms)
            speech_pad_ms=200,            # pad speech segments (was 150ms)
        )

    if not timestamps:
        return np.array([], dtype=np.float32), False

    # Check voiced ratio — need at least 15% of audio to be speech
    voiced = sum(t["end"] - t["start"] for t in timestamps)
    if (voiced / len(pcm)) < 0.15:
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
    r"(.{5,}?)\1{2,}",    # Any phrase 5+ chars repeated 3+ times
    r"^(you|I|the|a|an|is|it|this|that|and|or|but)\s*[\.\!\?]?$",  # single common words
    r"^(subtitle|caption|translated|translation|srt|sub)s?\s*[\.\!\?:]*$",
]

HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def ngram_repetition_score(text: str, n: int = 3) -> float:
    """Check if text has repeating n-gram phrases — a hallucination signature."""
    words = text.lower().split()
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    from collections import Counter
    counts = Counter(ngrams)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(ngrams)


def text_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two texts."""
    if not a or not b:
        return 0.0
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def is_valid_transcript(text: str, room: str = "") -> bool:
    text = text.strip()

    if len(text) < 3:
        return False

    # Check hallucination patterns
    for pat in HALLUCINATION_RE:
        if pat.search(text):
            print(f"[ML] Hallucination pattern match: '{text}'")
            return False

    words = text.split()

    # Single-word is usually hallucination for short audio
    if len(words) == 1 and len(text) < 15:
        return False

    # Two words that are very short — often hallucination
    if len(words) <= 2 and len(text) < 8:
        return False

    # Repetition hallucination: same word/phrase repeated many times
    if len(words) > 4:
        unique_words = set(w.lower().strip(".,!?") for w in words)
        if len(unique_words) <= 2:
            return False

    # Check for excessive repetition of any single word (> 50% of text)
    if len(words) > 3:
        from collections import Counter
        word_counts = Counter(w.lower().strip(".,!?") for w in words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.5:
            return False

    # N-gram repetition check (catches "I am going I am going I am going")
    if len(words) > 6:
        for n in (2, 3, 4):
            if ngram_repetition_score(text, n) > 0.4:
                print(f"[ML] N-gram repetition detected (n={n}): '{text}'")
                return False

    # Garbage symbols
    if not any(c.isalpha() for c in text):
        return False

    # Dedup: check if this is too similar to the last transcript from this room
    if room and room in room_last_transcript:
        last = room_last_transcript[room]
        sim = text_similarity(text, last)
        if sim > 0.85 and text.lower().strip(".,!? ") == last.lower().strip(".,!? "):
            print(f"[ML] Exact dedup with last transcript: '{text}'")
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
            await asyncio.sleep(60)
            # Clean up stale dedup entries (rooms with no recent activity)
            # In production, track timestamps; for now just keep it simple
            pass
    asyncio.create_task(loop())


# ================= CORE PROCESS =================

def process_audio(room: str, speaker_lang: str, pcm: np.ndarray):
    t0 = time.perf_counter()
    duration = len(pcm) / SAMPLE_RATE
    print(f"[ML] Processing {duration:.1f}s audio from room={room} lang={speaker_lang}")

    # 1. Quick energy gate (skip silence immediately)
    if rms_energy(pcm) < 0.005:
        print("[ML] Rejected: too quiet (RMS < 0.005)")
        return []

    # 2. Lightweight bandpass (frontend already does heavy filtering)
    pcm = light_bandpass(pcm)

    # 3. Single VAD pass — extracts speech segments AND checks validity
    speech_pcm, has_speech = extract_speech_segments(pcm)
    if not has_speech or len(speech_pcm) < SAMPLE_RATE * 0.5:
        print("[ML] Rejected: no real speech detected or too short")
        return []

    # Reject if speech is too short after VAD (< 0.8s of actual speech → hallucination risk)
    speech_duration = len(speech_pcm) / SAMPLE_RATE
    if speech_duration < 0.8:
        print(f"[ML] Rejected: speech too short after VAD ({speech_duration:.2f}s)")
        return []

    wav = pcm_to_wav(speech_pcm)

    # 4. ASR — NO context prompt. Previous transcripts as prompt cause Whisper
    #    to hallucinate/repeat old sentences, especially across languages.
    t_asr = time.perf_counter()
    asr_kwargs = {
        "file": ("audio.wav", wav),
        "model": "gpt-4o-mini-transcribe",
        "language": speaker_lang,
        "temperature": 0.0,
    }

    tr = client.audio.transcriptions.create(**asr_kwargs)
    text = tr.text.strip()
    print(f"[ML] ASR took {time.perf_counter() - t_asr:.2f}s → '{text}'")

    # 5. Text sanity gate
    if not is_valid_transcript(text, room=room):
        print(f"[ML] Rejected transcript: '{text}'")
        return []

    # 6. Update dedup tracking (context no longer used for prompts but kept for dedup)
    room_last_transcript[room] = text

    # 7. Translate + TTS in PARALLEL for each target language
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
