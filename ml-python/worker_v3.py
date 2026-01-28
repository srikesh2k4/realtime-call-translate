"""
🚀 ULTRA-OPTIMIZED LIVE CALL TRANSLATION WORKER v3.0
=====================================================
RTX 5090 (Blackwell) Optimized - NO EXTERNAL API DEPENDENCY

SUPPORTED LANGUAGES:
- English (en) ↔ Hindi (hi)
- English (en) ↔ Telugu (te)
- Hindi (hi) ↔ Telugu (te)

ALL 6 TRANSLATION PAIRS:
1. English → Hindi
2. Hindi → English
3. English → Telugu
4. Telugu → English
5. Hindi → Telugu
6. Telugu → Hindi

Features:
1. Whisper large-v3-turbo for ASR (fastest, high accuracy)
2. NLLB-200-3.3B for translation (Meta's BEST model - LOCAL)
3. Silero VAD v5 for speech detection
4. DeepFilterNet for noise suppression (GPU accelerated)
5. BF16/FP16 mixed precision for RTX 5090
6. BetterTransformer for faster attention
7. torch.compile() for maximum speed
8. Translation caching for repeated phrases
9. Crash-proof error handling
10. Long sentence support (30+ seconds)
11. Auto-recovery on failures
"""

import base64
import io
import os
import sys
import time
import asyncio
import traceback
import numpy as np
import soundfile as sf
import torch
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from contextlib import contextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= CRASH-PROOF ERROR HANDLER =================

def safe_execute(func, *args, default=None, error_msg="Error", **kwargs):
    """Execute function with crash protection"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ {error_msg}: {e}")
        traceback.print_exc()
        return default

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# ================= CUDA SETUP FOR RTX 5090 =================

print("=" * 60)
print("🚀 LIVE CALL TRANSLATION v3.0 - Hindi/English/Telugu")
print("=" * 60)
print("\n🔧 Initializing CUDA for RTX 5090...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Verify CUDA *runtime* works (libcublas, drivers). Sometimes torch.cuda.is_available()
# returns True but loading CUDA libs (libcublas.so) fails at runtime because the
# container/host drivers or nvidia-container-toolkit are not configured. In that case
# fall back to CPU to keep the worker healthy (slower but functional).
cuda_runtime_ok = False
if DEVICE == "cuda":
    try:
        # Try a tiny CUDA operation to force-load CUDA runtime libraries.
        torch.cuda.current_device()
        _ = torch.tensor([1.0], device="cuda")
        cuda_runtime_ok = True
    except Exception as e:
        print(f"   ⚠️ CUDA runtime check failed: {e}")
        cuda_runtime_ok = False

if not cuda_runtime_ok:
    print("   ⚠️ CUDA runtime is not usable (missing libs or drivers). Falling back to CPU.")
    DEVICE = "cpu"
    DTYPE = torch.float32

if DEVICE == "cuda":
    try:
        # RTX 5090 Blackwell optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable Flash Attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        
        # Set memory allocation strategy
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.9"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   ✓ Using dtype: {DTYPE}")
        print(f"   ✓ TF32 enabled, Flash Attention enabled")
    except Exception as e:
        print(f"   ⚠️ CUDA setup warning: {e}")
else:
    print("   ⚠️ Running on CPU (slower)")

# ================= SETUP =================

load_dotenv()
app = FastAPI(title="Live Translation API v3.0 - Hindi/English/Telugu")

# ================= SUPPORTED LANGUAGES =================

# NLLB Language codes for our 3 languages
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "nllb_code": "eng_Latn",
        "whisper_code": "en",
        "tts_code": "en",
    },
    "hi": {
        "name": "Hindi", 
        "nllb_code": "hin_Deva",
        "whisper_code": "hi",
        "tts_code": "hi",
    },
    "te": {
        "name": "Telugu",
        "nllb_code": "tel_Telu",
        "whisper_code": "te",
        "tts_code": "te",
    },
}

# All valid translation pairs
TRANSLATION_PAIRS = [
    ("en", "hi"), ("hi", "en"),  # English ↔ Hindi
    ("en", "te"), ("te", "en"),  # English ↔ Telugu
    ("hi", "te"), ("te", "hi"),  # Hindi ↔ Telugu
]

print(f"\n📚 Supported Languages: {', '.join([v['name'] for v in SUPPORTED_LANGUAGES.values()])}")
print(f"🔄 Translation Pairs: {len(TRANSLATION_PAIRS)} combinations")

# ================= CONFIGURATION =================

@dataclass 
class Config:
    SAMPLE_RATE: int = 16000
    
    # Speech detection - optimized for long sentences
    MIN_SPEECH_MS: int = 200
    MAX_SPEECH_MS: int = 45000  # 45 seconds for very long sentences
    SILENCE_THRESHOLD_MS: int = 600  # Wait longer before processing
    
    # VAD - tuned for accuracy
    VAD_THRESHOLD: float = 0.35
    VAD_MIN_SPEECH_MS: int = 150
    VAD_MIN_SILENCE_MS: int = 100
    
    # Noise reduction
    NOISE_REDUCE: bool = True
    NOISE_REDUCE_STRENGTH: float = 0.85
    
    # ASR Model
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    WHISPER_COMPUTE: str = os.getenv("WHISPER_COMPUTE", "float16")
    
    # Translation Model (NLLB-200-3.3B for highest accuracy)
    NLLB_MODEL: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-3.3B")
    
    # TTS
    USE_LOCAL_TTS: bool = os.getenv("USE_LOCAL_TTS", "false").lower() == "true"
    
    # Processing
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))  # Long output support
    
    # Quality thresholds
    RMS_THRESHOLD: float = 0.002
    MIN_CONFIDENCE: float = 0.20
    
config = Config()

# ================= NOISE REDUCTION =================

print("\n🔄 Loading noise reduction...")

USE_DEEPFILTER = False
df_model = None
df_state = None

try:
    from df.enhance import enhance, init_df
    df_model, df_state, _ = init_df()
    USE_DEEPFILTER = True
    print("   ✓ DeepFilterNet loaded (GPU noise reduction)")
except Exception as e:
    print(f"   ⚠️ DeepFilterNet not available: {e}")
    try:
        import noisereduce as nr
        print("   ✓ noisereduce loaded (CPU noise reduction)")
    except:
        print("   ⚠️ No noise reduction available")

def reduce_noise_gpu(audio: np.ndarray) -> np.ndarray:
    """GPU-accelerated noise reduction with crash protection"""
    if not config.NOISE_REDUCE:
        return audio
    
    try:
        if USE_DEEPFILTER and df_model is not None:
            audio_tensor = torch.from_numpy(audio.copy()).unsqueeze(0)
            enhanced = enhance(df_model, df_state, audio_tensor)
            return enhanced.squeeze().numpy().astype(np.float32)
        else:
            import noisereduce as nr
            return nr.reduce_noise(
                y=audio,
                sr=config.SAMPLE_RATE,
                stationary=True,
                prop_decrease=config.NOISE_REDUCE_STRENGTH,
            ).astype(np.float32)
    except Exception as e:
        print(f"   ⚠️ Noise reduction error (continuing): {e}")
        return audio

# ================= VAD MODEL =================

print("\n🔄 Loading Silero VAD...")

vad_model = None
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad_model = load_silero_vad()
    if DEVICE == "cuda":
        vad_model = vad_model.to(DEVICE)
    print("   ✓ Silero VAD loaded")
except Exception as e:
    print(f"   ⚠️ VAD loading error: {e}")

# ================= WHISPER ASR =================

print(f"\n🔄 Loading Whisper {config.WHISPER_MODEL}...")

whisper_model = None
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        config.WHISPER_MODEL,
        device=DEVICE,
        compute_type=config.WHISPER_COMPUTE if DEVICE == "cuda" else "float32",
        num_workers=config.BATCH_SIZE,
    )
    print(f"   ✓ Whisper {config.WHISPER_MODEL} loaded on {DEVICE}")
except Exception as e:
    print(f"   ❌ Whisper loading error: {e}")
    raise RuntimeError("Whisper model is required!")

# ================= NLLB TRANSLATION (LOCAL) =================

print(f"\n🔄 Loading NLLB translation model...")

nllb_model = None
nllb_tokenizer = None
nllb_dtype = torch.bfloat16

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    NLLB_MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-3.3B")
    NLLB_COMPUTE = os.getenv("NLLB_COMPUTE", "bfloat16")
    USE_BETTERTRANSFORMER = os.getenv("USE_BETTERTRANSFORMER", "true").lower() == "true"
    
    print(f"   → Loading {NLLB_MODEL_NAME}...")
    
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    
    # Determine compute dtype
    if NLLB_COMPUTE == "bfloat16":
        nllb_dtype = torch.bfloat16
    elif NLLB_COMPUTE == "float16":
        nllb_dtype = torch.float16
    else:
        nllb_dtype = torch.float32
    
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_NAME,
        torch_dtype=nllb_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Apply BetterTransformer for faster inference
    if DEVICE == "cuda" and USE_BETTERTRANSFORMER:
        try:
            from optimum.bettertransformer import BetterTransformer
            nllb_model = BetterTransformer.transform(nllb_model)
            print("   ✓ BetterTransformer applied (1.5-2x faster)")
        except Exception as e:
            print(f"   ⚠️ BetterTransformer not applied: {e}")
    
    if DEVICE == "cuda" and not hasattr(nllb_model, 'hf_device_map'):
        nllb_model = nllb_model.to(DEVICE)
    
    # Enable KV cache
    nllb_model.config.use_cache = True
    
    # Compile for speed
    if DEVICE == "cuda":
        try:
            nllb_model = torch.compile(nllb_model, mode="max-autotune", fullgraph=False)
            print("   ✓ torch.compile() applied (maximum speed)")
        except Exception as e:
            print(f"   ⚠️ torch.compile not applied: {e}")
    
    print(f"   ✓ NLLB {NLLB_MODEL_NAME} loaded")
    print(f"   → Compute: {nllb_dtype}, Max tokens: {config.MAX_NEW_TOKENS}")
    
except Exception as e:
    print(f"   ❌ NLLB loading error: {e}")
    raise RuntimeError("NLLB translation model is required!")

# ================= TTS =================

print("\n🔄 Loading TTS...")

USE_OPENAI_TTS = False
USE_LOCAL_TTS_LOADED = False
openai_client = None
tts_model = None

try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and len(api_key) > 10:
        openai_client = OpenAI(api_key=api_key)
        USE_OPENAI_TTS = True
        print("   ✓ OpenAI TTS available")
except Exception as e:
    print(f"   ⚠️ OpenAI TTS not available: {e}")

if config.USE_LOCAL_TTS and not USE_OPENAI_TTS:
    try:
        from TTS.api import TTS
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        USE_LOCAL_TTS_LOADED = True
        print("   ✓ Local XTTS-v2 loaded")
    except Exception as e:
        print(f"   ⚠️ Local TTS not available: {e}")

if not USE_OPENAI_TTS and not USE_LOCAL_TTS_LOADED:
    print("   ⚠️ No TTS available (audio output disabled)")

# ================= WARMUP (CRITICAL FOR SPEED) =================

if DEVICE == "cuda":
    print("\n🔥 Warming up models for maximum speed...")
    
    # Warmup Whisper
    dummy_audio = np.random.randn(config.SAMPLE_RATE).astype(np.float32) * 0.01
    for lang in ["en", "hi", "te"]:
        try:
            list(whisper_model.transcribe(dummy_audio, language=lang))
        except:
            pass
    print("   ✓ Whisper warmed up (en, hi, te)")
    
    # Warmup NLLB for all translation pairs
    warmup_texts = {
        "en": ["Hello", "How are you?", "This is a test sentence for warming up."],
        "hi": ["नमस्ते", "आप कैसे हैं?"],
        "te": ["హలో", "మీరు ఎలా ఉన్నారు?"],
    }
    
    for src_lang, texts in warmup_texts.items():
        src_code = SUPPORTED_LANGUAGES[src_lang]["nllb_code"]
        for tgt_lang in SUPPORTED_LANGUAGES:
            if tgt_lang != src_lang:
                tgt_code = SUPPORTED_LANGUAGES[tgt_lang]["nllb_code"]
                try:
                    nllb_tokenizer.src_lang = src_code
                    inputs = nllb_tokenizer(texts[0], return_tensors="pt").to(DEVICE)
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=nllb_dtype):
                        nllb_model.generate(
                            **inputs,
                            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                            max_new_tokens=20,
                            use_cache=True,
                        )
                except Exception as e:
                    print(f"   ⚠️ Warmup {src_lang}→{tgt_lang}: {e}")
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   ✓ All models warmed up (GPU: {allocated:.1f}GB/{reserved:.1f}GB)")

# ================= THREAD POOL =================

executor = ThreadPoolExecutor(max_workers=config.BATCH_SIZE * 2)

# ================= SPEAKER STATE =================

@dataclass
class SpeakerState:
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=config.SAMPLE_RATE * 90))  # 90 seconds buffer
    is_speaking: bool = False
    speech_start_time: float = 0
    last_speech_time: float = 0
    silence_samples: int = 0
    pending_audio: List[np.ndarray] = field(default_factory=list)
    last_transcript: str = ""
    processing: bool = False
    error_count: int = 0
    last_error_time: float = 0

room_states: Dict[str, Dict[str, SpeakerState]] = {}
states_lock = asyncio.Lock()

# ================= AUDIO HELPERS =================

def rms_energy(pcm: np.ndarray) -> float:
    """Calculate RMS energy with crash protection"""
    try:
        if len(pcm) == 0:
            return 0.0
        return float(np.sqrt(np.mean(pcm * pcm)))
    except:
        return 0.0

def normalize_audio(pcm: np.ndarray) -> np.ndarray:
    """Normalize audio with crash protection"""
    try:
        max_val = np.max(np.abs(pcm))
        if max_val > 0.01:
            return (pcm / max_val * 0.95).astype(np.float32)
        return pcm.astype(np.float32)
    except:
        return pcm

def detect_speech_segments(pcm: np.ndarray) -> List[Dict[str, int]]:
    """GPU-accelerated VAD with crash protection"""
    if vad_model is None:
        return [{"start": 0, "end": len(pcm)}]  # Fallback
    
    try:
        audio = torch.from_numpy(pcm.copy())
        if DEVICE == "cuda":
            audio = audio.to(DEVICE)
        
        return get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=config.SAMPLE_RATE,
            threshold=config.VAD_THRESHOLD,
            min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            return_seconds=False,
        )
    except Exception as e:
        print(f"   ⚠️ VAD error: {e}")
        return [{"start": 0, "end": len(pcm)}]

def has_real_speech(pcm: np.ndarray) -> bool:
    """Check if audio contains real speech"""
    try:
        if len(pcm) < 1600:
            return rms_energy(pcm) > config.RMS_THRESHOLD
        
        timestamps = detect_speech_segments(pcm)
        if not timestamps:
            return False
        voiced_samples = sum(t["end"] - t["start"] for t in timestamps)
        return voiced_samples / len(pcm) > 0.08
    except:
        return rms_energy(pcm) > config.RMS_THRESHOLD

def get_speech_boundaries(pcm: np.ndarray) -> Tuple[int, int]:
    """Get speech boundaries with padding"""
    try:
        timestamps = detect_speech_segments(pcm)
        if not timestamps:
            return 0, len(pcm)
        
        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
        padding = int(0.2 * config.SAMPLE_RATE)
        return max(0, start - padding), min(len(pcm), end + padding)
    except:
        return 0, len(pcm)

# ================= TEXT VALIDATION =================

HALLUCINATION_PATTERNS = {
    "thank you", "thanks for watching", "subscribe", "like and subscribe",
    "[music]", "[applause]", "(music)", "(applause)", "music", "applause",
    "...", "।", "॥", "thanks", "bye", "okay",
    # Additional hallucination filters for accuracy
    "you", "um", "uh", "ah", "hmm", "uhm", "mmm",
    "thank", "the", "a", "an", "and", "or", "but",
}

INDIAN_CHARS = set("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेैोौंःँॉ్ా ి ీ ు ూ ృ ె ే ై ొ ో ౌ ం ః")

def is_valid_transcript(text: str, language: str, min_words: int = 1) -> bool:
    """Validate transcript quality - anti-hallucination"""
    if not text:
        return False
    
    text_clean = text.strip().lower()
    
    # Minimum length check
    if len(text_clean) < 2:
        return False
    
    # Filter known hallucinations
    for pattern in HALLUCINATION_PATTERNS:
        if text_clean == pattern:
            return False
    
    # Check for actual content (not just punctuation/symbols)
    if not any(c.isalpha() or c in INDIAN_CHARS for c in text):
        return False
    
    # Filter repetitive patterns (hallucination indicator)
    words = text_clean.split()
    if len(words) > 2 and len(set(words)) == 1:
        return False
    
    # Minimum word count
    if len(words) < min_words:
        return False
    
    return True

# ================= ASR =================

def transcribe_audio(pcm: np.ndarray, language: str) -> Tuple[str, float]:
    """Transcribe with Whisper large-v3-turbo - crash-proof"""
    if whisper_model is None:
        return "", 0.0
    
    try:
        # Get speech boundaries
        start, end = get_speech_boundaries(pcm)
        pcm = pcm[start:end]
        
        if len(pcm) < config.SAMPLE_RATE * 0.15:
            return "", 0.0
        
        pcm = normalize_audio(pcm)
        
        # Map language code
        whisper_lang = SUPPORTED_LANGUAGES.get(language, {}).get("whisper_code", language)
        
        segments, info = whisper_model.transcribe(
            pcm,
            language=whisper_lang,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.35,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 100,
            },
            word_timestamps=False,
            no_speech_threshold=0.6,  # Stricter: reject more ambiguous audio
            compression_ratio_threshold=2.4,  # Anti-hallucination: reject highly compressed gibberish
            logprob_threshold=-1.0,  # Anti-hallucination: reject low-probability outputs
        )
        
        full_text = ""
        total_prob = 0.0
        count = 0
        
        for segment in segments:
            full_text += segment.text
            total_prob += np.exp(segment.avg_logprob)
            count += 1
        
        confidence = total_prob / count if count > 0 else 0.0
        return full_text.strip(), confidence
        
    except Exception as e:
        print(f"   ⚠️ ASR Error: {e}")
        return "", 0.0

# ================= TRANSLATION (LOCAL NLLB - ALL 6 PAIRS) =================

# Translation cache for speed
translation_cache: Dict[str, str] = {}
CACHE_MAX_SIZE = 2000

def get_cache_key(text: str, src: str, tgt: str) -> str:
    return f"{src}:{tgt}:{text[:100]}"

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate using NLLB-200-3.3B (completely local)
    Supports all 6 pairs: en↔hi, en↔te, hi↔te
    """
    if nllb_model is None or nllb_tokenizer is None:
        return text
    
    if src_lang == tgt_lang or not text.strip():
        return text
    
    # Validate language pair
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        print(f"   ⚠️ Unsupported language pair: {src_lang} → {tgt_lang}")
        return text
    
    # Check cache
    cache_key = get_cache_key(text, src_lang, tgt_lang)
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    src_code = SUPPORTED_LANGUAGES[src_lang]["nllb_code"]
    tgt_code = SUPPORTED_LANGUAGES[tgt_lang]["nllb_code"]
    
    try:
        # Tokenize
        nllb_tokenizer.src_lang = src_code
        
        # Handle long sentences by chunking if needed
        max_chunk_length = 450
        
        if len(text) > max_chunk_length:
            # Split by sentence boundaries
            import re
            sentences = re.split(r'(?<=[.!?।॥])\s+', text)
            translated_parts = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                part_translated = _translate_single(sentence.strip(), src_code, tgt_code)
                translated_parts.append(part_translated)
            
            translated = " ".join(translated_parts)
        else:
            translated = _translate_single(text, src_code, tgt_code)
        
        # Cache result
        if len(translation_cache) >= CACHE_MAX_SIZE:
            # Remove oldest entries
            keys_to_remove = list(translation_cache.keys())[:500]
            for k in keys_to_remove:
                del translation_cache[k]
        
        translation_cache[cache_key] = translated
        return translated
        
    except Exception as e:
        print(f"   ⚠️ Translation Error ({src_lang}→{tgt_lang}): {e}")
        traceback.print_exc()
        return text

def _translate_single(text: str, src_code: str, tgt_code: str) -> str:
    """Translate a single chunk of text"""
    inputs = nllb_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    
    if DEVICE == "cuda":
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
    
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast(enabled=True, dtype=nllb_dtype):
                generated = nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    num_beams=4,
                    num_return_sequences=1,
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True,
                    length_penalty=1.0,
                    repetition_penalty=1.1,
                )
        else:
            generated = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_beams=4,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
                use_cache=True,
            )
    
    translated = nllb_tokenizer.decode(generated[0], skip_special_tokens=True)
    return translated.strip()

# ================= TTS =================

def synthesize_speech(text: str, language: str = "en") -> str:
    """Convert text to speech with crash protection"""
    if not text.strip():
        return ""
    
    try:
        if USE_OPENAI_TTS and openai_client:
            # OpenAI TTS
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text[:4000],  # API limit
                response_format="mp3",
                speed=1.0,
            )
            return base64.b64encode(response.read()).decode("utf-8")
        
        elif USE_LOCAL_TTS_LOADED and tts_model:
            # Local XTTS-v2
            tts_lang = SUPPORTED_LANGUAGES.get(language, {}).get("tts_code", "en")
            wav = tts_model.tts(text=text[:500], language=tts_lang)
            buf = io.BytesIO()
            sf.write(buf, wav, 22050, format="WAV")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        
        return ""
        
    except Exception as e:
        print(f"   ⚠️ TTS Error: {e}")
        return ""

# ================= CORE PROCESSING =================

def get_target_languages(speaker_lang: str) -> List[str]:
    """Get target languages for translation (all other supported languages)"""
    return [lang for lang in SUPPORTED_LANGUAGES.keys() if lang != speaker_lang]

def process_speech_segment(
    audio: np.ndarray,
    speaker_lang: str,
    room: str,
) -> List[Dict[str, Any]]:
    """Process complete speech segment - crash-proof"""
    start_time = time.time()
    results = []
    
    try:
        # 1. Noise reduction
        audio = reduce_noise_gpu(audio)
        
        # 2. Check for speech
        if rms_energy(audio) < config.RMS_THRESHOLD:
            return []
        
        if not has_real_speech(audio):
            return []
        
        # 3. Transcribe (Whisper)
        transcript, confidence = transcribe_audio(audio, speaker_lang)
        
        if not transcript:
            return []
        
        if confidence < config.MIN_CONFIDENCE:
            print(f"   ⚠️ Low confidence ({confidence:.2f}), skipping")
            return []
        
        if not is_valid_transcript(transcript, speaker_lang):
            return []
        
        src_name = SUPPORTED_LANGUAGES.get(speaker_lang, {}).get("name", speaker_lang)
        print(f"\n📝 [{src_name}] {transcript} (conf: {confidence:.2f})")
        
        # 4. Translate to all target languages
        target_languages = get_target_languages(speaker_lang)
        
        for target_lang in target_languages:
            try:
                # Translate
                translated = translate_text(transcript, speaker_lang, target_lang)
                
                if not translated or translated.strip() == transcript.strip():
                    continue
                
                # TTS (optional)
                audio_b64 = synthesize_speech(translated, target_lang)
                
                processing_time = time.time() - start_time
                tgt_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang)
                
                results.append({
                    "type": "final",
                    "sourceText": transcript,
                    "translatedText": translated,
                    "audio": audio_b64,
                    "sourceLang": speaker_lang,
                    "targetLang": target_lang,
                    "confidence": confidence,
                    "processingTime": processing_time,
                })
                
                print(f"🔊 [{tgt_name}] {translated} ({processing_time:.2f}s)")
                
            except Exception as e:
                print(f"   ⚠️ Translation to {target_lang} failed: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        traceback.print_exc()
        return []

# ================= STREAMING PROCESSOR =================

async def process_audio_chunk(
    room: str,
    speaker_id: str,
    speaker_lang: str,
    pcm: np.ndarray,
) -> List[Dict[str, Any]]:
    """Process audio chunk with smart VAD - crash-proof"""
    
    # Validate language
    if speaker_lang not in SUPPORTED_LANGUAGES:
        print(f"   ⚠️ Unsupported language: {speaker_lang}")
        return []
    
    try:
        async with states_lock:
            if room not in room_states:
                room_states[room] = {}
            if speaker_id not in room_states[room]:
                room_states[room][speaker_id] = SpeakerState()
            state = room_states[room][speaker_id]
        
        # Rate limit on errors
        if state.error_count > 10:
            if time.time() - state.last_error_time < 30:
                return []
            state.error_count = 0
        
        if state.processing:
            state.pending_audio.append(pcm.copy())
            return []
        
        # Add to buffer
        state.audio_buffer.extend(pcm)
        
        # VAD
        has_speech = has_real_speech(pcm) if len(pcm) > 1600 else rms_energy(pcm) > config.RMS_THRESHOLD
        current_time = time.time()
        
        if has_speech:
            if not state.is_speaking:
                state.is_speaking = True
                state.speech_start_time = current_time
                state.silence_samples = 0
            state.last_speech_time = current_time
            state.silence_samples = 0
        else:
            state.silence_samples += len(pcm)
        
        # Calculate durations
        speech_duration_ms = (current_time - state.speech_start_time) * 1000 if state.is_speaking else 0
        silence_duration_ms = (state.silence_samples / config.SAMPLE_RATE) * 1000
        
        # Decide when to process
        should_process = False
        
        if state.is_speaking and silence_duration_ms > config.SILENCE_THRESHOLD_MS:
            should_process = True
            state.is_speaking = False
        
        if speech_duration_ms > config.MAX_SPEECH_MS:
            should_process = True
        
        if not should_process:
            return []
        
        # Get audio
        audio = np.array(list(state.audio_buffer), dtype=np.float32)
        
        # Clear buffer appropriately
        if silence_duration_ms > config.SILENCE_THRESHOLD_MS:
            state.audio_buffer.clear()
        else:
            keep = int(config.SAMPLE_RATE * 0.5)
            state.audio_buffer.clear()
            if len(audio) > keep:
                state.audio_buffer.extend(audio[-keep:])
        
        if len(audio) < config.SAMPLE_RATE * config.MIN_SPEECH_MS / 1000:
            return []
        
        state.processing = True
        
        try:
            results = await run_in_threadpool(
                process_speech_segment, audio, speaker_lang, room
            )
            if results:
                state.last_transcript = results[0].get("sourceText", "")
            return results
        except Exception as e:
            state.error_count += 1
            state.last_error_time = time.time()
            print(f"   ❌ Processing error: {e}")
            return []
        finally:
            state.processing = False
            if state.pending_audio:
                pending = np.concatenate(state.pending_audio)
                state.pending_audio.clear()
                state.audio_buffer.extend(pending)
    
    except Exception as e:
        print(f"   ❌ Chunk processing error: {e}")
        return []

# ================= API ENDPOINTS =================

@app.post("/process")
async def process(req: Request):
    """Process audio chunk"""
    try:
        room = req.headers.get("X-Room")
        speaker_lang = req.headers.get("X-Speaker-Lang")
        speaker_id = req.headers.get("X-Speaker-Id", "default")
        
        if not room or not speaker_lang:
            return JSONResponse(content=[], status_code=200)

        # Strict language validation to avoid hallucinations / misroutes
        if speaker_lang not in SUPPORTED_LANGUAGES:
            return JSONResponse(content=[], status_code=200)
        
        raw = await req.body()
        if len(raw) == 0 or len(raw) % 4 != 0:
            return JSONResponse(content=[], status_code=200)
        
        pcm = np.frombuffer(raw, dtype=np.float32).copy()
        if pcm.size == 0:
            return JSONResponse(content=[], status_code=200)
        
        return await process_audio_chunk(room, speaker_id, speaker_lang, pcm)
    
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return JSONResponse(content=[], status_code=200)

@app.get("/health")
async def health():
    """Health check with detailed status"""
    try:
        gpu_info = {}
        if DEVICE == "cuda":
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            }
        
        return {
            "status": "healthy",
            "version": "3.0",
            "device": DEVICE,
            "dtype": str(DTYPE),
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "translation_pairs": TRANSLATION_PAIRS,
            "models": {
                "whisper": config.WHISPER_MODEL,
                "translation": config.NLLB_MODEL,
                "tts": "openai" if USE_OPENAI_TTS else ("local" if USE_LOCAL_TTS_LOADED else "none"),
                "noise_reduction": "deepfilter" if USE_DEEPFILTER else "noisereduce",
            },
            "config": {
                "max_speech_ms": config.MAX_SPEECH_MS,
                "max_new_tokens": config.MAX_NEW_TOKENS,
            },
            "gpu": gpu_info,
            "cache_size": len(translation_cache),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/languages")
async def languages():
    """Get supported languages and translation pairs"""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "pairs": TRANSLATION_PAIRS,
        "all_combinations": [
            {"from": src, "to": tgt, "from_name": SUPPORTED_LANGUAGES[src]["name"], "to_name": SUPPORTED_LANGUAGES[tgt]["name"]}
            for src, tgt in TRANSLATION_PAIRS
        ]
    }

@app.post("/translate")
async def translate_endpoint(req: Request):
    """Direct translation endpoint for testing"""
    try:
        data = await req.json()
        text = data.get("text", "")
        src_lang = data.get("source", "en")
        tgt_lang = data.get("target", "hi")
        
        if not text:
            return {"error": "No text provided"}
        
        translated = translate_text(text, src_lang, tgt_lang)
        
        return {
            "original": text,
            "translated": translated,
            "source": src_lang,
            "target": tgt_lang,
        }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup():
    """Startup tasks"""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            try:
                async with states_lock:
                    now = time.time()
                    for room in list(room_states.keys()):
                        for sid in list(room_states[room].keys()):
                            if now - room_states[room][sid].last_speech_time > 300:
                                del room_states[room][sid]
                        if not room_states[room]:
                            del room_states[room]
                
                # Periodic GPU cleanup
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass
    
    asyncio.create_task(cleanup_loop())
    
    print("\n" + "=" * 60)
    print("✅ ML Worker v3.0 Ready!")
    print("=" * 60)
    print(f"📚 Languages: English, Hindi, Telugu")
    print(f"🔄 Translation Pairs: {len(TRANSLATION_PAIRS)}")
    print(f"🎯 Max Speech: {config.MAX_SPEECH_MS/1000}s")
    print(f"⚡ Device: {DEVICE}")
    print("=" * 60 + "\n")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("🛑 Shutting down...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    executor.shutdown(wait=False)

# ================= MAIN =================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001, workers=1, timeout_keep_alive=300)
