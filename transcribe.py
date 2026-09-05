"""
transcribe.py — Speech-to-Text using Groq Whisper API (cloud)

Sends audio chunks to Groq's hosted Whisper-large-v3 API.
No local model download needed. ~0.2-0.5s latency per request.

Uses requests library (urllib gets blocked by Cloudflare on Groq).
"""

import io
import wave
import time
import logging
import re
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GROQ_API_KEY, GROQ_WHISPER_MODEL, GROQ_BASE_URL


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """
    Convert numpy float32 audio array to WAV BytesIO buffer.
    No temp files — everything in memory.
    """
    # Clamp and convert to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)          # mono
        wav.setsampwidth(2)           # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return buffer


def transcribe_groq(
    audio: np.ndarray,
    sample_rate: int = 16000,
    api_key: str = GROQ_API_KEY,
    model: str = GROQ_WHISPER_MODEL,
    language: str = "en"
) -> dict:
    """
    Transcribe audio using Groq Whisper API.
    
    Returns:
        dict with:
            - text: transcribed text
            - latency_ms: time taken in milliseconds
    """
    start_time = time.time()
    
    # Convert audio to WAV buffer
    wav_buffer = audio_to_wav_bytes(audio, sample_rate)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhastlyAI/1.0",
    }
    
    files = {
        "file": ("audio.wav", wav_buffer, "audio/wav"),
    }
    
    data = {
        "model": model,
        "language": language,
        "response_format": "json",
    }
    
    try:
        response = requests.post(
            GROQ_BASE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=10
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            text = response.json().get("text", "").strip()
            logger.info(f"Groq STT ({latency_ms:.0f}ms): '{text[:80]}'")
            return {
                "text": text,
                "latency_ms": latency_ms,
            }
        else:
            logger.error(f"Groq API error {response.status_code}: {response.text}")
            return {
                "text": "",
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
    
    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Groq API timeout")
        return {"text": "", "latency_ms": latency_ms, "error": "timeout"}
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Transcription error: {e}")
        return {"text": "", "latency_ms": latency_ms, "error": str(e)}


def describe_stt_error(error: str) -> str:
    """
    Turn a transcribe() error into a line worth showing the user.

    The distinction earns its keep because the response differs: a rejected
    key needs a new key, a rate limit needs a pause, a timeout needs nothing
    but patience. Until this existed, every one of them reached the user as
    silence — the pill went back to "listening" and the app simply never
    answered again.
    """
    if not error:
        return ""

    lowered = error.lower()
    if error.startswith(("HTTP 401", "HTTP 403")):
        return "Speech-to-text rejected the API key — check GROQ_API_KEY."
    if error.startswith("HTTP 429"):
        return "Speech-to-text is rate limited — it should resume shortly."
    if error.startswith("HTTP 5"):
        return "Speech-to-text is down at the provider — nothing to fix here."
    if "timeout" in lowered or "timed out" in lowered:
        return "Speech-to-text timed out — check the connection."
    if any(w in lowered for w in ("connection", "resolve", "network", "unreachable")):
        return "Speech-to-text could not be reached — check the connection."
    if error.startswith("HTTP "):
        # Keep the status line, drop the response body.
        return f"Speech-to-text failed ({error.split(':', 1)[0]})."
    return f"Speech-to-text failed: {error[:120]}"


def transcribe(audio: np.ndarray, sample_rate: int = 16000, **kwargs) -> dict:
    """
    Main transcribe function — routes to Groq API.
    Same interface as before for drop-in compatibility.
    """
    return transcribe_groq(audio, sample_rate, **kwargs)


# Whisper hallucinates these on near-silence, and interviewers say them
# constantly. Every one used to cost an answer.
FILLER = {
    "hmm", "hm", "mhm", "mm-hmm", "mmhmm", "uh", "um", "er", "ah", "oh",
    "yeah", "yea", "yep", "yes", "no", "nope", "okay", "ok", "alright",
    "right", "sure", "cool", "nice", "great", "perfect", "exactly",
    "thank you", "thanks", "thank you so much", "bye", "goodbye",
    "good night", "good morning", "hello", "hi", "hey", "so", "you",
    "that's it", "seriously", "end", "the end", "you too", "same to you",
}

# A question can open with any of these, however short it is.
QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "who", "which", "whose",
    "can you", "could you", "would you", "will you", "do you", "does",
    "did you", "have you", "has", "are you", "is it", "is that", "was",
    "were", "should", "shall", "may i", "tell me", "explain", "describe",
    "walk me", "walk us", "take me", "take us", "run me", "give me",
    "show me", "let's talk", "talk me", "talk about", "difference between",
    "implement", "design", "solve", "optimize", "debug", "fix", "write",
    "compare", "suppose", "imagine", "consider",
)

# Enough of a question buried mid-sentence to count, given some length.
EMBEDDED_ASKS = (
    "tell me", "explain", "describe", "walk me", "walk us", "how do",
    "how would", "how did", "what is", "what's", "what would", "why did",
    "why do", "why not", "can you", "could you", "would you",
)

MIN_STARTER_WORDS = 2      # "Why Python?" is a real question
# "Hmm?" and "You too, right?" are not — but five was cutting real ones:
# "Any questions for me?" and "Big O of that?" are four words each, and
# neither opens with a starter.
MIN_QMARK_WORDS = 4
MIN_EMBEDDED_WORDS = 4

# Matched on word boundaries, which a plain startswith() is not: it read
# "Washington is where I grew up" as a question because it begins with "was",
# and "doesn't matter, let's move on" because it begins with "does". Both
# cost an answer over the panel while the interviewer was still talking.
# Longest alternative first, so "is that" wins over a prefix of itself.
_STARTER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(s) for s in
                       sorted(QUESTION_STARTERS, key=len, reverse=True)) + r")\b"
)


def is_question(text: str) -> bool:
    """
    Is this transcript worth spending an answer on?

    Deliberately strict. Everything reaching here already cost a
    transcription request; what it gates is an LLM call and a panel full of
    text, and an interviewer saying "mm-hmm" every few seconds used to
    trigger both. A missed question costs one press of the retry button or
    the ask box; a false positive buries the answer you actually needed.
    """
    if not text:
        return False

    cleaned = text.strip()
    lowered = cleaned.lower().strip(" .,!?\"'“”‘’-")
    words = [w for w in re.split(r"[^\w']+", lowered) if w]

    # Covers "Thank you." with a full stop, "Bye!", "Okay?" too — the
    # punctuation is stripped out of `lowered` before this runs.
    if not words or lowered in FILLER:
        return False

    if _STARTER_RE.match(lowered) and len(words) >= MIN_STARTER_WORDS:
        return True

    if cleaned.endswith("?") and len(words) >= MIN_QMARK_WORDS:
        return True

    padded = f" {lowered} "
    if len(words) >= MIN_EMBEDDED_WORDS and any(f" {a} " in padded for a in EMBEDDED_ASKS):
        return True

    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Groq Whisper API Test ===\n")
    
    # Test 1: Silence
    print("--- Test 1: Silence ---")
    silence = np.zeros(32000, dtype=np.float32)
    result = transcribe(silence)
    print(f'Text: "{result["text"]}" | Latency: {result["latency_ms"]:.0f}ms')
    
    # Test 2: Tone (should return empty or noise)
    print("\n--- Test 2: 440Hz Tone ---")
    t = np.linspace(0, 2, 32000, False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = transcribe(tone)
    print(f'Text: "{result["text"]}" | Latency: {result["latency_ms"]:.0f}ms')
    
    # Test 3: Question detection
    print("\n--- Test 3: Question Detection ---")
    tests = [
        "What's the difference between SQL and NoSQL?",
        "Tell me about yourself",
        "Hi",
        "How does a neural network work?",
        "um",
        "yeah so",
        "Design a scalable web application",
    ]
    for t in tests:
        print(f'  is_question("{t}") = {is_question(t)}')