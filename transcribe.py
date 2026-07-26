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
        "User-Agent": "GhostInterviewAgent/1.0",
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


def transcribe(audio: np.ndarray, sample_rate: int = 16000, **kwargs) -> dict:
    """
    Main transcribe function — routes to Groq API.
    Same interface as before for drop-in compatibility.
    """
    return transcribe_groq(audio, sample_rate, **kwargs)


def is_question(text: str) -> bool:
    """
    Heuristic: check if transcribed text is likely a question.
    Filters out small talk, filler, and non-questions.
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text_lower = text.lower().strip()
    
    # Question indicators
    question_words = [
        "what", "how", "why", "when", "where", "who", "which",
        "can you", "could you", "would you", "do you", "did you",
        "have you", "are you", "is it", "tell me", "explain",
        "describe", "walk me", "give me", "show me",
        "what's", "difference between", "implement", "design",
        "solve", "optimize", "debug", "fix",
        "walk us", "take us", "let's talk",
    ]
    
    for word in question_words:
        if text_lower.startswith(word):
            return True
    
    if text.strip().endswith("?"):
        return True
    
    # Long enough to be a meaningful statement
    if len(text.split()) >= 5:
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