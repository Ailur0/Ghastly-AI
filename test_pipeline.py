"""
test_pipeline.py — End-to-end pipeline test (no overlay, just terminal output)

Audio capture → Groq Whisper STT → Question filter → Ollama LLM → Print to terminal
"""

import sys
import os
import time
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from audio_capture import AudioCapture
from transcribe import transcribe, is_question
from llm_query import query_ollama_stream
from context_manager import ContextManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test-pipeline")

def main():
    print("=" * 60)
    print("  Ghost Interview Agent — Pipeline Test (no overlay)")
    print("=" * 60)
    print()
    
    # Init context
    cm = ContextManager()
    cm.load_context()
    cm.reset_state()
    logger.info(f"Context loaded: {len(cm.static_context)} chars")
    
    # Init audio
    audio = AudioCapture(
        sample_rate=config.SAMPLE_RATE,
        silence_threshold=config.SILENCE_THRESHOLD,
        silence_duration=config.SILENCE_DURATION
    )
    
    print("\n🎵 Audio devices:")
    audio.list_devices()
    
    print("\n" + "=" * 60)
    print("  LISTENING — Play a YouTube interview video NOW")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    audio.start()
    
    chunk_count = 0
    question_count = 0
    
    try:
        while True:
            chunk = audio.get_audio_chunk(timeout=60)
            if chunk is None:
                logger.info("No audio for 60s, still waiting...")
                continue
            
            chunk_count += 1
            rms = np.sqrt(np.mean(np.square(chunk)))
            duration = len(chunk) / config.SAMPLE_RATE
            logger.info(f"Chunk #{chunk_count}: {duration:.1f}s, RMS: {rms:.4f}")
            
            # Transcribe via Groq
            t0 = time.time()
            result = transcribe(chunk, sample_rate=config.SAMPLE_RATE)
            stt_ms = (time.time() - t0) * 1000
            
            text = result["text"].strip()
            if not text or len(text) < 3:
                logger.info(f"  STT empty ({stt_ms:.0f}ms), skipping")
                continue
            
            logger.info(f"  STT ({stt_ms:.0f}ms): \"{text[:100]}\"")
            
            # Check if it's a question
            if not is_question(text):
                logger.info(f"  Not a question, skipping")
                continue
            
            question_count += 1
            print(f"\n{'='*60}")
            print(f"🎤 QUESTION #{question_count}: {text}")
            print(f"{'='*60}")
            print(f"📝 ANSWER: ", end="", flush=True)
            
            # Query Ollama LLM
            t1 = time.time()
            full_answer = ""
            meta = None
            for chunk in query_ollama_stream(
                question=text,
                context=cm.get_context_string(),
                state=cm.get_state(),
                api_key=config.OLLAMA_API_KEY,
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            ):
                if isinstance(chunk, dict) and "_meta" in chunk:
                    meta = chunk["_meta"]
                else:
                    print(chunk, end="", flush=True)
                    full_answer += chunk
            
            llm_ms = (time.time() - t1) * 1000
            total_ms = (time.time() - t0) * 1000
            
            print(f"\n\n⏱ STT: {stt_ms:.0f}ms | LLM: {llm_ms:.0f}ms | TOTAL: {total_ms:.0f}ms | TTFT: {meta.get('ttft_ms', 0):.0f}ms")
            print(f"✅ Under 2s: {'YES ✅' if total_ms < 2000 else 'NO ❌'}")
            
            # Save to context
            cm.add_qa(text, full_answer)
            
    except KeyboardInterrupt:
        print(f"\n\nStopped. Captured {chunk_count} chunks, {question_count} questions answered.")
    finally:
        audio.stop()


if __name__ == "__main__":
    main()