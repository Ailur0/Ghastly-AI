"""
main.py — Ghost Interview Agent: Entry point & orchestration

Pipeline:
  System Audio → VAD → Groq Whisper STT (cloud) → Question Filter →
  Context+State → Ollama LLM (cloud, streaming) → Ghost Overlay

Latency target: <2s from end of interviewer's question to answer displayed.
Zero local model downloads — STT and LLM both cloud-based.
"""

import sys
import os
import time
import logging
import threading
import signal

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from audio_capture import AudioCapture
from transcribe import transcribe, is_question
from llm_query import query_ollama_stream, query_openrouter_vision_stream
from context_manager import ContextManager
from ghost_overlay import GhostOverlay
import base64
from screen_capture import ScreenCapture, HotkeyListener

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s) %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ghost-agent")


class GhostInterviewAgent:
    def __init__(self):
        # Initialize components
        self.audio = AudioCapture(
            sample_rate=config.SAMPLE_RATE,
            chunk_duration=config.CHUNK_DURATION,
            silence_threshold=config.SILENCE_THRESHOLD,
            silence_duration=config.SILENCE_DURATION
        )
        
        self.context_mgr = ContextManager(
            context_file=config.CONTEXT_FILE,
            max_context_chars=config.MAX_CONTEXT_CHARS
        )
        
        self.overlay = GhostOverlay(
            bar_width=config.OVERLAY_BAR_WIDTH,
            bar_height=config.OVERLAY_BAR_HEIGHT,
            panel_width=config.OVERLAY_PANEL_WIDTH,
            panel_height=config.OVERLAY_PANEL_HEIGHT,
            position=config.OVERLAY_POSITION,
        )

        self.screen_capture = ScreenCapture()
        self.hotkey_listener = HotkeyListener(
            config.SCREEN_CAPTURE_HOTKEY,
            self.on_screen_capture_hotkey
        )
        self._screen_capture_lock = threading.Lock()

        self.is_running = False
    
    def initialize(self):
        """Initialize context, overlay, and verify API connectivity."""
        logger.info("=" * 50)
        logger.info("Ghost Interview Agent — Initializing")
        logger.info("=" * 50)
        
        # Load context
        logger.info("Loading context...")
        self.context_mgr.load_context()
        self.context_mgr.load_state()
        self.context_mgr.reset_state()  # fresh interview
        logger.info(f"Context loaded: {len(self.context_mgr.static_context)} chars")
        
        # Start overlay window on main thread
        logger.info("Starting ghost overlay...")
        self.overlay.init_window()
        self.overlay.set_status("ready")
        logger.info("Overlay initialized")
        
        # List audio devices
        logger.info("Available audio devices:")
        self.audio.list_devices()

        # Register screen capture hotkey
        logger.info("Registering screen capture hotkey...")
        if self.hotkey_listener.start():
            logger.info(f"Screen capture hotkey registered: {config.SCREEN_CAPTURE_HOTKEY}")
        else:
            logger.warning(
                f"Screen capture hotkey registration failed — "
                f"screen capture disabled, audio pipeline unaffected"
            )

        # Verify Groq API key is set
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your-groq-api-key":
            logger.warning("Groq API key not set! Edit config.py")
        
        # Verify Ollama API key is set
        if not config.OLLAMA_API_KEY or config.OLLAMA_API_KEY == "your-ollama-api-key":
            logger.warning("Ollama API key not set! Edit config.py")
    
    def process_question(self, question_text: str):
        """
        Process a single question: send to LLM, stream answer to overlay.
        Target: <2s total after question is complete.
        """
        start_time = time.time()
        
        # Update status → Answering
        self.overlay.set_status("answering")
        
        # Show question on overlay immediately
        self.overlay.show_question(question_text)
        
        # Get context and state
        context = self.context_mgr.get_context_string()
        state = self.context_mgr.get_state()
        
        # Stream answer from LLM
        full_answer = ""
        meta = None
        
        try:
            for chunk in query_ollama_stream(
                question=question_text,
                context=context,
                state=state,
                api_key=config.OLLAMA_API_KEY,
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                max_tokens=config.MAX_ANSWER_CHARS // 4
            ):
                if isinstance(chunk, dict) and "_meta" in chunk:
                    meta = chunk["_meta"]
                else:
                    full_answer += chunk
                    self.overlay.stream_answer(chunk)
            
            # Show latency if enabled
            if config.SHOW_LATENCY and meta:
                total_ms = (time.time() - start_time) * 1000
                ttft = meta.get("ttft_ms", 0)
                self.overlay.show_latency(total_ms, ttft)
                
                logger.info(f"Q: '{question_text[:60]}...'")
                logger.info(f"A: '{full_answer[:60]}...'")
                logger.info(f"Total: {total_ms:.0f}ms | TTFT: {ttft:.0f}ms | "
                              f"Tokens: {meta.get('token_count', 0)}")
            
            # Save to state
            self.context_mgr.add_qa(question_text, full_answer)
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            self.overlay.stream_answer(f"\n[Error: {e}]")
            self.overlay.set_status("error")
            return
        
        # Back to listening
        self.overlay.set_status("listening")

    def on_screen_capture_hotkey(self):
        """
        Callback for the screen-capture hotkey. Runs on the `keyboard`
        library's internal hook thread — must return immediately.
        """
        if not self._screen_capture_lock.acquire(blocking=False):
            logger.debug("Screen capture already in progress, ignoring hotkey press")
            return

        proc_thread = threading.Thread(
            target=self._process_screen_capture,
            daemon=True
        )
        proc_thread.start()

    def _process_screen_capture(self):
        """Capture the screen, query the vision LLM, stream to overlay."""
        start_time = time.time()
        try:
            self.overlay.set_status("answering")
            self.overlay.show_question("[Screen capture]")

            try:
                png_bytes = self.screen_capture.capture_primary_monitor()
            except Exception as e:
                logger.error(f"Screen capture failed: {e}")
                self.overlay.stream_answer("[Error: screen capture failed]")
                return

            image_b64 = base64.b64encode(png_bytes).decode("utf-8")

            context = self.context_mgr.get_context_string()
            state = self.context_mgr.get_state()

            full_answer = ""
            meta = None

            for chunk in query_openrouter_vision_stream(
                image_b64=image_b64,
                prompt=config.SCREEN_CAPTURE_PROMPT,
                context=context,
                state=state,
                api_key=config.OPENROUTER_API_KEY,
                model=config.OPENROUTER_VISION_MODEL,
                base_url=config.OPENROUTER_BASE_URL,
                max_tokens=config.MAX_ANSWER_CHARS // 4
            ):
                if isinstance(chunk, dict) and "_meta" in chunk:
                    meta = chunk["_meta"]
                else:
                    full_answer += chunk
                    self.overlay.stream_answer(chunk)

            if config.SHOW_LATENCY and meta:
                total_ms = (time.time() - start_time) * 1000
                ttft = meta.get("ttft_ms", 0)
                self.overlay.show_latency(total_ms, ttft)

            self.context_mgr.add_qa("[Screen capture]", full_answer)

        except Exception as e:
            logger.error(f"Screen capture query failed: {e}")
            self.overlay.stream_answer(f"\n[Error: {e}]")
        finally:
            self._screen_capture_lock.release()
            self.overlay.set_status("listening")

    def run(self):
        """Main loop: audio processing in background thread, Qt event loop on main thread."""
        self.is_running = True
        
        # Start audio listening in background thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        
        # Run GUI event loop on main thread (blocks until overlay closed or Ctrl+C)
        self.overlay.exec()

    def _listen_loop(self):
        """Background audio capture → transcribe → query LLM."""
        logger.info("Starting audio capture...")
        self.audio.start()
        self.overlay.set_status("listening")
        
        logger.info("Ghost Interview Agent is LISTENING")
        logger.info("=" * 50)
        
        try:
            while self.is_running:
                # Get next audio chunk (blocks until available)
                audio_chunk = self.audio.get_audio_chunk(timeout=60)
                
                if audio_chunk is None:
                    continue
                if not self.is_running:
                    break
                
                # Transcribe via Groq API
                self.overlay.set_status("transcribing")
                logger.info("Transcribing via Groq Whisper API...")
                result = transcribe(
                    audio_chunk,
                    sample_rate=config.SAMPLE_RATE,
                )
                
                text = result["text"].strip()
                stt_latency = result.get("latency_ms", 0)
                
                if not text or len(text) < 3:
                    logger.debug(f"Empty transcription, skipping (STT: {stt_latency:.0f}ms)")
                    self.overlay.set_status("listening")
                    continue
                
                logger.info(f"Transcribed ({stt_latency:.0f}ms): '{text[:80]}...'")
                
                # Filter: is this a question or meaningful statement?
                if not is_question(text):
                    logger.debug(f"Not a question, skipping: '{text[:60]}'")
                    self.overlay.set_status("listening")
                    continue
                
                # Process the question in a separate thread
                proc_thread = threading.Thread(
                    target=self.process_question,
                    args=(text,),
                    daemon=True
                )
                proc_thread.start()
                
        except Exception as e:
            logger.error(f"Listening loop error: {e}")
            self.overlay.set_status("error")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all components."""
        logger.info("Stopping Ghost Interview Agent...")
        self.is_running = False
        
        self.audio.stop()
        self.hotkey_listener.stop()
        self.overlay.stop()

        logger.info("Ghost Interview Agent stopped")


def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    logger.info("Interrupt received, shutting down...")
    if hasattr(signal_handler, 'agent'):
        signal_handler.agent.stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    agent = GhostInterviewAgent()
    signal_handler.agent = agent
    
    # Initialize (load context, start overlay window on main thread, verify APIs)
    agent.initialize()
    
    # Start listening background thread + main thread GUI loop
    agent.run()