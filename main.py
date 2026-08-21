"""
main.py — Ghastly AI: Entry point & orchestration

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
from llm_query import (
    query_ollama_stream, query_ollama_vision_stream, tokens_for_style
)
from context_manager import ContextManager, resolve_writable_path
from ghost_overlay import GhostOverlay
import base64
from screen_capture import ScreenCapture, HotkeyListener

# Setup logging — console plus a rotating file. The packaged build runs
# windowed (console=False), so stderr goes nowhere and the file is the only
# record of what the app did.
_log_handlers = [logging.StreamHandler()]
try:
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    _log_path = Path(resolve_writable_path(config.LOG_FILE))
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(RotatingFileHandler(
        _log_path, maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUPS, encoding="utf-8"))
except Exception as _log_err:                      # never die over logging
    print(f"File logging unavailable: {_log_err}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s) %(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=_log_handlers
)
logger = logging.getLogger("ghost-agent")


def _log_unhandled(exc_type, exc, tb):
    """
    Last-resort handler so a crash leaves a trace.

    PyQt aborts the process when an exception escapes a slot, and the windowed
    build has no console — without this the app just vanishes and the log ends
    mid-sentence.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))


sys.excepthook = _log_unhandled


class GhostInterviewAgent:
    def __init__(self):
        # Initialize components
        self.audio = AudioCapture(
            sample_rate=config.SAMPLE_RATE,
            chunk_duration=config.CHUNK_DURATION,
            silence_threshold=config.SILENCE_THRESHOLD,
            silence_duration=config.SILENCE_DURATION,
            min_utterance_sec=config.MIN_UTTERANCE_SEC
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
            opacity_opaque=config.OVERLAY_OPACITY_OPAQUE,
            opacity_translucent=config.OVERLAY_OPACITY_TRANSLUCENT,
            window_title=config.WINDOW_TITLE,
            hotkeys=[("Screen capture", config.SCREEN_CAPTURE_HOTKEY),
                     ("Hide / show", config.PANIC_HOTKEY),
                     ("Opaque / translucent", config.OPACITY_HOTKEY),
                     ("Answer last question", config.RETRY_HOTKEY)],
            languages=config.CODE_LANGUAGES,
            code_language=config.DEFAULT_CODE_LANGUAGE,
            answer_styles=config.ANSWER_STYLES,
            answer_style=config.DEFAULT_ANSWER_STYLE,   # refreshed in initialize()
            audio_devices=self.audio.list_input_devices(),
            audio_device=config.AUDIO_DEVICE,
            auto_scroll=config.AUTO_SCROLL,
            on_question_typed=self.on_question_typed,
            on_retry=self.on_retry,
            on_setup_changed=self.on_setup_changed,
        )

        self.screen_capture = ScreenCapture()
        self.hotkey_listener = HotkeyListener(
            config.SCREEN_CAPTURE_HOTKEY,
            self.on_screen_capture_hotkey
        )
        self.panic_listener = HotkeyListener(
            config.PANIC_HOTKEY,
            self.on_panic_hotkey
        )
        self.opacity_listener = HotkeyListener(
            config.OPACITY_HOTKEY,
            self.on_opacity_hotkey
        )
        self.retry_listener = HotkeyListener(
            config.RETRY_HOTKEY,
            self.on_retry
        )
        # Last question asked, so the retry button has something to re-run.
        self._last_question = None
        # One answer streams at a time. Two questions in quick succession
        # would otherwise interleave their tokens in the same panel.
        self._answer_lock = threading.Lock()
        self._screen_capture_lock = threading.Lock()
        self._audio_answering_lock = threading.Lock()
        self._audio_answering_count = 0

        self.is_running = False
    
    def on_question_typed(self, text: str):
        """A question typed into the overlay — same path as a heard one."""
        logger.info(f"Typed question: {text[:80]}")
        threading.Thread(target=self.process_question, args=(text,),
                         daemon=True).start()

    def on_retry(self):
        """Answer the last question again."""
        if not self._last_question:
            self.overlay.set_status("ready")
            self.overlay.append_html(
                '<div style="color:#334155;font-size:12px;padding-left:4px;">'
                'Nothing to retry yet.</div>')
            logger.info("Retry pressed with no previous question")
            return
        logger.info(f"Retrying: {self._last_question[:80]}")
        threading.Thread(target=self.process_question,
                         args=(self._last_question,), daemon=True).start()

    def on_panic_hotkey(self):
        """Hide or show the overlay without quitting it."""
        logger.info("Panic hotkey pressed")
        self.overlay.toggle_visibility()

    def on_opacity_hotkey(self):
        """Flip the overlay between opaque and translucent."""
        logger.info("Opacity hotkey pressed")
        self.overlay.toggle_opacity()

    def on_setup_changed(self, kind: str, value):
        """
        Setup panel callback: documents were added/removed, or the answer
        language changed. Runs on the Qt thread.
        """
        if kind == "files":
            chars = self.context_mgr.reload_context()
            logger.info(f"Context reloaded after upload: {chars} chars")
        elif kind == "language":
            self.context_mgr.set_code_language(value)
        elif kind == "style":
            self.context_mgr.set_answer_style(value)
        elif kind == "audio_device":
            self.context_mgr.set_audio_device(value)
            self.restart_audio(value)

    def _audio_watchdog(self):
        """
        Watch for capture dying quietly.

        The recorder holds one device open. If Windows switches the default
        speaker — headphones, a Bluetooth headset, a call app taking over —
        the old handle keeps returning silence and the status pill happily
        says "listening" while nothing is heard again.
        """
        while self.is_running:
            time.sleep(config.AUDIO_WATCHDOG_SEC)
            if not self.is_running:
                return
            try:
                if self.audio.default_device_changed():
                    logger.warning("Default audio device changed — reopening capture")
                    self.notify("Audio device changed — reconnected to the new one.")
                    self.restart_audio(self.context_mgr.get_audio_device())
                    continue

                stalled = self.audio.seconds_since_last_frame()
                if stalled is not None and stalled > config.AUDIO_STALL_SEC:
                    logger.warning(f"No audio for {stalled:.0f}s — reopening capture")
                    self.notify("Audio capture stalled — restarting it.")
                    self.restart_audio(self.context_mgr.get_audio_device())
            except Exception as e:
                logger.error(f"Audio watchdog error: {e}")

    def notify(self, message: str):
        """A small grey line in the answer panel."""
        self.overlay.append_html(
            '<div style="color:#334155;font-size:12px;padding-left:4px;'
            f'margin:6px 0;">{message}</div>')

    def restart_audio(self, device_id: str):
        """Point capture at a new device. The queue survives the swap, so the
        listening loop keeps running without noticing."""
        try:
            self.audio.stop()
            self.audio.set_device(device_id)
            self.audio.start()
            logger.info(f"Audio capture restarted on {device_id}")
            self.overlay.set_status("listening")
        except Exception as e:
            logger.error(f"Could not restart audio on {device_id}: {e}")
            self.overlay.set_status("error")

    def initialize(self):
        """Initialize context, overlay, and verify API connectivity."""
        logger.info("=" * 50)
        logger.info("Ghastly AI — Initializing")
        logger.info("=" * 50)
        # Say where data actually went. The chosen folder is resolved before
        # logging exists, so without this line a support question ("where are
        # my uploads?") has no answer anywhere.
        import file_context
        logger.info(f"App data folder: {file_context.writable_base()}")
        
        # Load context
        logger.info("Loading context...")
        self.context_mgr.load_context()
        self.context_mgr.load_state()
        self.context_mgr.reset_state()  # fresh interview, keeps preferences

        # The setup panel's choices persist across launches. A value pinned in
        # .env still wins, and anything unset falls back to the default.
        if config.CODE_LANGUAGE_PINNED or not self.context_mgr.get_code_language():
            self.context_mgr.set_code_language(config.DEFAULT_CODE_LANGUAGE)
        if config.ANSWER_STYLE_PINNED or not self.context_mgr.get_answer_style():
            self.context_mgr.set_answer_style(config.DEFAULT_ANSWER_STYLE)
        # The setup panel is built lazily, so seeding these now is enough for
        # the dropdowns to open on the restored values rather than defaults.
        saved_device = self.context_mgr.get_audio_device()
        if config.AUDIO_DEVICE != "Auto":
            saved_device = config.AUDIO_DEVICE          # .env pins it
            self.context_mgr.set_audio_device(saved_device)
        self.audio.set_device(saved_device)

        self.overlay.code_language = self.context_mgr.get_code_language()
        self.overlay.answer_style = self.context_mgr.get_answer_style()
        self.overlay.audio_device = saved_device
        logger.info(f"Preferences: language={self.context_mgr.get_code_language()}, "
                    f"style={self.context_mgr.get_answer_style()}")
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
        for label, listener, combo in (
            ("Panic", self.panic_listener, config.PANIC_HOTKEY),
            ("Opacity", self.opacity_listener, config.OPACITY_HOTKEY),
            ("Retry", self.retry_listener, config.RETRY_HOTKEY),
        ):
            logger.info(f"Registering {label.lower()} hotkey...")
            if listener.start():
                logger.info(f"{label} hotkey registered: {combo}")
            else:
                logger.warning(f"{label} hotkey registration failed — "
                               f"'{combo}' may be taken by another app")

        logger.info("Registering screen capture hotkey...")
        if self.hotkey_listener.start():
            logger.info(f"Screen capture hotkey registered: {config.SCREEN_CAPTURE_HOTKEY}")
        else:
            logger.warning(
                "Screen capture hotkey registration failed — "
                "screen capture disabled, audio pipeline unaffected"
            )

        # Verify Groq API key is set
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your-groq-api-key":
            logger.warning("Groq API key not set! Edit config.py")
        
        # Verify Ollama API key is set
        if not config.OLLAMA_API_KEY or config.OLLAMA_API_KEY == "your-ollama-api-key":
            logger.warning("Ollama API key not set! Edit config.py")

        # Verify OpenRouter API key is set (used for screen-capture vision queries)
        if not config.OPENROUTER_API_KEY:
            logger.warning(
                "OpenRouter API key not set! Screen capture feature will not work "
                "until OPENROUTER_API_KEY is set in .env"
            )

    def process_question(self, question_text: str):
        """
        Process a single question: send to LLM, stream answer to overlay.
        Target: <2s total after question is complete.
        """
        start_time = time.time()
        self._last_question = question_text

        # Wait for any answer in flight rather than writing over it.
        if not self._answer_lock.acquire(timeout=90):
            logger.warning("Gave up waiting for the previous answer to finish")
            return

        with self._audio_answering_lock:
            self._audio_answering_count += 1
        try:
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
                    max_tokens=tokens_for_style(
                        self.context_mgr.get_answer_style(),
                        config.MAX_ANSWER_CHARS // 4)
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
        finally:
            self._answer_lock.release()
            with self._audio_answering_lock:
                self._audio_answering_count -= 1

    def on_screen_capture_hotkey(self):
        """
        Callback for the screen-capture hotkey. Runs on the `keyboard`
        library's internal hook thread — must return immediately.
        """
        with self._audio_answering_lock:
            audio_busy = self._audio_answering_count > 0
        if audio_busy:
            logger.debug("Audio answer in progress, ignoring screen capture hotkey press")
            return

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
        # Same lock as spoken answers, so a capture cannot interleave with
        # one. Non-blocking: a hotkey press during an answer is dropped
        # rather than queued, which is what pressing it again is for.
        if not self._answer_lock.acquire(blocking=False):
            logger.info("Answer in progress, ignoring the screen capture")
            self._screen_capture_lock.release()
            return
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

            for chunk in query_ollama_vision_stream(
                image_b64=image_b64,
                prompt=config.SCREEN_CAPTURE_PROMPT,
                context=context,
                state=state,
                api_key=config.OLLAMA_API_KEY,
                model=config.OLLAMA_VISION_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                max_tokens=tokens_for_style(
                    self.context_mgr.get_answer_style(),
                    config.MAX_ANSWER_CHARS // 4)
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

            if not (meta and meta.get("error")):
                self.context_mgr.add_qa("[Screen capture]", full_answer)

        except Exception as e:
            logger.error(f"Screen capture query failed: {e}")
            self.overlay.stream_answer(f"\n[Error: {e}]")
        finally:
            self._answer_lock.release()
            self._screen_capture_lock.release()
            self.overlay.set_status("listening")

    def run(self):
        """Main loop: audio processing in background thread, Qt event loop on main thread."""
        self.is_running = True
        
        # Start audio listening in background thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()

        watchdog = threading.Thread(target=self._audio_watchdog, daemon=True)
        watchdog.start()
        
        # Run GUI event loop on main thread (blocks until overlay closed or Ctrl+C)
        self.overlay.exec()

    def _listen_loop(self):
        """Background audio capture → transcribe → query LLM."""
        logger.info("Starting audio capture...")
        self.audio.start()
        self.overlay.set_status("listening")
        
        logger.info("Ghastly AI is LISTENING")
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
        logger.info("Stopping Ghastly AI...")
        self.is_running = False
        
        self.audio.stop()
        self.hotkey_listener.stop()
        self.panic_listener.stop()
        self.opacity_listener.stop()
        self.retry_listener.stop()
        self.overlay.stop()

        logger.info("Ghastly AI stopped")


def claim_single_instance() -> bool:
    """
    True if this is the only copy running.

    Two copies fight over the audio device and the global hotkeys — the
    second one's registration fails, which looks like "the hotkeys stopped
    working" rather than "you started it twice".
    """
    if not sys.platform == "win32":
        return True
    try:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        handle = ctypes.windll.kernel32.CreateMutexW(
            None, False, "Global\\GhastlyAI_SingleInstance")
        if not handle:
            return True                       # cannot tell; do not block
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        # Deliberately leaked: the mutex must outlive this call and is freed
        # by Windows when the process exits.
        return True
    except Exception as e:
        logger.warning(f"Single-instance check skipped: {e}")
        return True


def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    logger.info("Interrupt received, shutting down...")
    if hasattr(signal_handler, 'agent'):
        signal_handler.agent.stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    if config.SINGLE_INSTANCE and not claim_single_instance():
        logger.warning("Another copy of Ghastly AI is already running — "
                       "exiting so the two do not fight over the audio "
                       "device and the hotkeys")
        sys.exit(0)

    agent = GhostInterviewAgent()
    signal_handler.agent = agent
    
    # Initialize (load context, start overlay window on main thread, verify APIs)
    agent.initialize()
    
    # Start listening background thread + main thread GUI loop
    agent.run()