"""
main.py — Ghastly AI: Entry point & orchestration

Pipeline:
  System Audio → VAD → Groq Whisper STT (cloud) → Question Filter →
  Context+State → Groq LLM (cloud, streaming) → Ghost Overlay

Latency target: <2s from end of interviewer's question to answer displayed.
Zero local model downloads — STT and LLM both cloud-based.
"""

import sys
import os
import time
import logging
import threading
import signal
import difflib
import re
import atexit

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from audio_capture import AudioCapture
from transcribe import transcribe, is_question, describe_stt_error
from llm_query import (
    query_llm_stream, query_vision_stream, tokens_for_style
)
from context_manager import ContextManager, resolve_writable_path
import ghost_overlay
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
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
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


# ── Crash and hang diagnostics ───────────────────────────────────────────
# "It froze and closed and there was nothing in the log" has three causes,
# and the log was blind to all of them:
#
#   a native crash        an access violation inside soundcard, Qt or mss
#                         kills the process outright. Python's excepthook
#                         never runs; there is nothing to catch.
#   a worker thread dying sys.excepthook covers the main thread only, and
#                         this app listens, watches, grabs and answers on
#                         daemon threads. Their tracebacks went to stderr,
#                         which a windowed build does not have.
#   a hang                no exception at all. The log just stops, and stops
#                         somewhere uninformative.
#
# The first two are caught below. The third is what the heartbeat is for.
_crash_log = None
CRASH_LOG_PATH = None
try:
    import faulthandler
    CRASH_LOG_PATH = _log_path.with_name("crash.log")
    # Line-buffered and held open for the life of the process: a native crash
    # gets no chance to flush anything, so nothing may be left pending.
    _crash_log = open(CRASH_LOG_PATH, "a", buffering=1, encoding="utf-8")
    _crash_log.write(f"\n===== session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    faulthandler.enable(file=_crash_log, all_threads=True)
except Exception as _fh_err:
    print(f"Crash traces unavailable: {_fh_err}")


def _log_thread_exception(args):
    """An exception in a worker thread, which sys.excepthook never sees."""
    logger.critical(
        f"Unhandled exception in thread '{getattr(args.thread, 'name', '?')}'",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


threading.excepthook = _log_thread_exception


def dump_all_stacks(reason: str):
    """
    Where every thread is standing right now.

    For a freeze there is no traceback to print, so this is the only way to
    learn which thread stopped and on what. Written while the app is still
    running, because afterwards there is nothing left to ask.
    """
    logger.critical(f"Dumping thread stacks: {reason}")
    if _crash_log is None:
        return
    try:
        _crash_log.write(f"\n--- {time.strftime('%H:%M:%S')} stacks: {reason} ---\n")
        faulthandler.dump_traceback(file=_crash_log, all_threads=True)
        logger.critical(f"Thread stacks written to {CRASH_LOG_PATH}")
    except Exception as e:
        logger.error(f"Could not dump thread stacks: {e}")


@atexit.register
def _log_exit():
    """
    Last line in the log, whatever happened.

    An ordinary quit and a process that was killed look identical in a log
    that simply ends; this distinguishes them.
    """
    logger.info("Process exiting normally")
    if _crash_log is not None:
        try:
            _crash_log.write(f"===== clean exit {time.strftime('%H:%M:%S')} =====\n")
            _crash_log.close()
        except Exception:
            pass


# Did a previous run die with the file picker open? Read once, at import,
# before anything can write the marker again.
import file_context as _file_context
_picker_crashed = bool(_file_context.picker_crashed_last_time())
if _picker_crashed:
    logger.warning("The file picker crashed the app last time — using the "
                   "native Windows picker from now on (it is NOT hidden from "
                   "screen capture)")


def log_environment():
    """
    Everything about this machine worth knowing before reading the rest of
    the log.

    Written for the case where the app is misbehaving on someone else's
    computer and all we will ever get back is this file. Nearly every
    machine-specific failure so far — a decommissioned model, an unwritable
    folder, a dropdown that would not open — came down to something here, and
    none of it was recorded. Values of secrets are never logged, only whether
    they are set and how long they are, which is enough to tell a missing key
    from a truncated one.
    """
    import platform

    logger.info("--- environment ---")
    logger.info(f"  app          : {'frozen exe' if getattr(sys, 'frozen', False) else 'source'}"
                f" | {sys.executable}")
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        logger.info(f"  bundle       : {sys._MEIPASS}")
    logger.info(f"  python       : {platform.python_version()} ({platform.machine()})")

    try:
        v = sys.getwindowsversion()
        # WDA_EXCLUDEFROMCAPTURE needs build 19041. Below that the overlay is
        # plainly visible in a screen share, which is worth knowing up front
        # rather than discovering mid-interview.
        wda = "supported" if v.build >= 19041 else "NOT SUPPORTED (needs build 19041)"
        logger.info(f"  windows      : {platform.version()} build {v.build} — "
                    f"capture hiding {wda}")
    except Exception:
        logger.info(f"  platform     : {platform.platform()}")

    logger.info(f"  working dir  : {os.getcwd()}")
    logger.info(f"  app data     : {_file_context.writable_base()}")
    logger.info(f"  log file     : {_log_path if '_log_path' in globals() else 'n/a'}")

    for note in getattr(config, "STARTUP_NOTES", []):
        logger.info(f"  config       : {note}")

    def _key(name, value):
        return f"{name}={'set, ' + str(len(value)) + ' chars' if value else 'MISSING'}"

    logger.info("  keys         : " + ", ".join([
        _key("GROQ_API_KEY", config.GROQ_API_KEY),
        _key("GROQ_LLM_API_KEY", config.GROQ_LLM_API_KEY)]))
    logger.info(f"  models       : llm={config.GROQ_LLM_MODEL} "
                f"vision={config.GROQ_LLM_VISION_MODEL} stt={config.GROQ_WHISPER_MODEL}")
    logger.info(f"  answers      : temp={config.LLM_TEMPERATURE} "
                f"context_cap={config.MAX_CONTEXT_CHARS} history={config.KEEP_HISTORY}")
    logger.info(f"  vad          : threshold={config.SILENCE_THRESHOLD} "
                f"silence={config.SILENCE_DURATION}s "
                f"utterance={config.MIN_UTTERANCE_SEC}-{config.MAX_UTTERANCE_SEC}s "
                f"ring={config.AUDIO_RING_SEC}s grab={config.GRAB_SECONDS}s")

    from audio_capture import SOUNDCARD_AVAILABLE, SD_AVAILABLE
    logger.info(f"  audio libs   : soundcard={SOUNDCARD_AVAILABLE} "
                f"sounddevice={SD_AVAILABLE}")
    logger.info("--- end environment ---")


# A JPEG starts FF D8; the screen capture falls back to PNG when PIL is
# missing, and the vision request has to declare which it is sending.
JPEG_MAGIC = bytes((0xFF, 0xD8))

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalize_speech(text: str) -> str:
    """
    Flatten a transcript for comparison against another of the same audio.

    Whisper is not deterministic about punctuation or hyphenation between two
    passes — "trade-offs" one time, "tradeoffs" the next — so none of that can
    be allowed to decide whether two transcripts are the same speech.
    """
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


class GhostInterviewAgent:
    def __init__(self):
        # Initialize components
        self.audio = AudioCapture(
            sample_rate=config.SAMPLE_RATE,
            chunk_duration=config.CHUNK_DURATION,
            silence_threshold=config.SILENCE_THRESHOLD,
            silence_duration=config.SILENCE_DURATION,
            min_utterance_sec=config.MIN_UTTERANCE_SEC,
            max_utterance_sec=config.MAX_UTTERANCE_SEC,
            ring_seconds=config.AUDIO_RING_SEC
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
            hotkeys=[("Answer what was just said", config.GRAB_HOTKEY),
                     ("Screen capture", config.SCREEN_CAPTURE_HOTKEY),
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
            force_native_picker=_picker_crashed,
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
        self.grab_listener = HotkeyListener(
            config.GRAB_HOTKEY,
            self.on_grab_hotkey
        )
        # What retry would re-run: ("text", question) for anything spoken or
        # typed, ("vision", image_b64, mime) for a screen capture. Tagged
        # rather than a bare string because the two are answered differently.
        self._last_request = None

        # An outage produces one failure per utterance. The pill tracks every
        # one, but the panel only hears about a given problem once a minute —
        # otherwise a dropped connection writes a wall of identical lines over
        # the answer you were reading.
        self._last_stt_error = None
        self._last_stt_error_at = 0.0

        # What the grab hotkey most recently answered, so the live loop can
        # recognise the same speech arriving through the VAD and not answer
        # it a second time.
        self._grab_covers_until = 0.0
        self._last_grab_text = ""
        self._last_grab_at = 0.0

        # One answer owns the panel at a time, and the newest question wins.
        # Each answer carries a ticket; the moment a newer one is claimed the
        # older ticket goes stale and every write it attempts is dropped, so
        # a slow answer can never land under a question it does not belong
        # to. Waiting on a lock instead used to stall the new question behind
        # the old one for as long as the model took.
        self._answer_state_lock = threading.Lock()
        self._answer_seq = 0
        self._answer_active = False
        self._grab_lock = threading.Lock()

        self.is_running = False

    # === Answer ownership ===

    def _claim_answer(self, supersede: bool = True):
        """
        Take ownership of the answer panel and return a ticket, or None if
        `supersede` is False and something is already streaming.

        Superseding does not stop the old thread — it only makes it stale.
        The old thread notices on its next token, stops reading, and lets go.
        """
        with self._answer_state_lock:
            if self._answer_active and not supersede:
                return None
            was_active = self._answer_active
            self._answer_seq += 1
            self._answer_active = True
            ticket = self._answer_seq
        if was_active:
            logger.info(f"Answer #{ticket} supersedes the one still streaming")
        return ticket

    def _release_answer(self, ticket) -> None:
        """Give the panel back, unless a newer answer already took it."""
        with self._answer_state_lock:
            if ticket == self._answer_seq:
                self._answer_active = False

    def _is_current(self, ticket) -> bool:
        with self._answer_state_lock:
            return ticket == self._answer_seq

    def _emit(self, ticket, method, *args) -> bool:
        """
        Write to the overlay only while this answer is still the current one.

        The check and the write happen under the same lock so a token from a
        superseded answer cannot slip through in between. Overlay methods
        only emit Qt signals, so nothing blocks here.
        """
        with self._answer_state_lock:
            if ticket != self._answer_seq:
                return False
            method(*args)
            return True
    
    def on_question_typed(self, text: str):
        """A question typed into the overlay — same path as a heard one."""
        logger.info(f"Typed question: {text[:80]}")
        threading.Thread(target=self.process_question, args=(text,), name="answer",
                         daemon=True).start()

    def on_retry(self):
        """
        Answer the last question again — or re-read the last screen capture.

        Retry used to only know about spoken and typed questions, so pressing
        it after a capture silently re-answered something older.
        """
        request = self._last_request
        if not request:
            self.overlay.set_status("ready")
            self.overlay.notice("Nothing to retry yet.")
            logger.info("Retry pressed with no previous question")
            return

        if request[0] == "vision":
            _, image_b64, mime = request
            logger.info("Retrying the last screen capture")
            threading.Thread(target=self._answer_screen_capture, name="answer-vision",
                             args=(image_b64, mime), daemon=True).start()
            return

        question = request[1]
        # Forget the recorded attempt first, so regenerating does not show the
        # model its own previous answer to the very question it is redoing.
        # Done here rather than inside process_question so two fast retries
        # cannot interleave.
        self.context_mgr.pop_last_qa(question)
        logger.info(f"Retrying: {question[:80]}")
        threading.Thread(target=self.process_question,
                         args=(question,), daemon=True).start()

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
        elif kind == "hotkey":
            return self._rebind_hotkey(*value)
        elif kind == "style":
            self.context_mgr.set_answer_style(value)
        elif kind == "audio_device":
            self.context_mgr.set_audio_device(value)
            self.restart_audio(value)

    # How long after a grab press the live loop assumes the grab is covering
    # the same speech. Long enough for the grab's own transcription to finish
    # and its answer to be claimed.
    GRAB_COVER_SEC = 4.0
    # And how long its transcript stays around to recognise a late duplicate.
    GRAB_ECHO_SEC = 25.0

    def _duplicates_recent_grab(self, text: str) -> bool:
        """
        Has the grab hotkey already answered this speech?

        Timing alone cannot decide it. The grab reaches back over audio the
        VAD may have emitted a moment earlier, a moment later, or be midway
        through transcribing — every ordering happens, and a live run showed
        the VAD usually wins the race. So the comparison is on what was
        actually said.

        The grab's window is longer than one utterance, so its transcript
        normally *contains* the VAD's rather than equalling it; and the two
        transcriptions of the same audio are not always identical, which is
        why near-matches count too.
        """
        if time.time() < self._grab_covers_until:
            logger.info("Skipping an utterance the grab hotkey is answering")
            return True

        if not self._last_grab_text:
            return False
        if time.time() - self._last_grab_at > self.GRAB_ECHO_SEC:
            return False

        mine = _normalize_speech(text)
        theirs = self._last_grab_text
        if not mine:
            return False
        if mine in theirs:
            logger.info("Skipping an utterance the grab hotkey already answered")
            return True
        if difflib.SequenceMatcher(None, mine, theirs).ratio() >= 0.85:
            logger.info("Skipping a near-duplicate of what the grab answered")
            return True
        return False

    STT_ERROR_REPEAT_SEC = 60

    def _report_stt_error(self, error: str) -> None:
        """
        Tell the user speech-to-text failed, instead of letting it read as
        silence.

        This is the failure that used to be invisible: transcribe() hands back
        an empty string and an `error` nobody looked at, so a rejected key, an
        exhausted quota or a dead connection all arrived as "nothing was said"
        — the pill went back to listening and the app simply never answered
        again for the rest of the interview.
        """
        message = describe_stt_error(error)
        logger.error(f"STT failed: {error}")
        self.overlay.set_status("error")

        now = time.time()
        if (message != self._last_stt_error
                or now - self._last_stt_error_at > self.STT_ERROR_REPEAT_SEC):
            self.overlay.notice(message)
            self._last_stt_error = message
            self._last_stt_error_at = now

    def _rebind_hotkey(self, label: str, new_val: str):
        """
        Point one global hotkey at a new combo. Returns (ok, message).

        Registration can fail — `keyboard` rejects combos Qt is happy to hand
        us — and by then the old binding is already gone, so the action would
        be dead until restart with the panel still claiming success. Put the
        old combo back and say so instead.
        """
        listener_map = {
            "Answer what was just said": ("GRAB_HOTKEY", self.grab_listener),
            "Screen capture": ("SCREEN_CAPTURE_HOTKEY", self.hotkey_listener),
            "Hide / show": ("PANIC_HOTKEY", self.panic_listener),
            "Opaque / translucent": ("OPACITY_HOTKEY", self.opacity_listener),
            "Answer last question": ("RETRY_HOTKEY", self.retry_listener),
        }

        if label not in listener_map:
            logger.warning(f"Unknown hotkey label: {label}")
            return False, f"No hotkey called '{label}'."

        env_key, listener = listener_map[label]
        old_val = listener.hotkey

        if old_val == new_val:
            return True, f"'{label}' is already {new_val}."

        # Two actions sharing a combo means both fire, and stop() unregisters
        # by combo string — so it would tear down the wrong one. Refuse.
        for other_label, other_val in self.overlay.hotkeys:
            if other_label != label and other_val == new_val:
                return False, f"{new_val} is already '{other_label}'."

        logger.info(f"Updating hotkey {env_key}: {old_val} -> {new_val}")
        listener.stop()
        listener.hotkey = new_val
        if not listener.start():
            listener.hotkey = old_val
            listener.start()
            logger.warning(f"Hotkey '{new_val}' rejected — kept '{old_val}'")
            return False, f"Couldn't register {new_val} — kept {old_val}."

        # Only persist a combo that actually took.
        saved = config.update_env_file(env_key, new_val)
        os.environ[env_key] = new_val
        setattr(config, env_key, new_val)
        self.overlay.set_hotkey(label, new_val)

        if not saved:
            return True, f"'{label}' is now {new_val}, but won't survive a restart."
        return True, f"'{label}' is now {new_val}."

    UI_STALL_SEC = 6.0

    def _heartbeat(self):
        """
        Proof of life, and the last useful line in the log when it freezes.

        A hang leaves no exception and no traceback — the log simply stops,
        and until now it stopped somewhere that said nothing. Every interval
        this records what was alive: the threads still running, whether the Qt
        event loop is still servicing timers, whether audio is still arriving,
        and what the app was in the middle of. Reading backwards from the end
        of a truncated log, that is the difference between "the UI wedged" and
        "the capture thread died" and "the process was killed from outside".

        A stalled event loop also dumps every thread's stack while the app is
        still running to write it.
        """
        stalled_reported = False
        while self.is_running:
            time.sleep(config.HEARTBEAT_SEC)
            if not self.is_running:
                return
            try:
                ui_age = ghost_overlay.seconds_since_ui_tick()
                alive, capture_reason = self.audio.capture_alive()
                threads = sorted(t.name for t in threading.enumerate() if t.is_alive())

                logger.info(
                    f"Heartbeat: ui_last_ran={('%.1fs ago' % ui_age) if ui_age is not None else 'never'} "
                    f"| capture={'alive' if alive else 'DEAD (%s)' % capture_reason} "
                    f"| answering={self._answer_active} "
                    f"| queued_audio={self.audio.audio_queue.qsize()} "
                    f"| threads={len(threads)} {threads}")

                # The interface has stopped responding but the process has
                # not died, which is the state a person describes as "it
                # froze". Catch it while there is still something to ask.
                if ui_age is not None and ui_age > self.UI_STALL_SEC:
                    if not stalled_reported:
                        dump_all_stacks(f"UI thread has not run for {ui_age:.1f}s")
                        stalled_reported = True
                elif stalled_reported:
                    logger.warning("UI thread is responding again")
                    stalled_reported = False
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}", exc_info=True)

    MAX_CAPTURE_RESTARTS = 3

    def _audio_watchdog(self):
        """
        Watch for capture dying quietly.

        Two ways it can. Windows switches the default speaker — headphones, a
        Bluetooth headset, a call app taking over — and the old handle keeps
        returning silence while the pill says "listening". Or the capture
        thread throws, logs, and returns: is_running stays True, the queue
        never fills again, and nothing anywhere says so.

        The old stall check could not catch either. WASAPI loopback delivers
        frames of zeros when nothing is playing, so "nobody is talking" and
        "the device is dead" produce identical frame timestamps — which is
        why it is gone and thread liveness took its place.
        """
        restarts = 0
        while self.is_running:
            time.sleep(config.AUDIO_WATCHDOG_SEC)
            if not self.is_running:
                return
            try:
                alive, reason = self.audio.capture_alive()
                if not alive:
                    if restarts >= self.MAX_CAPTURE_RESTARTS:
                        continue          # already told them; don't thrash
                    restarts += 1
                    logger.error(f"Audio capture stopped ({reason}) — "
                                 f"restart {restarts}/{self.MAX_CAPTURE_RESTARTS}")
                    self.restart_audio(self.context_mgr.get_audio_device())
                    still_alive, _ = self.audio.capture_alive()
                    if still_alive:
                        self.notify("Audio capture stopped and was restarted.")
                    elif restarts >= self.MAX_CAPTURE_RESTARTS:
                        self.overlay.set_status("error")
                        self.notify(f"Audio capture keeps failing ({reason}). "
                                    f"Pick a different source in setup.")
                    continue

                if self.audio.default_device_changed():
                    logger.warning("Default audio device changed — reopening capture")
                    self.notify("Audio device changed — reconnected to the new one.")
                    self.restart_audio(self.context_mgr.get_audio_device())
            except Exception as e:
                logger.error(f"Audio watchdog error: {e}")

    def notify(self, message: str):
        """A small grey line in the answer panel."""
        self.overlay.notice(message)

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
        log_environment()
        
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
        if _picker_crashed:
            self.overlay.append_html(
                '<div style="color:#B45309;font-size:12px;padding-left:4px;'
                'margin:6px 0;">The file picker closed the app last time, so '
                'the Windows one is being used instead. It is visible in a '
                'screen share while open.</div>')
        logger.info("Overlay initialized")
        
        # List audio devices
        logger.info("Available audio devices:")
        self.audio.list_devices()

        # Register screen capture hotkey
        for label, listener, combo in (
            ("Panic", self.panic_listener, config.PANIC_HOTKEY),
            ("Opacity", self.opacity_listener, config.OPACITY_HOTKEY),
            ("Retry", self.retry_listener, config.RETRY_HOTKEY),
            ("Grab", self.grab_listener, config.GRAB_HOTKEY),
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

        # Two separate keys, and each fails differently: without the first
        # nothing is ever transcribed, without the second nothing is ever
        # answered. Both name the variable and the file to put it in, because
        # the old wording sent people to config.py to edit a key that is read
        # from .env.
        if not config.GROQ_API_KEY or config.GROQ_API_KEY.startswith("your-"):
            logger.warning("GROQ_API_KEY is not set in .env — speech-to-text "
                           "will fail on every utterance")
        if not config.GROQ_LLM_API_KEY or config.GROQ_LLM_API_KEY.startswith("your-"):
            logger.warning("GROQ_LLM_API_KEY is not set in .env — questions "
                           "will be transcribed but never answered")

    def process_question(self, question_text: str):
        """
        Process a single question: send to LLM, stream answer to overlay.
        Target: <2s total after question is complete.
        """
        start_time = time.time()
        self._last_request = ("text", question_text)

        # The newest question always wins the panel.
        ticket = self._claim_answer()

        try:
            self._emit(ticket, self.overlay.set_status, "answering")
            self._emit(ticket, self.overlay.show_question, question_text)

            # Get context and state
            context = self.context_mgr.get_context_string()
            state = self.context_mgr.get_state()

            # Stream answer from LLM
            full_answer = ""
            meta = None
            superseded = False

            stream = query_llm_stream(
                question=question_text,
                context=context,
                state=state,
                api_key=config.GROQ_LLM_API_KEY,
                model=config.GROQ_LLM_MODEL,
                base_url=config.GROQ_LLM_BASE_URL,
                max_tokens=tokens_for_style(
                    self.context_mgr.get_answer_style(),
                    config.MAX_ANSWER_CHARS // 4)
            )
            try:
                for chunk in stream:
                    if isinstance(chunk, dict) and "_meta" in chunk:
                        meta = chunk["_meta"]
                    else:
                        full_answer += chunk
                        # A write that gets refused means a newer question
                        # owns the panel now; stop reading this one.
                        if not self._emit(ticket, self.overlay.stream_answer, chunk):
                            superseded = True
                            break

                if superseded:
                    logger.info(f"Dropped a superseded answer after "
                                f"{len(full_answer)} chars")
                    return

                # Show latency if enabled
                if config.SHOW_LATENCY and meta:
                    total_ms = (time.time() - start_time) * 1000
                    ttft = meta.get("ttft_ms", 0)
                    self._emit(ticket, self.overlay.show_latency, total_ms, ttft)

                    logger.info(f"Q: '{question_text[:60]}...'")
                    logger.info(f"A: '{full_answer[:60]}...'")
                    logger.info(f"Total: {total_ms:.0f}ms | TTFT: {ttft:.0f}ms | "
                                f"Tokens: {meta.get('token_count', 0)}")

                # Save to state
                self.context_mgr.add_qa(question_text, full_answer)

            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                self._emit(ticket, self.overlay.stream_answer, f"\n[Error: {e}]")
                self._emit(ticket, self.overlay.set_status, "error")
                return
            finally:
                # Abandoned mid-stream when superseded — closing it returns
                # the connection instead of leaving it to the collector.
                stream.close()

            # Back to listening
            self._emit(ticket, self.overlay.set_status, "listening")
        finally:
            self._release_answer(ticket)

    def on_screen_capture_hotkey(self):
        """
        Callback for the screen-capture hotkey. Runs on the `keyboard`
        library's internal hook thread — must return immediately.
        """
        proc_thread = threading.Thread(
            target=self._process_screen_capture,
            daemon=True
        )
        proc_thread.start()

    def on_grab_hotkey(self):
        """
        Callback for the grab hotkey. Runs on the `keyboard` library's
        internal hook thread — must return immediately.
        """
        if not self._grab_lock.acquire(blocking=False):
            logger.debug("Grab already in progress, ignoring hotkey press")
            return
        # Claimed here rather than inside the thread: the VAD may already be
        # transcribing this same speech, and the listening loop checks this
        # before spending an LLM call on it.
        self._grab_covers_until = time.time() + self.GRAB_COVER_SEC
        threading.Thread(target=self._process_grab, name="grab", daemon=True).start()

    def _process_grab(self):
        """
        Transcribe the last GRAB_SECONDS of audio and answer it.

        This is the deliberate path: it takes the window as heard rather than
        as VAD carved it, and it skips is_question() entirely. Pressing the
        key is the intent signal, so a question the filter would have thrown
        away — or one chopped in half by a pause — still gets answered.
        """
        try:
            audio = self.audio.grab_recent(config.GRAB_SECONDS)
            if audio is None:
                self.notify("Nothing buffered yet — is the audio source right?")
                return

            secs = len(audio) / config.SAMPLE_RATE
            logger.info(f"Grab hotkey: transcribing the last {secs:.1f}s")
            self.overlay.set_status("transcribing")

            # The VAD is still holding the very speech just taken out of the
            # ring. Left alone it emits it a moment later, it gets transcribed
            # a second time, and that duplicate answer supersedes this one —
            # so the answer deliberately asked for gets wiped while being read.
            self.audio.suppress_pending(config.SILENCE_DURATION + 0.5)

            result = transcribe(audio, sample_rate=config.SAMPLE_RATE)

            # Say what actually went wrong. Reporting a failed request as
            # "nothing was said" sent the user looking at their audio device
            # when the real answer was a rejected key.
            if result.get("error"):
                self._report_stt_error(result["error"])
                return

            text = result["text"].strip()
            self._last_stt_error = None
            if not text or len(text) < 3:
                logger.info("Grab found no speech in the window")
                self.overlay.set_status("listening")
                self.notify(f"Nothing was said in the last {secs:.0f} seconds.")
                return

            logger.info(f"Grab transcribed ({result.get('latency_ms', 0):.0f}ms): "
                        f"'{text[:80]}'")
            self._last_grab_text = _normalize_speech(text)
            self._last_grab_at = time.time()
        except Exception as e:
            logger.error(f"Grab failed: {e}")
            self.overlay.set_status("error")
            self.notify(f"Could not answer what was just said: {e}")
            return
        finally:
            # Released before the answer starts, not after: pressing again
            # mid-answer is how you supersede one, so it must not be
            # swallowed as "a grab is already running".
            self._grab_lock.release()

        self.process_question(text)

    def _process_screen_capture(self):
        """Capture the screen, query the vision LLM, stream to overlay."""
        try:
            png_bytes = self.screen_capture.capture_primary_monitor()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            self.overlay.set_status("error")
            self.notify("Could not capture the screen.")
            return

        # JPEG from the compressed path, PNG from the no-PIL fallback.
        mime = "image/jpeg" if png_bytes[:2] == JPEG_MAGIC else "image/png"
        image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        self._answer_screen_capture(image_b64, mime)

    def _answer_screen_capture(self, image_b64: str, mime: str):
        """
        Ask the vision model about an already-captured frame.

        Separate from the capture itself so retry can re-ask about the same
        screen. The frame used to be discarded the moment the request was
        built, which left retry after a capture silently re-answering an
        older spoken question instead.
        """
        start_time = time.time()
        # Unlike a spoken question, a capture does not supersede: a press
        # while an answer is streaming is dropped rather than queued, which
        # is what pressing it again afterwards is for.
        ticket = self._claim_answer(supersede=False)
        if ticket is None:
            logger.info("Answer in progress, ignoring the screen capture")
            return
        try:
            self._emit(ticket, self.overlay.set_status, "answering")
            self._emit(ticket, self.overlay.show_question, "[Screen capture]")

            context = self.context_mgr.get_context_string()
            state = self.context_mgr.get_state()

            full_answer = ""
            meta = None
            superseded = False

            # Route screen captures to Groq vision model (much faster than OpenRouter).
            # Cap at 500 tokens — screen descriptions are brief, and some Groq
            # free-tier vision models have a 1000 OTPM limit (requesting 1024
            # would hit a 429 before a single response could complete).
            _vision_tokens = min(
                tokens_for_style(self.context_mgr.get_answer_style(),
                                 config.MAX_ANSWER_CHARS // 4),
                500
            )
            stream = query_vision_stream(
                image_b64=image_b64,
                prompt=config.SCREEN_CAPTURE_PROMPT,
                context=context,
                state=state,
                api_key=config.GROQ_LLM_API_KEY,
                model=config.GROQ_LLM_VISION_MODEL,
                base_url=config.GROQ_LLM_BASE_URL,
                max_tokens=_vision_tokens,
                mime=mime,
            )
            try:
                for chunk in stream:
                    if isinstance(chunk, dict) and "_meta" in chunk:
                        meta = chunk["_meta"]
                    else:
                        full_answer += chunk
                        # A spoken question asked during the 7-15s a vision
                        # model takes supersedes this, and it matters more.
                        if not self._emit(ticket, self.overlay.stream_answer, chunk):
                            superseded = True
                            break
            finally:
                stream.close()

            if superseded:
                logger.info("Dropped a superseded screen capture answer")
                return

            if config.SHOW_LATENCY and meta:
                total_ms = (time.time() - start_time) * 1000
                ttft = meta.get("ttft_ms", 0)
                self._emit(ticket, self.overlay.show_latency, total_ms, ttft)

            if not (meta and meta.get("error")):
                # Deliberately not recorded in the Q&A history: "[Screen
                # capture]" is not a question, and it would spend one of the
                # three prompt history slots on something the model cannot
                # interpret. Retry gets at it through _last_request instead.
                self._last_request = ("vision", image_b64, mime)

        except Exception as e:
            logger.error(f"Screen capture query failed: {e}")
            self._emit(ticket, self.overlay.stream_answer, f"\n[Error: {e}]")
        finally:
            self._emit(ticket, self.overlay.set_status, "listening")
            self._release_answer(ticket)

    def run(self):
        """Main loop: audio processing in background thread, Qt event loop on main thread."""
        self.is_running = True
        
        # Start audio listening in background thread
        listen_thread = threading.Thread(target=self._listen_loop, name="listen", daemon=True)
        listen_thread.start()

        watchdog = threading.Thread(target=self._audio_watchdog,
                                    name="audio-watchdog", daemon=True)
        watchdog.start()

        heartbeat = threading.Thread(target=self._heartbeat,
                                     name="heartbeat", daemon=True)
        heartbeat.start()
        
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

                # Anything that queued up behind a slow transcription is
                # already stale — answering it would land under a question
                # the interviewer has moved past.
                audio_chunk = self.audio.newest_pending(audio_chunk)


                # Transcribe via Groq API
                self.overlay.set_status("transcribing")
                logger.info("Transcribing via Groq Whisper API...")
                result = transcribe(
                    audio_chunk,
                    sample_rate=config.SAMPLE_RATE,
                )
                
                if result.get("error"):
                    self._report_stt_error(result["error"])
                    continue

                text = result["text"].strip()
                stt_latency = result.get("latency_ms", 0)

                # A transcription got through, so whatever was wrong is over.
                self._last_stt_error = None

                if not text or len(text) < 3:
                    logger.debug(f"Empty transcription, skipping (STT: {stt_latency:.0f}ms)")
                    self.overlay.set_status("listening")
                    continue
                
                logger.info(f"Transcribed ({stt_latency:.0f}ms): '{text[:80]}...'")
                
                if self._duplicates_recent_grab(text):
                    self.overlay.set_status("listening")
                    continue

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
        self.grab_listener.stop()
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