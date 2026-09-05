# config.py — Ghastly AI Configuration
# API keys loaded from environment variables / .env file
# Never hardcode keys. Never commit .env.

import sys
import os
import logging
from pathlib import Path

import file_context

logger = logging.getLogger(__name__)

# PyInstaller base & executable paths
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _base_dir = Path(sys._MEIPASS)
    _exe_dir = Path(sys.executable).parent
    # Settings changed from the setup panel have to land somewhere that
    # outlives the process. sys._MEIPASS is a temp folder Windows deletes on
    # exit, and the .exe's own folder may be unwritable — so share the
    # fallback chain the uploads and the saved state already use.
    _env_write_path = file_context.writable_base() / ".env"
else:
    _base_dir = Path(__file__).parent
    _exe_dir = _base_dir
    _env_write_path = _base_dir / ".env"


def _env_files():
    """
    Every .env worth reading, highest precedence first.

    Keys load with setdefault, so the first file to mention one wins: a
    setting changed in the setup panel beats a .env shipped beside the .exe,
    which beats the copy baked into the bundle. That layering is what lets
    the writable file hold nothing but the handful of changed settings
    while the API keys keep coming from the shipped file.
    """
    seen = set()
    for path in (_env_write_path, _exe_dir / ".env", _base_dir / ".env"):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            yield path


for _env_file in _env_files():
    with open(_env_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def update_env_file(key: str, value: str) -> bool:
    """
    Persist one setting to the writable .env, preserving comments and
    structure. Returns True if it actually stuck.

    Never raises: a settings file the app cannot write is worth a warning and
    a change that lasts the session, not a crash inside the setup panel.
    """
    try:
        _env_write_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        if _env_write_path.exists():
            with open(_env_write_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("#") and "=" in line:
                k, _ = line.split("=", 1)
                if k.strip() == key:
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break

        if not updated:
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append(f"{key}={value}\n")

        with open(_env_write_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        logger.warning(f"Could not save {key} to {_env_write_path}: {e}")
        return False


# === STT (Groq Whisper API) ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# === LLM (Groq Cloud) ===
# Was Ollama Cloud / nemotron until v2026-09-04; `git log -- config.py` has
# the old settings if that ever needs undoing.
GROQ_LLM_API_KEY = os.environ.get("GROQ_LLM_API_KEY", "")
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
# Screen captures need a model that takes images. The llama-3.2-*-vision-preview
# models this used to point at have been decommissioned by Groq, and no llama-4
# vision model is offered on this key — qwen3.8-27b is what's actually available
# and accepts image_url blocks. Check /v1/models before changing this.
GROQ_LLM_VISION_MODEL = os.environ.get("GROQ_LLM_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_LLM_BASE_URL = os.environ.get("GROQ_LLM_BASE_URL", "https://api.groq.com/openai/v1")
# Aliases so existing code referencing OLLAMA_* keeps working without changes.
OLLAMA_API_KEY = GROQ_LLM_API_KEY
OLLAMA_MODEL = GROQ_LLM_MODEL
OLLAMA_VISION_MODEL = GROQ_LLM_VISION_MODEL
OLLAMA_BASE_URL = GROQ_LLM_BASE_URL

# === Screen Capture ===
SCREEN_CAPTURE_HOTKEY = os.environ.get("SCREEN_CAPTURE_HOTKEY", "ctrl+shift+h")
SCREEN_CAPTURE_PROMPT = (
    "Look at the screen. If it's a coding/technical problem, explain the "
    "approach concisely; otherwise describe and answer what's shown."
)

# === Identity ===
# The OS-level window title, which anything enumerating windows can read —
# Task Manager, proctoring tools. Kept in step with the executable name; the
# name shown inside the overlay is separate and stays "Ghastly AI".
WINDOW_TITLE = os.environ.get("WINDOW_TITLE", "System Audio Helper")
# Escape hatch: the file picker uses Qt's own widget so it can be hidden from
# screen capture. Set NATIVE_FILE_DIALOG=1 if Qt's picker misbehaves on a
# machine — the native Explorer dialog is NOT hidden from capture.
NATIVE_FILE_DIALOG = os.environ.get("NATIVE_FILE_DIALOG", "0").lower() in ("1", "true", "yes")
# A second copy would fight the first for the audio device and the hotkeys.
SINGLE_INSTANCE = os.environ.get("SINGLE_INSTANCE", "1").lower() not in ("0", "false", "no")

# === Hotkeys ===
# Hides / shows the overlay instantly without quitting it.
PANIC_HOTKEY = os.environ.get("PANIC_HOTKEY", "ctrl+shift+space")
# Opaque <-> translucent, without reaching for the sun/moon button.
OPACITY_HOTKEY = os.environ.get("OPACITY_HOTKEY", "ctrl+shift+o")
# Answer the last question again — the hotkey twin of the retry button.
RETRY_HOTKEY = os.environ.get("RETRY_HOTKEY", "ctrl+shift+r")
# Answer what was just said: transcribe the last GRAB_SECONDS of audio and
# answer it, no matter what VAD or the question filter made of it. Press it
# after the question has been asked — there is nothing to hold and nothing
# to anticipate, because the audio is already buffered.
#
# Two keys, not three, and both under the right hand: this is the only
# hotkey pressed mid-question, so it has to be hittable without looking
# down. Semicolon is unbound in VS Code, CoderPad, Chrome and the call
# apps, and Ctrl avoids Alt's menu accelerators.
GRAB_HOTKEY = os.environ.get("GRAB_HOTKEY", "ctrl+;")

# === Logging ===
# A windowed build has nowhere to print, so the log is the only way to see
# what happened. Lands next to the .exe.
LOG_FILE = os.environ.get("LOG_FILE", "logs/ghastly.log")
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 2

# === Audio Capture ===
# "Auto" lets the capture layer pick; otherwise a device id from
# AudioCapture.list_input_devices().
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "Auto")
# Anything shorter than this is a grunt, not a question. Each utterance costs
# one Groq transcription request, so the floor is money as well as latency.
MIN_UTTERANCE_SEC = float(os.environ.get("MIN_UTTERANCE_SEC", "1.2"))
# And a ceiling. The gate only closes on silence, so continuous sound — music,
# room tone, a fan — kept one "utterance" growing without limit until the room
# finally went quiet, then posted all of it in a single request.
MAX_UTTERANCE_SEC = float(os.environ.get("MAX_UTTERANCE_SEC", "30"))
# Capture can die quietly: plugging in headphones mid-call switches the
# default device and the old recorder just returns silence forever; or the
# capture thread throws and returns, and nothing fills the queue again. The
# watchdog checks for both and re-opens.
#
# There is no stall timeout any more: WASAPI loopback keeps delivering frames
# of zeros when nothing is playing, so a quiet room and a dead device produced
# identical frame timestamps and the check could never fire.
AUDIO_WATCHDOG_SEC = float(os.environ.get("AUDIO_WATCHDOG_SEC", "15"))
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 3  # seconds per audio chunk
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
SILENCE_DURATION = 1.0  # seconds of silence before processing chunk
# How far back the grab hotkey reaches. Long enough for a question asked
# slowly with a pause in the middle, short enough not to drag in the answer
# to the previous one.
GRAB_SECONDS = float(os.environ.get("GRAB_SECONDS", "20"))
# The rolling window the grab slices from. Must exceed GRAB_SECONDS, with
# room for the second or two between the question ending and you pressing.
AUDIO_RING_SEC = float(os.environ.get("AUDIO_RING_SEC", "45"))

# === Ghost Overlay (Cluely Design System) ===
OVERLAY_OPACITY_OPAQUE = 1.0        # 0.0-1.0 (Qt window opacity scale)
OVERLAY_OPACITY_TRANSLUCENT = 0.5   # 0.0-1.0
OVERLAY_BAR_WIDTH = 420       # command bar width (collapsed)
OVERLAY_BAR_HEIGHT = 38       # command bar height
OVERLAY_PANEL_WIDTH = 480     # answer panel width (expanded)
OVERLAY_PANEL_HEIGHT = 380    # answer panel max height
OVERLAY_POSITION = "top-center"

# Cluely Color Palette (High-Contrast Clean Glass)
OVERLAY_GLASS_BG = "rgba(255, 255, 255, 0.94)"
OVERLAY_GLASS_BORDER = "rgba(255, 255, 255, 0.50)"
OVERLAY_ACCENT = "#0284C7"           # deep sky-blue
OVERLAY_ACCENT_HOVER = "#0369A1"
OVERLAY_ACCENT_GLOW = "rgba(14, 165, 233, 0.35)"
OVERLAY_TEXT_PRIMARY = "#020617"      # slate-950 (high contrast dark)
OVERLAY_TEXT_SECONDARY = "#334155"    # slate-700 (dark readable gray)
OVERLAY_TEXT_ANSWER = "#020617"       # slate-950 (high contrast text)
OVERLAY_SUCCESS = "#16A34A"
OVERLAY_ERROR = "#DC2626"
OVERLAY_SHADOW = "0 8px 32px rgba(0, 0, 0, 0.18)"

# Typography (Increased sizes for readability)
OVERLAY_FONT_FAMILY = "Segoe UI"      # fallback: Inter, system-ui
OVERLAY_FONT_SIZE = 16
OVERLAY_FONT_SIZE_SMALL = 14
OVERLAY_FONT_SIZE_META = 12

# === Uploaded documents & answer language ===
# Resumes and notes added from the overlay's setup panel land here (next to
# the .exe in a frozen build) and are folded into the context on every load.
UPLOADS_DIR = "context/uploaded"
# "Auto" = let the model follow whatever the question implies.
CODE_LANGUAGES = [
    "Auto", "Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
    "Go", "Rust", "SQL", "Ruby", "PHP", "Swift", "Kotlin",
]
_CODE_LANGUAGE_ENV = os.environ.get("CODE_LANGUAGE")
DEFAULT_CODE_LANGUAGE = _CODE_LANGUAGE_ENV or "Auto"
# Set in .env, this pins the language every launch and outranks whatever
# was last picked in the setup panel. Unset, the saved choice wins.
CODE_LANGUAGE_PINNED = bool(_CODE_LANGUAGE_ENV)

# How much of an answer is code vs. spoken explanation.
#   Balanced        — a sentence of reasoning plus the smallest snippet
#   Snippet only    — code, nothing else
#   Text only       — spoken explanation, no code at all
#   Full walkthrough— complete code, then approach, then decisions made
ANSWER_STYLES = ["Balanced", "Snippet only", "Text only", "Full walkthrough"]
_ANSWER_STYLE_ENV = os.environ.get("ANSWER_STYLE")
DEFAULT_ANSWER_STYLE = _ANSWER_STYLE_ENV or "Balanced"
ANSWER_STYLE_PINNED = bool(_ANSWER_STYLE_ENV)

# === Context ===
CONTEXT_FILE = "context/interview-context.md"
# How much of your documents reaches the model. gpt-oss-120b has a 128k
# window, so 8000 chars (~2k tokens) was leaving almost all of it unused —
# and it was the reason one long resume truncated and every other document
# was silently dropped. Raise further if you upload a lot; it costs TTFT.
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "16000"))

# === Behavior ===
# 0.85 was high for an assistant answering technical questions under time
# pressure — it buys spoken-sounding variety at the cost of steadiness on
# facts and code. Settable so it can be tuned against real interviews rather
# than guessed at: raise it if answers start sounding canned.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
MAX_ANSWER_CHARS = 1000  # limit answer length for quick reading
SHOW_LATENCY = True  # show time-to-answer in overlay
AUTO_SCROLL = True  # auto-scroll to latest answer
KEEP_HISTORY = 3  # previous Q&A pairs carried into each prompt