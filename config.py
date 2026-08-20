# config.py — Ghastly AI Configuration
# API keys loaded from environment variables / .env file
# Never hardcode keys. Never commit .env.

import sys
import os
from pathlib import Path

# PyInstaller base & executable paths
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _base_dir = Path(sys._MEIPASS)
    _exe_dir = Path(sys.executable).parent
else:
    _base_dir = Path(__file__).parent
    _exe_dir = _base_dir

# Load .env file (checks .exe folder first, then bundled _base_dir)
_env_path = _exe_dir / ".env"
if not _env_path.exists():
    _env_path = _base_dir / ".env"

if _env_path.exists():
    with open(_env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# === STT (Groq Whisper API) ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# === LLM (Ollama Cloud) ===
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-super")
# Screen captures go to Ollama too — of the free models, only gemma4 takes
# images (the nemotron family returns "does not support image input").
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "gemma4:31b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://api.ollama.com/api")

# === Vision LLM (OpenRouter) ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "openrouter/free")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# === Screen Capture ===
SCREEN_CAPTURE_HOTKEY = os.environ.get("SCREEN_CAPTURE_HOTKEY", "ctrl+shift+h")
SCREEN_CAPTURE_PROMPT = (
    "Look at the screen. If it's a coding/technical problem, explain the "
    "approach concisely; otherwise describe and answer what's shown."
)

# === Hotkeys ===
# Hides / shows the overlay instantly without quitting it.
PANIC_HOTKEY = os.environ.get("PANIC_HOTKEY", "ctrl+shift+space")

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
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 3  # seconds per audio chunk
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
SILENCE_DURATION = 1.0  # seconds of silence before processing chunk

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
MAX_CONTEXT_CHARS = 8000  # trim context if too long

# === Behavior ===
MAX_ANSWER_CHARS = 1000  # limit answer length for quick reading
SHOW_LATENCY = True  # show time-to-answer in overlay
AUTO_SCROLL = True  # auto-scroll to latest answer
KEEP_HISTORY = 3  # show last N answers in overlay