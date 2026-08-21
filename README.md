# ⚡ Ghastly AI

Real-time interview assistant with an invisible screen overlay.

**Captures system audio → transcribes with Whisper → queries an LLM → displays answers on a ghost overlay invisible to all screen sharing.**

## How it works

```
System Audio → VAD (silence detection) → Whisper STT → Question Filter → Context+State → Ollama LLM (streaming) → Ghost Overlay
                                                                                                                    ↓
                                                                                                         WDA_EXCLUDEFROMCAPTURE
                                                                                                         (invisible to capture)
```

You can also type a question into the overlay, or press the screen-capture
hotkey to send what's on screen to a vision model — both land in the same
answer panel.

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | What it does |
|---|---|
| `GROQ_API_KEY` | Groq API key (Whisper STT) |
| `GROQ_WHISPER_MODEL` | STT model (default: `whisper-large-v3`) |
| `OLLAMA_API_KEY` | Ollama cloud API key |
| `OLLAMA_MODEL` | Answer model (default: `nemotron-3-super`) |
| `OLLAMA_VISION_MODEL` | Model for screen captures (default: `gemma4:31b` — the only free Ollama model that accepts images) |
| `SCREEN_CAPTURE_HOTKEY` | Send the screen to the vision model (default: `ctrl+shift+h`) |
| `PANIC_HOTKEY` | Hide / show the overlay without quitting (default: `ctrl+shift+space`) |
| `OPACITY_HOTKEY` | Toggle opaque / translucent (default: `ctrl+shift+o`) |
| `RETRY_HOTKEY` | Answer the last question again (default: `ctrl+shift+r`) |
| `CODE_LANGUAGE` | Pins the answer language, overriding the setup panel (e.g. `Java`) |
| `ANSWER_STYLE` | Pins the answer style (`Balanced`, `Snippet only`, `Text only`, `Full walkthrough`) |
| `AUDIO_DEVICE` | Pins the capture device; otherwise picked in the setup panel |
| `MIN_UTTERANCE_SEC` | Audio shorter than this never reaches the STT API (default: `1.2`) |
| `LOG_FILE` | Log path, relative to the app (default: `logs/ghastly.log`) |

`OPENROUTER_API_KEY` / `OPENROUTER_VISION_MODEL` are only needed if you switch
screen captures back to OpenRouter — its free tier caps out at 50 requests a
day, which is why captures go through Ollama instead.

Overlay appearance (opacity, position, colors) is set in `config.py`, not `.env`.

### 3. Give it your background

Two ways, and they stack:

- **The setup panel** (📎 in the command bar) — attach a resume, a job
  description, notes. PDF, DOCX, and plain-text formats are read directly.
  Uploads take effect on the next question, no restart.
- **`context/interview-context.md`** — a hand-written profile, if you prefer.

### 4. Run

```bash
python main.py
```

The overlay appears at the top of your screen. Start your interview call; when
the interviewer asks a question, the answer appears in the panel.

## The overlay

| Control | What it does |
|---|---|
| ☀️ / 🌙 | Opaque or translucent (`Ctrl+Shift+O`) |
| ℹ️ | Lists the active hotkeys |
| 📎 | Setup panel — documents, answer language, answer style, audio source |
| ↻ | Answer the last question again (`Ctrl+Shift+R`) |
| ● | Close |
| Corner squares | Drag to resize |
| Double-click the bar | Collapse / expand the answer panel |
| Ask box | Type a question and press Enter |

The mouse cursor never changes shape anywhere over the overlay. Capture
exclusion hides the window's pixels but not the OS cursor sprite, so a cursor
that turned into a hand would give the game away.

### Answer styles

| Style | What comes back |
|---|---|
| Balanced | A sentence of reasoning, then the smallest snippet |
| Snippet only | Code, nothing else |
| Text only | Spoken explanation, no code at all |
| Full walkthrough | Full code, then the approach, then the decisions made |

## Platform Support

| Feature | Linux (dev) | Windows (deploy) |
|---|---|---|
| Audio capture (loopback) | ✅ PulseAudio monitor | ✅ WASAPI loopback |
| Whisper STT | ✅ | ✅ |
| Ollama cloud LLM | ✅ | ✅ |
| Ghost overlay (visible) | ✅ | ✅ |
| WDA_EXCLUDEFROMCAPTURE | ❌ | ✅ |
| Invisible to screen share | ❌ | ✅ |

**On Linux:** the overlay works but IS visible in a screen share. Dev only.

**On Windows:** the overlay is invisible to all screen capture (Zoom, Meet,
Teams, Slack, screenshots, recordings). So are its tooltips, dropdowns, dialogs
and file picker — each is a separate window, and each is excluded explicitly.

## Context & Memory

**Static:** `context/interview-context.md` plus anything attached in the setup
panel. Uploaded documents are extracted to `context/uploaded/` and folded in on
every load, resumes first — if the character cap bites, the notes get dropped
rather than half your work history.

**Dynamic:** `context/interview-state.json`, beside the executable in a packaged
build. Holds the Q&A history, the interviewer's mood, persona and topic, and
your language / style / device preferences, which survive closing the app. The
last `KEEP_HISTORY` Q&A pairs go into each prompt for continuity.

## Architecture

```
ghastly-ai/
├── main.py              ← Entry point + orchestration
├── config.py            ← All settings
├── audio_capture.py     ← System audio loopback + VAD
├── transcribe.py        ← Whisper STT + question detection
├── llm_query.py         ← Ollama cloud client (text + vision, streaming)
├── file_context.py      ← Resume / document uploads (PDF, DOCX, text)
├── context_manager.py   ← Context + state management
├── screen_capture.py    ← Screenshot grab + global hotkeys
├── ghost_overlay.py     ← WDA_EXCLUDEFROMCAPTURE overlay + setup panel
├── context/
│   ├── interview-context.md   ← Static context (optional, hand-written)
│   ├── interview-state.json   ← Session state + preferences (auto)
│   └── uploaded/              ← Extracted text of attached documents (auto)
├── logs/ghastly.log     ← Rotating log (the packaged build has no console)
├── requirements.txt
└── README.md
```

## Latency

Measured on `nemotron-3-super`, end of question to first token on screen:

```
Whisper STT (~0.5s) → context prep (~0.01s) → LLM TTFT (~1.0-1.6s)
Total: ~1.5-2.1s
```

Screen captures are slower — `gemma4:31b` takes roughly 7-15s to first token.
It is the trade for a vision model with no daily request cap.

## Troubleshooting

**Nothing happens when the interviewer talks.** Open the setup panel and check
the audio source. The default picks a WASAPI loopback device automatically, but
if the call's audio is routed elsewhere the pipeline never sees it. The log
records which device was chosen. Meanwhile, type the question into the ask box.

**Blank answers.** Reasoning models can spend the entire token budget on
thinking tokens, which are discarded. The panel says so when it happens; switch
models or use a shorter answer style.

**The packaged build has no console** — `logs/ghastly.log`, next to the .exe, is
the only record of what it did. Start there.

## Tips for Best Results

1. **Use headphones** — otherwise the agent hears your own voice through the speakers
2. **Attach the real resume** — the answers quote it, and specifics are what make them land
3. **Paraphrase, don't read** — answers are written to be spoken, but in your own voice
4. **Test before the interview** — play a YouTube interview and watch the pipeline run
5. **Check your quota** — the free tiers are finite; every utterance costs an STT request

## License

Private project. Not for distribution.
