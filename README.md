# ⚡ Ghost Interview Agent

Real-time interview assistant with an invisible screen overlay.

**Captures system audio → transcribes with Whisper → queries LLM → displays answers on a ghost overlay invisible to all screen sharing.**

## How it works

```
System Audio → VAD (silence detection) → Whisper STT → Question Filter → Context+State → Ollama LLM (streaming) → Ghost Overlay
                                                                                                                    ↓
                                                                                                         WDA_EXCLUDEFROMCAPTURE
                                                                                                         (invisible to capture)
```

**Latency target:** <2 seconds from end of question to answer displayed.

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Configure

Edit `config.py`:
- `OLLAMA_API_KEY` — your Ollama cloud API key
- `OLLAMA_MODEL` — model to use (default: glm-4.5)
- `WHISPER_MODEL` — STT model size (tiny/base/small/medium)
- Overlay settings (opacity, position, colors)

### 3. Set interview context

Edit `context/interview-context.md`:
- Candidate profile (name, role, skills, experience)
- Job description
- Persona (tone, answer style)
- Projects to highlight
- Weaknesses to deflect

### 4. Run

```bash
python main.py
```

The ghost overlay appears in the corner of your screen. Start your interview call. When the interviewer asks a question, the agent transcribes it and displays the answer on the overlay.

## Platform Support

| Feature | Linux (dev) | Windows (deploy) |
|---|---|---|
| Audio capture (loopback) | ✅ PulseAudio monitor | ✅ WASAPI loopback |
| Whisper STT | ✅ | ✅ |
| Ollama cloud LLM | ✅ | ✅ |
| Ghost overlay (visible) | ✅ | ✅ |
| WDA_EXCLUDEFROMCAPTURE | ❌ | ✅ |
| Invisible to screen share | ❌ | ✅ |

**On Linux:** The overlay works but IS visible in screen share. Use for dev/testing only.

**On Windows:** The overlay is invisible to ALL screen capture (Zoom, Meet, Teams, Slack, screenshots, recordings). Only your physical monitor shows it.

## Context & Memory

**Static (set before interview):** `context/interview-context.md`
- Candidate profile, skills, projects, persona, job description

**Dynamic (updated during interview):** `context/interview-state.json`
- Questions asked, answers given
- Interviewer mood (aggressive/friendly/confused/impressed/skeptical)
- Interviewer persona (technical/behavioral/casual/managerial)
- Current topic (ml/mlops/python/database/system_design/etc.)
- Last 3 Q&A pairs prepended to LLM prompt for continuity

## Architecture

```
ghost-interview-agent/
├── main.py              ← Entry point + orchestration
├── config.py            ← All settings
├── audio_capture.py     ← System audio loopback + VAD
├── transcribe.py        ← Whisper STT + question detection
├── llm_query.py         ← Ollama cloud client (streaming)
├── context_manager.py   ← Context + state management
├── ghost_overlay.py     ← WDA_EXCLUDEFROMCAPTURE overlay
├── context/
│   ├── interview-context.md   ← Static context (edit before interview)
│   └── interview-state.json   ← Dynamic state (auto-updated)
├── requirements.txt
└── README.md
```

## Latency Budget

```
Audio chunk (1.5s silence) → Whisper STT (~0.3s) → Context prep (~0.01s) → Ollama LLM TTFT (~0.5-1s) → Display (~0.01s)
Total: ~0.8-1.3s after question ends ✅ (under 2s target)
```

## Tips for Best Results

1. **Use headphones** — prevents the agent from hearing your own voice through speakers
2. **Whisper model:** `small` for accuracy, `base` for speed, `tiny` for emergencies
3. **Context file:** be specific about your projects — the LLM can't know what you didn't tell it
4. **Paraphrase, don't read** — the overlay gives you the answer, but say it in your own words
5. **Test before the interview** — play a YouTube interview video and verify the pipeline works

## License

Private project. Not for distribution.