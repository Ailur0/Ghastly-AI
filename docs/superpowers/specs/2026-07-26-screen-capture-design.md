# Screen Capture Feature — Design Spec

Date: 2026-07-26

## Context

Ghost Interview Agent currently answers questions from a single input source: system audio, transcribed via Groq Whisper, filtered for questions, and answered by a text-only Ollama LLM (`glm-4.5`) streamed into the invisible overlay.

This misses any question that depends on what's *visible* on screen — a coding problem, a shared slide, a diagram — since the pipeline has no way to see it. This feature adds a manually-triggered screen capture that sends a screenshot to a vision-capable model, independent of the audio pipeline, so the assistant can answer based on visual content too.

## Goal

Let the user press a global hotkey at any time to capture the primary monitor and get a streamed answer about what's on screen, using the same overlay UI already used for audio-driven answers.

## Out of scope

- Correlating the screenshot with a specific spoken question (rejected in favor of a standalone trigger — simpler and decouples the two pipelines).
- Multi-monitor capture (primary monitor only, per decision).
- OCR-based text extraction pipeline (rejected in favor of true vision-model image input).
- Any change to the existing audio→STT→LLM pipeline; it continues to run unmodified in parallel.
- Automated test framework (pytest, etc.) — not introduced here since the project currently uses manual/terminal test scripts.

## Architecture

### New module: `screen_capture.py`

- `ScreenCapture` — wraps the `mss` library.
  - `capture_primary_monitor() -> bytes`: grabs the primary display, returns PNG-encoded bytes.
- `HotkeyListener` — thin wrapper around the `keyboard` library.
  - Registers a global hotkey (works even when the app is unfocused/click-through, which matters since the overlay is often invisible) and invokes a callback on its own thread when pressed.

### `config.py` additions

```python
SCREEN_CAPTURE_HOTKEY = os.environ.get("SCREEN_CAPTURE_HOTKEY", "ctrl+shift+h")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "openrouter/free")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SCREEN_CAPTURE_PROMPT = (
    "Look at the screen. If it's a coding/technical problem, explain the "
    "approach concisely; otherwise describe and answer what's shown."
)
```

**Revision note (during implementation):** the original design used an Ollama Cloud vision model (`qwen3-vl:235b-cloud`) for this path. During Task 4's implementation, that tag was found to be retired (HTTP 410), and three follow-up candidate tags (`qwen3-vl:8b-cloud`, `qwen3-vl:32b-cloud`, `glm-4.6:cloud`, `gemma3:27b-cloud`) were empirically probed against the live API and *all* failed (404s and 410s) — Ollama Cloud's free/available model catalog churns too fast for a hardcoded tag to stay reliable. Switched to **OpenRouter** instead, using its `openrouter/free` router model: OpenRouter auto-selects among its currently-available free models, filtered by required capability (image input, in this case), so the app's code doesn't need to track catalog churn itself. This is the vision path's provider; the existing text-only audio pipeline (`OLLAMA_MODEL`, Ollama Cloud) is unaffected.

### `llm_query.py` additions

- `query_openrouter_vision_stream(image_b64, prompt, context, state, api_key=OPENROUTER_API_KEY, model=OPENROUTER_VISION_MODEL, base_url=OPENROUTER_BASE_URL, max_tokens)`:
  - Calls OpenRouter's OpenAI-compatible `POST {base_url}/chat/completions` endpoint (not Ollama's `/api/chat` — different provider, different wire format).
  - Message payload uses OpenAI vision format: user message `content` is a list with a text part and an `{"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}` part.
  - Response is SSE (`data: {...}\n\n` lines, terminated by `data: [DONE]`), not Ollama's NDJSON — parsed by a dedicated helper distinct from `_stream_chat` (which stays Ollama-specific, used only by the existing text path).
  - Reuses `build_prompt`'s context/state formatting so tone/history stays consistent with audio-driven answers.
  - Yields the same chunk / `_meta` dict contract as `query_ollama_stream`, so callers (the overlay wiring in `main.py`) don't need to know which provider is behind it.

### `main.py` wiring

- `GhostInterviewAgent.__init__` constructs a `ScreenCapture` and a `HotkeyListener`.
- `initialize()` registers the hotkey via `HotkeyListener`, pointed at `on_screen_capture_hotkey()`. Registration failure (e.g. restricted environment) is caught, logged as a warning, and the app continues audio-only — not a startup crash.
- `on_screen_capture_hotkey()` runs in its own thread (mirrors the existing `process_question` threading pattern) so it never blocks or is blocked by the audio `_listen_loop`:
  1. Debounce: ignore if a screen-capture query is already in flight.
  2. `overlay.set_status("answering")`; `overlay.show_question("[Screen capture]")`.
  3. Capture → base64-encode. On failure: log, `overlay.stream_answer("[Error: screen capture failed]")`, return.
  4. `query_openrouter_vision_stream(...)`, streaming chunks into `overlay.stream_answer(chunk)` exactly like `process_question` does for audio answers.
  5. `context_mgr.add_qa("[Screen capture]", full_answer)` so it folds into subsequent Q&A history/context.
  6. `overlay.set_status("listening")`.

### `ghost_overlay.py`

No structural changes. The vision answer reuses the existing `stream_answer` / `set_status` signal path, so it renders in the same answer panel as audio-driven answers.

## Data flow

```
Hotkey press (global, works while unfocused/click-through)
  → HotkeyListener callback (own thread)
  → ScreenCapture.capture_primary_monitor() → PNG bytes → base64
  → on_screen_capture_hotkey() worker thread (mirrors process_question)
      → overlay: status "answering", show_question("[Screen capture]")
      → query_openrouter_vision_stream(image_b64, SCREEN_CAPTURE_PROMPT, context, state, model=OPENROUTER_VISION_MODEL)
      → streamed chunks → overlay.stream_answer(chunk)
      → context_mgr.add_qa("[Screen capture]", full_answer)
      → overlay: status "listening"
```

Runs independently of the audio `_listen_loop` thread — a hotkey press mid-question neither interrupts nor is interrupted by the audio pipeline.

## Error handling

| Failure | Handling |
|---|---|
| Screenshot capture fails (display access, driver edge case) | Caught in `ScreenCapture`, logged, `overlay.stream_answer("[Error: screen capture failed]")`, return early — never send a broken/empty image to the API. |
| Hotkey registration fails (e.g. elevated-privilege environment) | Caught at `initialize()`, logged as a warning (same pattern as missing API keys), app continues audio-only. |
| Vision API error (bad model tag, timeout, non-200, malformed SSE) | Same pattern as `query_ollama_stream`: yield an `[Error: ...]` string chunk plus a `_meta` dict with `error` set, shown inline in the overlay like existing audio-path errors. |
| Hotkey pressed while a capture query is already in flight | Debounced: new press ignored (logged at debug level), no queuing/overlapping. |

## Testing

- Manual verification via a small `test_screen_capture.py`, following the existing standalone/no-overlay style of `test_pipeline.py`: press the hotkey, confirm a screenshot is captured, base64-encoded, sent to `OPENROUTER_VISION_MODEL` via OpenRouter, and a streamed answer prints to the terminal.
- No automated test framework is introduced, matching the project's current manual/terminal-script testing convention.

## New dependencies

Added to `requirements.txt`:
- `mss` — screenshot capture.
- `keyboard` — global hotkey registration.

(No new HTTP client dependency needed — OpenRouter's REST API is called with the project's existing `requests` library, same as Groq and Ollama.)
