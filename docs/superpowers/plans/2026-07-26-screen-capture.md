# Screen Capture Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user press a global hotkey to capture the primary monitor and get a streamed answer about what's on screen, using a vision-capable Ollama Cloud model and the existing overlay UI.

**Architecture:** A new `screen_capture.py` module handles screenshotting (`mss`) and global hotkey registration (`keyboard`), independent of the existing audio pipeline. `llm_query.py` gains a vision-capable sibling to its existing streaming query function, sharing its NDJSON-parsing logic via a new internal helper. `main.py` wires the hotkey to a worker thread that mirrors the existing `process_question` pattern, streaming into the same overlay.

**Tech Stack:** Python 3.12, `mss` 10.2.0 (screenshot), `keyboard` 0.13.5 (global hotkey), existing PyQt5 overlay, Ollama Cloud `/api/chat` vision endpoint.

## Global Constraints

- Capture the primary monitor only (not all-monitors, not a selectable region).
- Trigger is a global hotkey; default `ctrl+shift+h`, configurable via `SCREEN_CAPTURE_HOTKEY` env var.
- Standalone trigger: the hotkey immediately captures + queries with a fixed generic prompt — it does NOT wait for or correlate with a spoken question.
- Vision provider: OpenRouter (not Ollama Cloud — the original Ollama vision model choice was found to be retired mid-implementation, see Task 4's revision note). Default model `openrouter/free` (OpenRouter's capability-filtered free-model router), configurable via `OPENROUTER_VISION_MODEL` env var. Separate from the existing text-only `OLLAMA_MODEL` (`glm-4.5`), which is unaffected and stays on Ollama Cloud.
- New dependencies (`mss==10.2.0`, `keyboard==0.13.5`) are exact-pinned in `requirements.txt`, matching the project's existing pinning style.
- No pytest or other test framework is introduced — this project's convention is manual, terminal-driven test scripts (see `test_pipeline.py`, and the `__main__` blocks in `audio_capture.py` / `llm_query.py`). All "tests" below follow that same convention: run a script, observe printed output.
- Screenshot capture failure, hotkey registration failure, and vision API errors must never crash the app — audio-only operation must continue to work in all cases.
- Concurrent hotkey presses must be debounced (ignore a press while a previous screen-capture query is still in flight).

---

### Task 1: Add new dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-lock.txt`

**Interfaces:**
- Produces: `mss` and `keyboard` importable in the venv for later tasks.

- [ ] **Step 1: Add pinned entries to `requirements.txt`**

Add these lines after the existing `requests==2.34.2` line (under a new comment section):

```
# Screen capture feature
mss==10.2.0
keyboard==0.13.5
```

- [ ] **Step 2: Install into the existing venv**

Run:
```bash
venv/Scripts/python.exe -m pip install mss==10.2.0 keyboard==0.13.5
```
Expected: both install successfully with no dependency conflicts.

- [ ] **Step 3: Regenerate the lockfile**

Run:
```bash
venv/Scripts/python.exe -m pip freeze > requirements-lock.txt
```
Then re-add the header comment block that was at the top of the old `requirements-lock.txt` (it gets stripped by `pip freeze`):
```
# Locked dependency versions (full, incl. transitive) for reproducible installs.
# Generated via: venv\Scripts\python.exe -m pip freeze
# Install with: pip install -r requirements-lock.txt
# For the human-readable top-level list, see requirements.txt.

```
(blank line, then the `pip freeze` output below it)

- [ ] **Step 4: Verify import**

Run:
```bash
venv/Scripts/python.exe -c "import mss, keyboard; print('ok')"
```
Expected output: `ok`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-lock.txt
git commit -m "Add mss and keyboard dependencies for screen capture feature"
```

---

### Task 2: Add config constants

> **Superseded by Task 4 rework:** this task shipped as written below (commit
> `9f19d7f`), adding `OLLAMA_VISION_MODEL`. During Task 4's implementation,
> that Ollama Cloud vision path was found to be unworkable (the model tag was
> retired mid-implementation, and follow-up candidate tags all failed too —
> see Task 4's revision note). The project switched to OpenRouter for the
> vision path; Task 4's rework replaces `OLLAMA_VISION_MODEL` with
> `OPENROUTER_API_KEY` / `OPENROUTER_VISION_MODEL` / `OPENROUTER_BASE_URL` in
> `config.py`. This section is kept for history; treat Task 4's version as
> current.

**Files:**
- Modify: `config.py`

**Interfaces:**
- Produces: `config.SCREEN_CAPTURE_HOTKEY: str`, `config.OLLAMA_VISION_MODEL: str`, `config.SCREEN_CAPTURE_PROMPT: str`

- [ ] **Step 1: Add the new constants**

In `config.py`, immediately after the existing `OLLAMA_BASE_URL` line (currently line 40):

```python
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:235b-cloud")

# === Screen Capture ===
SCREEN_CAPTURE_HOTKEY = os.environ.get("SCREEN_CAPTURE_HOTKEY", "ctrl+shift+h")
SCREEN_CAPTURE_PROMPT = (
    "Look at the screen. If it's a coding/technical problem, explain the "
    "approach concisely; otherwise describe and answer what's shown."
)
```

- [ ] **Step 2: Verify it loads**

Run:
```bash
venv/Scripts/python.exe -c "import config; print(config.OLLAMA_VISION_MODEL, config.SCREEN_CAPTURE_HOTKEY)"
```
Expected output: `qwen3-vl:235b-cloud ctrl+shift+h`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Add config constants for screen capture feature"
```

---

### Task 3: Implement screen capture module

**Files:**
- Create: `screen_capture.py`

**Interfaces:**
- Consumes: `mss.mss()`, `mss.tools.to_png()` (from Task 1); `keyboard.add_hotkey()`, `keyboard.remove_hotkey()` (from Task 1)
- Produces:
  - `ScreenCapture.capture_primary_monitor() -> bytes` — PNG-encoded screenshot of the primary monitor
  - `HotkeyListener(hotkey: str, callback: Callable[[], None])` with `.start() -> bool` and `.stop() -> None`

- [ ] **Step 1: Write the module**

Create `screen_capture.py`:

```python
"""
screen_capture.py — Primary-monitor screenshot capture + global hotkey registration

Used by the screen capture feature: user presses a global hotkey, we grab
the primary monitor as a PNG, base64-encode it (in main.py), and send it
to a vision-capable Ollama model.
"""

import logging
from typing import Callable

import mss
import mss.tools
import keyboard

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captures the primary monitor as PNG bytes."""

    def capture_primary_monitor(self) -> bytes:
        with mss.mss() as sct:
            # monitors[0] is the all-monitors bounding box; monitors[1] is the primary monitor
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            return mss.tools.to_png(sct_img.rgb, sct_img.size)


class HotkeyListener:
    """Registers a global hotkey that invokes a callback when pressed."""

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        self.hotkey = hotkey
        self.callback = callback
        self._registered = False

    def start(self) -> bool:
        """Register the hotkey. Returns True on success, False on failure."""
        try:
            keyboard.add_hotkey(self.hotkey, self.callback)
            self._registered = True
            logger.info(f"Hotkey registered: {self.hotkey}")
            return True
        except Exception as e:
            logger.warning(f"Hotkey registration failed for '{self.hotkey}': {e}")
            return False

    def stop(self):
        """Unregister the hotkey if it was registered."""
        if self._registered:
            try:
                keyboard.remove_hotkey(self.hotkey)
            except Exception:
                pass
            self._registered = False


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    cap = ScreenCapture()
    png_bytes = cap.capture_primary_monitor()
    print(f"Captured {len(png_bytes)} bytes")

    with open("screen_capture_test.png", "wb") as f:
        f.write(png_bytes)
    print("Saved to screen_capture_test.png — open it to confirm it's a valid screenshot")

    def on_hotkey():
        print("Hotkey fired!")

    listener = HotkeyListener("ctrl+shift+h", on_hotkey)
    if listener.start():
        print("Press ctrl+shift+h within the next 10 seconds...")
        time.sleep(10)
        listener.stop()
    else:
        print("Hotkey registration failed")
```

- [ ] **Step 2: Run the manual verification script**

Run:
```bash
venv/Scripts/python.exe screen_capture.py
```
Expected: prints `Captured N bytes` with N > 0, saves `screen_capture_test.png`. Opening that file shows a real screenshot of your primary monitor. If you press `ctrl+shift+h` within 10 seconds, it prints `Hotkey fired!`.

- [ ] **Step 3: Delete the test artifact**

```bash
rm screen_capture_test.png
```

- [ ] **Step 4: Commit**

```bash
git add screen_capture.py
git commit -m "Add ScreenCapture and HotkeyListener for screen capture feature"
```

---

### Task 4: Add vision-capable OpenRouter query function

> **Revised during implementation:** the original version of this task (below,
> before this revision) targeted an Ollama Cloud vision model
> (`OLLAMA_VISION_MODEL`, default `qwen3-vl:235b-cloud`). While implementing
> it, that tag was found to be retired (HTTP 410), and three follow-up
> candidates (`qwen3-vl:8b-cloud`, `qwen3-vl:32b-cloud`, `glm-4.6:cloud`,
> `gemma3:27b-cloud`) were probed live against the API and *all* failed
> (404s/410s) — Ollama Cloud's free-tier catalog churns too fast for a
> hardcoded tag. Switched to **OpenRouter** instead, using its `openrouter/free`
> router model, which auto-selects among currently-available free models
> filtered by required capability (image input here) — so the app's code
> doesn't need to track catalog churn. The `_stream_chat` extraction for the
> existing Ollama text path (already implemented in commit `8aa2e46`) is
> unaffected and stays as-is; only the vision function changes provider.

**Files:**
- Modify: `config.py` (replace `OLLAMA_VISION_MODEL` with OpenRouter config)
- Modify: `llm_query.py`

**Interfaces:**
- Produces (`config.py`): `config.OPENROUTER_API_KEY: str`, `config.OPENROUTER_VISION_MODEL: str`, `config.OPENROUTER_BASE_URL: str`
- Produces (`llm_query.py`): `query_openrouter_vision_stream(image_b64: str, prompt: str, context: str, state: dict, api_key: str, model: str, base_url: str, max_tokens: int) -> Generator` — same chunk/`_meta` yielding contract as `query_ollama_stream`
- Consumes: existing `build_prompt()`, `SYSTEM_PROMPT` (already in `llm_query.py`). Does NOT reuse `_stream_chat` — OpenRouter's SSE format differs from Ollama's NDJSON, so this function parses its own response stream.
- Removes: `query_ollama_vision_stream` (the retired-model version from the superseded Task 2/4 text) and `OLLAMA_VISION_MODEL` (from `config.py`) — replaced by the above.

- [ ] **Step 1: Replace the vision config constants in `config.py`**

Find and remove this line (added by the superseded Task 2):
```python
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:235b-cloud")
```

Replace it with:
```python
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "openrouter/free")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
```

(Leave `SCREEN_CAPTURE_HOTKEY` and `SCREEN_CAPTURE_PROMPT`, both already in `config.py`, untouched.)

- [ ] **Step 2: Verify config loads**

Run:
```bash
venv/Scripts/python.exe -c "import config; print(config.OPENROUTER_VISION_MODEL, config.OPENROUTER_BASE_URL)"
```
Expected output: `openrouter/free https://openrouter.ai/api/v1`

- [ ] **Step 3: Replace `query_ollama_vision_stream` with `query_openrouter_vision_stream`**

In `llm_query.py`: update the top-level config import to add the new OpenRouter constants:
```python
from config import (
    OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_VISION_MODEL, OPENROUTER_BASE_URL,
)
```

Remove the existing `query_ollama_vision_stream` function entirely (it called Ollama's `/api/chat` with an `"images"` field — that endpoint/format doesn't apply to OpenRouter). Replace it with:

```python
def query_openrouter_vision_stream(
    image_b64: str,
    prompt: str,
    context: str,
    state: dict,
    api_key: str = OPENROUTER_API_KEY,
    model: str = OPENROUTER_VISION_MODEL,
    base_url: str = OPENROUTER_BASE_URL,
    max_tokens: int = 250
) -> Generator:
    """
    Stream response from OpenRouter's OpenAI-compatible /chat/completions
    endpoint with an image attached.

    OpenRouter stream format (SSE):
        data: {"choices":[{"delta":{"content":"Hello"}}]}
        data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
        data: [DONE]

    This does NOT reuse _stream_chat (that helper is Ollama-NDJSON-specific);
    OpenRouter's SSE framing and "choices[0].delta.content" shape differ from
    Ollama's "message.content" NDJSON lines.

    Used by the screen capture feature: `prompt` is the fixed generic
    SCREEN_CAPTURE_PROMPT (not a transcribed question), `image_b64` is a
    base64-encoded PNG screenshot.
    """
    full_prompt = build_prompt(prompt, context, state)

    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhostInterviewAgent/1.0",
    }

    start_time = time.time()
    first_token_time = None
    total_text = ""
    token_count = 0

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)

        if response.status_code != 200:
            error_msg = f"OpenRouter API error {response.status_code}: {response.text[:200]}"
            logger.error(error_msg)
            yield error_msg
            yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                             "full_text": "", "error": error_msg}}
            return

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue

            data_str = decoded[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft_ms = (first_token_time - start_time) * 1000
                    logger.info(f"TTFT: {ttft_ms:.0f}ms")

                total_text += content
                token_count += 1
                yield content

        total_ms = (time.time() - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0

        logger.info(f"Vision LLM: {token_count} chunks, {total_ms:.0f}ms total, {ttft_ms:.0f}ms TTFT")

        yield {
            "_meta": {
                "total_ms": total_ms,
                "ttft_ms": ttft_ms,
                "token_count": token_count,
                "full_text": total_text
            }
        }

    except requests.exceptions.Timeout:
        logger.error("OpenRouter API timeout")
        yield "[Error: OpenRouter API timeout]"
        yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                         "full_text": "", "error": "timeout"}}
    except Exception as e:
        logger.error(f"Vision LLM error: {e}")
        yield f"[Error: {e}]"
        yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                         "full_text": "", "error": str(e)}}
```

- [ ] **Step 4: Verify the existing text path still works (regression check)**

Run:
```bash
venv/Scripts/python.exe llm_query.py
```
Expected: unaffected by this task — still streams an answer about SQL vs NoSQL via `query_ollama_stream` → `_stream_chat`, ending with `--- Total: ...ms | TTFT: ...ms ---`. This function isn't touched by this task, so this is a pure regression check.

- [ ] **Step 5: Replace the manual vision-path check in the `__main__` block**

Find the vision test block added by the superseded version of this task (it references `query_ollama_vision_stream` and builds the image via inline `mss` calls). Replace it with:

```python

    print("\n\n=== OpenRouter Vision Test ===\n")

    import base64
    from screen_capture import ScreenCapture

    png_bytes = ScreenCapture().capture_primary_monitor()
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    from config import SCREEN_CAPTURE_PROMPT

    for chunk in query_openrouter_vision_stream(
        image_b64=image_b64,
        prompt=SCREEN_CAPTURE_PROMPT,
        context=test_context,
        state=test_state,
    ):
        if isinstance(chunk, dict) and "_meta" in chunk:
            meta = chunk["_meta"]
            print(f"\n\n--- Total: {meta['total_ms']:.0f}ms | TTFT: {meta['ttft_ms']:.0f}ms ---")
        else:
            print(chunk, end="", flush=True)
```

(This now reuses `ScreenCapture` from Task 3 instead of duplicating the raw `mss` calls — Task 3 is already implemented, so this removes the duplication the earlier draft had.)

- [ ] **Step 6: Run it again and check the vision output**

Run:
```bash
venv/Scripts/python.exe llm_query.py
```
Expected: after the existing text-model output, a second section prints a streamed description of whatever is currently on your primary monitor, ending with a `--- Total: ...ms | TTFT: ...ms ---` line. If it instead prints an `[Error: ...]` line, check `config.OPENROUTER_API_KEY` is set in `.env` and that OpenRouter's status page shows no outage.

- [ ] **Step 7: Commit**

```bash
git add config.py llm_query.py
git commit -m "Switch vision query from retired Ollama model to OpenRouter"
```

---

### Task 5: Wire screen capture into main.py

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `ScreenCapture`, `HotkeyListener` (Task 3); `query_openrouter_vision_stream` (Task 4); `config.SCREEN_CAPTURE_HOTKEY`, `config.OPENROUTER_API_KEY`, `config.OPENROUTER_VISION_MODEL`, `config.OPENROUTER_BASE_URL`, `config.SCREEN_CAPTURE_PROMPT` (Tasks 2/4); existing `self.overlay.set_status/show_question/stream_answer/show_latency`, `self.context_mgr.get_context_string/get_state/add_qa`

- [ ] **Step 1: Add imports**

At the top of `main.py`, after `from ghost_overlay import GhostOverlay` (line 27):

```python
import base64
from screen_capture import ScreenCapture, HotkeyListener
```

Change the existing import line:
```python
from llm_query import query_ollama_stream
```
to:
```python
from llm_query import query_ollama_stream, query_openrouter_vision_stream
```

- [ ] **Step 2: Add screen capture state to `__init__`**

In `GhostInterviewAgent.__init__` (after the `self.overlay = GhostOverlay(...)` block, before `self.is_running = False`):

```python
        self.screen_capture = ScreenCapture()
        self.hotkey_listener = HotkeyListener(
            config.SCREEN_CAPTURE_HOTKEY,
            self.on_screen_capture_hotkey
        )
        self._screen_capture_lock = threading.Lock()
```

- [ ] **Step 3: Register the hotkey in `initialize()`**

In `initialize()`, after the audio device listing block (after `self.audio.list_devices()`), before the Groq API key check:

```python
        # Register screen capture hotkey
        logger.info("Registering screen capture hotkey...")
        if self.hotkey_listener.start():
            logger.info(f"Screen capture hotkey registered: {config.SCREEN_CAPTURE_HOTKEY}")
        else:
            logger.warning(
                f"Screen capture hotkey registration failed — "
                f"screen capture disabled, audio pipeline unaffected"
            )
```

- [ ] **Step 4: Add the hotkey callback and worker method**

Add these two new methods to `GhostInterviewAgent`, right after `process_question` (after line 152, before `def run(self):`):

```python
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
            self.overlay.set_status("error")
            return
        finally:
            self._screen_capture_lock.release()

        self.overlay.set_status("listening")
```

- [ ] **Step 5: Stop the hotkey listener on shutdown**

In `stop()`, add `self.hotkey_listener.stop()` after `self.audio.stop()`:

```python
    def stop(self):
        """Stop all components."""
        logger.info("Stopping Ghost Interview Agent...")
        self.is_running = False

        self.audio.stop()
        self.hotkey_listener.stop()
        self.overlay.stop()

        logger.info("Ghost Interview Agent stopped")
```

- [ ] **Step 6: Manual end-to-end run**

Run:
```bash
venv/Scripts/python.exe main.py
```
Expected in the log output: `Registering screen capture hotkey...` followed by `Screen capture hotkey registered: ctrl+shift+h`, then the normal `Ghost Interview Agent is LISTENING` message. Press `ctrl+shift+h` — the overlay should show `[Screen capture]` as the question and stream an answer about your screen, then return to "listening" status. Press it again immediately while an answer is still streaming — the second press should be silently ignored (check for the debug log `Screen capture already in progress...` if `logging.basicConfig` level is lowered to DEBUG).

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "Wire screen capture hotkey into main agent loop"
```

---

### Task 6: Add standalone manual test script

**Files:**
- Create: `test_screen_capture.py`

**Interfaces:**
- Consumes: `ScreenCapture`, `HotkeyListener` (Task 3), `query_openrouter_vision_stream` (Task 4), `ContextManager` (existing, unchanged)

- [ ] **Step 1: Write the test script**

Create `test_screen_capture.py`:

```python
"""
test_screen_capture.py — End-to-end manual test for the screen capture feature (no overlay)

Press the configured hotkey → capture screen → query vision LLM → stream answer to terminal.
"""

import sys
import os
import time
import base64
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from screen_capture import ScreenCapture, HotkeyListener
from llm_query import query_openrouter_vision_stream
from context_manager import ContextManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test-screen-capture")


def run_capture_and_query(screen_capture: ScreenCapture, cm: ContextManager):
    print(f"\n{'='*60}")
    print("Capturing screen...")
    png_bytes = screen_capture.capture_primary_monitor()
    print(f"Captured {len(png_bytes)} bytes")

    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    print(f"Querying {config.OPENROUTER_VISION_MODEL}...")
    print(f"{'='*60}\n")

    t0 = time.time()
    full_answer = ""
    meta = None
    for chunk in query_openrouter_vision_stream(
        image_b64=image_b64,
        prompt=config.SCREEN_CAPTURE_PROMPT,
        context=cm.get_context_string(),
        state=cm.get_state(),
        api_key=config.OPENROUTER_API_KEY,
        model=config.OPENROUTER_VISION_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
    ):
        if isinstance(chunk, dict) and "_meta" in chunk:
            meta = chunk["_meta"]
        else:
            print(chunk, end="", flush=True)
            full_answer += chunk

    total_ms = (time.time() - t0) * 1000
    ttft = meta.get("ttft_ms", 0) if meta else 0
    print(f"\n\n⏱ TOTAL: {total_ms:.0f}ms | TTFT: {ttft:.0f}ms")

    cm.add_qa("[Screen capture]", full_answer)


def main():
    print("=" * 60)
    print("  Ghost Interview Agent — Screen Capture Test")
    print("=" * 60)

    cm = ContextManager()
    cm.load_context()
    cm.reset_state()

    screen_capture = ScreenCapture()

    print(f"\nPress {config.SCREEN_CAPTURE_HOTKEY} to capture the screen and query the vision model.")
    print("Press Ctrl+C to stop.\n")

    hotkey = HotkeyListener(
        config.SCREEN_CAPTURE_HOTKEY,
        lambda: run_capture_and_query(screen_capture, cm)
    )
    if not hotkey.start():
        print("Failed to register hotkey. Exiting.")
        return

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        hotkey.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and exercise the hotkey**

Run:
```bash
venv/Scripts/python.exe test_screen_capture.py
```
Expected: prints the header, then waits. Press `ctrl+shift+h` — it should print "Capturing screen...", a byte count, "Querying openrouter/free...", then a streamed description/answer about your screen, ending with a `TOTAL: ...ms | TTFT: ...ms` line. Press `Ctrl+C` to stop; it should print "Stopped." and exit cleanly.

- [ ] **Step 3: Commit**

```bash
git add test_screen_capture.py
git commit -m "Add standalone manual test script for screen capture feature"
```

---

## Post-plan verification checklist

- [ ] `venv/Scripts/python.exe main.py` starts cleanly, registers the hotkey, and the existing audio→question→answer path still works unchanged.
- [ ] Pressing the hotkey while `main.py` is running shows `[Screen capture]` in the overlay and streams a vision answer.
- [ ] Killing network access or using an invalid `OPENROUTER_VISION_MODEL` shows an inline `[Error: ...]` in the overlay instead of crashing the app.
- [ ] `OPENROUTER_API_KEY` is documented as a required `.env` entry alongside `GROQ_API_KEY` / `OLLAMA_API_KEY` for anyone setting up the project fresh.
- [ ] `requirements.txt` and `requirements-lock.txt` both include `mss` and `keyboard` at the versions installed.
