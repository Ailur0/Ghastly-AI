"""
test_screen_capture.py — End-to-end manual test for the screen capture feature (no overlay)

Press the configured hotkey → capture screen → query vision LLM → stream answer to terminal.
"""

import sys
import os
import time
import base64
import logging
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from screen_capture import ScreenCapture, HotkeyListener
from llm_query import query_ollama_vision_stream
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

    print(f"Querying {config.OLLAMA_VISION_MODEL}...")
    print(f"{'='*60}\n")

    t0 = time.time()
    full_answer = ""
    meta = None
    for chunk in query_ollama_vision_stream(
        image_b64=image_b64,
        prompt=config.SCREEN_CAPTURE_PROMPT,
        context=cm.get_context_string(),
        state=cm.get_state(),
        api_key=config.OLLAMA_API_KEY,
        model=config.OLLAMA_VISION_MODEL,
        base_url=config.OLLAMA_BASE_URL,
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
    print("  Ghastly AI — Screen Capture Test")
    print("=" * 60)

    cm = ContextManager()
    cm.load_context()
    cm.reset_state()

    screen_capture = ScreenCapture()

    print(f"\nPress {config.SCREEN_CAPTURE_HOTKEY} to capture the screen and query the vision model.")
    print("Press Ctrl+C to stop.\n")

    capture_lock = threading.Lock()

    def on_hotkey():
        if not capture_lock.acquire(blocking=False):
            logger.debug("Capture already in progress, ignoring hotkey press")
            return
        try:
            run_capture_and_query(screen_capture, cm)
        finally:
            capture_lock.release()

    def on_hotkey_threaded():
        threading.Thread(target=on_hotkey, daemon=True).start()

    hotkey = HotkeyListener(config.SCREEN_CAPTURE_HOTKEY, on_hotkey_threaded)
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
