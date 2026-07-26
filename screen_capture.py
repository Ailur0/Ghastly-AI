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
