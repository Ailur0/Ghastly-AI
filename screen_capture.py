"""
screen_capture.py — Primary-monitor screenshot capture + global hotkey registration

Used by the screen capture feature: user presses a global hotkey, we grab
the primary monitor as a PNG, base64-encode it (in main.py), and send it
to a vision-capable model via OpenRouter.
"""

import logging
from typing import Callable

import mss
import mss.tools
import keyboard

try:
    from PIL import Image
    import io
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captures the primary monitor as PNG bytes."""

    def capture_primary_monitor(self, max_width: int = 1280, jpeg_quality: int = 75) -> bytes:
        with mss.MSS() as sct:
            # monitors[0] is the all-monitors bounding box; monitors[1] is the primary monitor
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)

            if _PIL_AVAILABLE:
                # Downscale to max_width and convert to JPEG to drastically
                # reduce payload size (full-res PNG can be 2-4 MB; this gets
                # it under 200 KB and cuts TTFT by ~1-2 s on vision APIs).
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                logger.debug(f"Screenshot compressed: {len(buf.getvalue())} bytes "
                             f"({img.width}x{img.height} JPEG q{jpeg_quality})")
                return buf.getvalue()
            else:
                # PIL not installed — fall back to raw PNG
                return mss.tools.to_png(sct_img.rgb, sct_img.size)


class HotkeyListener:
    """Registers a global hotkey that invokes a callback when pressed."""

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        self.hotkey = hotkey
        self.callback = callback
        self._registered = False

    def start(self) -> bool:
        """Register the hotkey. Returns True on success, False on failure."""
        if self._registered:
            return True
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
    image_bytes = cap.capture_primary_monitor()
    # The compressed path returns JPEG; only the no-PIL fallback is a PNG.
    ext = "jpg" if image_bytes[:2] == b"\xff\xd8" else "png"
    out = f"screen_capture_test.{ext}"
    print(f"Captured {len(image_bytes)} bytes ({ext.upper()})")

    with open(out, "wb") as f:
        f.write(image_bytes)
    print(f"Saved to {out} — open it to confirm it's a valid screenshot")

    def on_hotkey():
        print("Hotkey fired!")

    listener = HotkeyListener("ctrl+shift+h", on_hotkey)
    if listener.start():
        print("Press ctrl+shift+h within the next 10 seconds...")
        time.sleep(10)
        listener.stop()
    else:
        print("Hotkey registration failed")
