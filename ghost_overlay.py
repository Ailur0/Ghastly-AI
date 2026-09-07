"""
ghost_overlay.py — Cluely-Inspired Dark Glass Overlay

Near-black translucent surfaces, a hairline of light for every edge, and
almost no colour until something needs attention:
  - Pill-shaped command bar (draggable, collapsible) with monochrome glyphs
  - The headline hotkey shown as a chip, the rest behind the info button
  - Status pill reduced to a dot and a word
  - Scrollable answer panel; questions get an accented card, answers do not
  - One palette in class T — restyle there, not in twenty stylesheets

The cursor is pinned to a plain arrow over the entire overlay — hovering
buttons, dragging the bar, and the text panel all keep the default shape,
since WDA hides this window's pixels but not the OS cursor sprite.

On Windows:
  - WDA_EXCLUDEFROMCAPTURE is always active (invisible to screen capture)
"""

import sys
import os
import json
import time
import logging
import ctypes
import random

import config
import file_context

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == 'win32'
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_NONE = 0x00000000
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020

# ── Win32 prototypes ─────────────────────────────────────────────────────
# Every one of these used to be called unprototyped, which means ctypes had
# to guess: a Python int handle went across as a 32-bit C int rather than a
# 64-bit HWND, and the EnumWindows callback was declared to return c_bool —
# one byte — where Win32 reads a four-byte BOOL. Guessing wrong about a
# calling convention does not fail cleanly; it corrupts whatever the callee
# reads next, and it does so differently on different machines.
#
# A crash dump from the machine that fails puts its UI thread inside that
# callback, at the GetWindowThreadProcessId call, which is the first thing it
# does with the handle it was passed.
_U32 = None
_ENUM_WINDOWS_PROC = None
if IS_WINDOWS:
    try:
        from ctypes import wintypes

        _U32 = ctypes.WinDLL("user32", use_last_error=True)
        _ENUM_WINDOWS_PROC = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        _U32.EnumWindows.argtypes = [_ENUM_WINDOWS_PROC, wintypes.LPARAM]
        _U32.EnumWindows.restype = wintypes.BOOL
        _U32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        _U32.GetWindowThreadProcessId.restype = wintypes.DWORD
        _U32.IsWindow.argtypes = [wintypes.HWND]
        _U32.IsWindow.restype = wintypes.BOOL
        _U32.IsWindowVisible.argtypes = [wintypes.HWND]
        _U32.IsWindowVisible.restype = wintypes.BOOL
        _U32.GetWindowDisplayAffinity.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        _U32.GetWindowDisplayAffinity.restype = wintypes.BOOL
        _U32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        _U32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        _U32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        _U32.GetWindowLongW.restype = wintypes.LONG
        _U32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        _U32.SetWindowLongW.restype = wintypes.LONG
    except Exception as _w32_err:                  # never fail to import
        _U32 = None
        logger.error(f"Could not prototype the Win32 calls: {_w32_err}")

# ════════════════════════════════════════════════════════════════
#  Theme — one palette, referenced everywhere
# ════════════════════════════════════════════════════════════════
# Dark glass, the way Cluely does it: a near-black translucent slab, a
# hairline of light for the edge, white text at two or three weights, and
# almost no colour until something needs attention. Every surface in this
# file pulls from here, so a restyle is one block rather than a hunt through
# twenty stylesheets.
class T:
    BG          = "rgba(17, 17, 20, 0.86)"      # bar and panel fill
    BG_RAISED   = "rgba(255, 255, 255, 0.06)"   # inputs, chips
    BORDER      = "rgba(255, 255, 255, 0.10)"   # hairline edge
    BORDER_HI   = "rgba(255, 255, 255, 0.20)"   # focus / hover edge
    HOVER       = "rgba(255, 255, 255, 0.09)"   # button hover fill
    TEXT        = "#F4F4F5"                     # primary
    TEXT_DIM    = "#A1A1AA"                     # secondary
    TEXT_MUTE   = "#71717A"                     # captions, hints
    ACCENT      = "#A5B4FC"                     # the one colour, used sparingly
    ACCENT_SOFT = "rgba(165, 180, 252, 0.14)"
    DANGER      = "#F87171"
    DANGER_SOFT = "rgba(248, 113, 113, 0.14)"
    SELECT      = "rgba(165, 180, 252, 0.30)"
    FONT        = "'Inter', 'Segoe UI', system-ui, sans-serif"
    RADIUS      = 14                            # panel corner


def format_combo(combo: str) -> str:
    """
    ctrl+shift+h -> Ctrl+Shift+H, for anything a person reads.

    The stored value stays lowercase because that is what the keyboard
    library registers; this is presentation only.
    """
    return "+".join(part.capitalize() for part in combo.split("+"))

# ── PyQt5 imports ──
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QTextEdit, QFrame, QGraphicsDropShadowEffect, QPushButton,
        QDialog, QListWidget, QListWidgetItem, QComboBox,
        QFileDialog, QLineEdit, QFileIconProvider
    )
    from PyQt5.QtCore import QUrl, QStandardPaths
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent, QRect
    from PyQt5.QtGui import (
        QColor, QTextCursor, QCursor, QPainter, QPen, QBrush, QKeySequence,
        QTextBlockFormat, QTextCharFormat
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 not installed")


# When the Qt event loop last ran a timer. Read from other threads to tell a
# busy app from a frozen one; a plain float assignment needs no lock.
UI_ALIVE_AT = 0.0


def _ui_tick():
    """
    The UI thread's pulse, and — when enabled — the window sweep.

    The pulse is the reason this timer still runs at all: the heartbeat reads
    it to tell a busy app from a frozen one. The sweep it used to carry is off
    by default now; see CAPTURE_SWEEP.
    """
    global UI_ALIVE_AT
    UI_ALIVE_AT = time.time()
    if config.CAPTURE_SWEEP:
        exclude_process_windows()


def seconds_since_ui_tick():
    """How long the Qt event loop has gone without servicing its timer."""
    return None if not UI_ALIVE_AT else time.time() - UI_ALIVE_AT


def _log_display_environment(app):
    """
    Screens, scaling and Qt versions.

    A dropdown that would not open on someone else's machine cost a long
    round of guessing precisely because none of this was written down: how
    many monitors, what scaling, which Qt. Every window this app places is
    positioned against these numbers.
    """
    try:
        from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        logger.info(f"  qt           : Qt {QT_VERSION_STR} / PyQt {PYQT_VERSION_STR} "
                    f"| platform plugin '{app.platformName()}'")
        screens = app.screens()
        primary = app.primaryScreen()
        logger.info(f"  screens      : {len(screens)}")
        for i, sc in enumerate(screens):
            g, a = sc.geometry(), sc.availableGeometry()
            logger.info(
                f"    [{i}]{' primary' if sc is primary else '        '} "
                f"{g.width()}x{g.height()} at ({g.x()},{g.y()}) "
                f"| work area {a.width()}x{a.height()} "
                f"| dpi {sc.logicalDotsPerInch():.0f} "
                f"| ratio {sc.devicePixelRatio()}")
        # Mixed scaling across monitors is the classic reason a window lands
        # somewhere its owner did not intend.
        ratios = {sc.devicePixelRatio() for sc in screens}
        dpis = {round(sc.logicalDotsPerInch()) for sc in screens}
        if len(ratios) > 1 or len(dpis) > 1:
            logger.warning(f"  screens differ in scaling (ratios={sorted(ratios)}, "
                           f"dpi={sorted(dpis)}) — window placement may be off")
    except Exception as e:
        logger.warning(f"Could not read the display environment: {e}")


# ════════════════════════════════════════════════════════════════
#  Screen-capture exclusion
# ════════════════════════════════════════════════════════════════
def exclude_from_capture(widget) -> bool:
    """
    Hide one top-level window's pixels from screen capture.

    WDA_EXCLUDEFROMCAPTURE applies per HWND, so every window the app puts on
    screen needs its own call — the affinity set on the overlay does not
    inherit to anything else.

    The result is read back rather than assumed. The flag needs Windows 10
    build 19041; on anything older the call fails and the window is plainly
    visible in a screen share, which is worth knowing before an interview
    rather than during one.
    """
    if not IS_WINDOWS or not config.CAPTURE_HIDING:
        return False
    try:
        # Logged before the call, not after: if this is what kills the
        # process, the last line in the log names the window it was touching.
        logger.debug(f"Excluding from capture: {type(widget).__name__} "
                     f"'{widget.windowTitle() or widget.objectName()}'")
        hwnd = int(widget.winId())
        _U32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        return is_excluded(hwnd)
    except Exception as e:
        logger.error(f"Capture exclusion failed for {widget}: {e}")
        return False


def is_excluded(hwnd) -> bool:
    """Whether this window really is hidden from capture, per Windows."""
    if not IS_WINDOWS:
        return False
    try:
        affinity = wintypes.DWORD()
        if not _U32.GetWindowDisplayAffinity(int(hwnd), ctypes.byref(affinity)):
            return False
        return affinity.value == WDA_EXCLUDEFROMCAPTURE
    except Exception as e:
        logger.error(f"Could not read display affinity: {e}")
        return False


def exclude_soon(widget):
    """
    Hide a window from capture on the next turn of the event loop, not now.

    Called from a Show event, "now" means part-way through Qt's own code for
    showing that window — QComboBox.showPopup, QDialog.show — and reaching
    into the native handle there means touching an HWND Qt has not finished
    setting up. A crash dump from the machine that fails puts its UI thread
    inside Qt's showPopup, and the crashes cluster on exactly the two moments
    this app shows a new window.

    The file picker has deferred its exclusion this way from the start and has
    never appeared in a dump. Everything else now does the same.

    One turn of the event loop is a real gap: the window is on screen and
    capturable for a frame or two. That is the price of not reaching into Qt's
    hands while they are full.
    """
    def run():
        try:
            if widget is not None and widget.isVisible():
                exclude_from_capture(widget)
        except RuntimeError:
            # The window closed before we got to it — a dropdown dismissed
            # quickly does this — and there is nothing left to hide.
            logger.debug("Window went away before it could be hidden")
    QTimer.singleShot(0, run)


def exclude_process_windows() -> int:
    """
    Exclude every top-level window this process owns, Qt's or not.

    Windows puts its own drop-shadow window (class SysShadow) behind tooltips
    and menus. It belongs to us but is never created through Qt, so no event
    filter can see it — leaving a shadow-shaped box visible in a screen share
    exactly where the tooltip was, with the text correctly hidden inside it.

    Returns how many windows this call had to fix.
    """
    if not IS_WINDOWS or not config.CAPTURE_HIDING or not config.CAPTURE_SWEEP:
        return 0

    if _U32 is None:
        return 0

    pid_here = ctypes.windll.kernel32.GetCurrentProcessId()
    fixed = 0

    def visit(hwnd, _):
        # Everything in here runs inside a native EnumWindows frame. An
        # exception escaping a ctypes callback into native code is undefined
        # behaviour, so nothing is allowed out — enumeration continues either
        # way, and one unreadable window is not worth the process.
        nonlocal fixed
        try:
            # The window may have been destroyed since EnumWindows listed it.
            # Transient popups — a dropdown closing under the user's finger —
            # make that ordinary rather than rare.
            if not _U32.IsWindow(hwnd):
                return True
            pid = wintypes.DWORD()
            _U32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value != pid_here or not _U32.IsWindowVisible(hwnd):
                return True
            affinity = wintypes.DWORD()
            if (_U32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
                    and affinity.value != WDA_EXCLUDEFROMCAPTURE):
                if _U32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                    fixed += 1
        except Exception as e:
            logger.debug(f"Sweep skipped a window: {e}")
        return True

    try:
        # Held in a local for the duration of the call: if this is collected
        # while EnumWindows is still calling it, the callback address is
        # dangling.
        callback = _ENUM_WINDOWS_PROC(visit)
        _U32.EnumWindows(callback, 0)
    except Exception as e:
        logger.error(f"Window sweep failed: {e}")
    return fixed


if HAS_PYQT:
    class _PopupProbe(QObject):
        """
        Reports the mouse events a dropdown list actually receives.

        Written because a click test using QTest passed on every machine here
        while two other machines could not use the list at all — QTest
        delivers straight to the widget and bypasses the mouse grab, so it
        proved the wiring and not the thing that was broken. Only a real
        press arriving here proves the click got that far.
        """

        def __init__(self, combo):
            super().__init__(combo)
            self.combo = combo

        def eventFilter(self, obj, event):
            kind = {QEvent.MouseButtonPress: "press",
                    QEvent.MouseButtonRelease: "release"}.get(event.type())
            if kind:
                try:
                    pos = event.pos()
                    idx = self.combo.view().indexAt(pos)
                    where = (f"row {idx.row()} = {idx.data()!r}" if idx.isValid()
                             else "no row under the cursor")
                    logger.info(f"Dropdown {kind}: {self.combo.objectName() or 'combo'} "
                                f"at ({pos.x()},{pos.y()}) -> {where}")
                except Exception as e:
                    logger.debug(f"Could not describe a dropdown {kind}: {e}")
            return False           # never consume; only observe


    class TopMostComboBox(QComboBox):
        """
        A dropdown that opens in front of the panel holding it.

        The overlay and the setup panel are both WindowStaysOnTop. A combo's
        popup is a separate top-level window and is not, so Windows draws the
        panel over it — measured at 100% of the popup covered. The list is
        still open and, being a Qt.Popup, still holds the mouse grab, so it
        quietly takes the clicks meant for it. From the outside the dropdown
        appears and simply refuses to change, which is exactly the same shape
        as the file-picker bug: the window is there, just underneath.

        The flag goes on before the popup is shown. Setting it afterwards
        re-creates the native window, which drops the grab.

        The topmost flag did NOT fix the report it was written for — a second
        machine still opens the list and cannot choose from it, with onTop
        confirmed true in its log. So the list also reports the mouse events
        it receives. Three outcomes, three different bugs:

          opened, then nothing            the click never reached the widget
          opened, presses, no activation  it received clicks and mis-hit them
          opened, activated               the click worked; look elsewhere

        Activation is logged rather than only the change, because re-choosing
        the item already selected emits no change and would otherwise read as
        a click that failed.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._probe = _PopupProbe(self)
            self.view().viewport().installEventFilter(self._probe)
            self.activated.connect(
                lambda i: logger.info(
                    f"Dropdown activated: {self.objectName() or 'combo'} "
                    f"row {i} = {self.itemText(i)!r}"))

        def showPopup(self):
            super().showPopup()
            popup = self.view().window()
            # The topmost flag that used to be forced on here is gone. It never
            # did anything — screenshots with and without it were identical,
            # because a Qt.Popup already draws above its parent — and
            # setWindowFlags destroys and recreates a widget's native window.
            # Doing that to a popup Qt owns and reuses leaves it holding a
            # handle it did not make, and Qt touches that popup again when it
            # closes. Which is exactly where the failing machine dies: press
            # and release both land on the right row, then nothing.
            # Its own HWND, so it needs its own exclusion — a dropdown listing
            # answer styles is not something to leak into a screen share — but
            # scheduled rather than done here, for the reason in exclude_soon.
            exclude_soon(popup)
            # Logged on both sides: an "opened" with no "chose" after it is
            # the signature of a list the user could see but not use, which is
            # otherwise indistinguishable from never having clicked at all.
            g = popup.frameGeometry()
            logger.info(f"Dropdown opened: {self.objectName() or 'combo'} "
                        f"({self.count()} items, showing '{self.currentText()}') "
                        f"at ({g.x()},{g.y()}) {g.width()}x{g.height()} "
                        f"| onTop={bool(popup.windowFlags() & Qt.WindowStaysOnTopHint)} "
                        f"| hiding deferred to the next event loop turn")

        def hidePopup(self):
            super().hidePopup()
            logger.debug(f"Dropdown closed: {self.objectName() or 'combo'} "
                         f"on '{self.currentText()}'")


    class BlankIconProvider(QFileIconProvider):
        """
        Hands back nothing instead of asking Windows for file icons.

        QFileDialog's default provider calls SHGetFileInfo, which loads
        third-party shell extensions — cloud sync, antivirus, archivers —
        into this process. One faulty extension takes the whole app down
        with no Python exception to catch, which is exactly what building
        the picker did on one machine.
        """

        def icon(self, _info):
            from PyQt5.QtGui import QIcon
            return QIcon()


    class CaptureShield(QObject):
        """
        Application-wide event filter that excludes every window the app shows.

        Tooltips, combo-box dropdowns and menus are separate top-level windows
        owned by Qt, not children of the overlay — so they were being captured
        in a screen share while the overlay itself stayed invisible. Hovering a
        button was enough to leak its description. Filtering Show events
        catches them all, including any Qt creates internally later.

        Qt windows are handled the moment they appear; the sweep afterwards
        picks up the shadow windows Windows creates for them a beat later.
        """

        SWEEP_DELAYS_MS = (0, 40, 120)

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
                # Deferred, not immediate: this fires inside Qt's own show
                # sequence for the window in question.
                exclude_soon(obj)
                if config.CAPTURE_SWEEP:
                    for delay in self.SWEEP_DELAYS_MS:
                        QTimer.singleShot(delay, exclude_process_windows)
            return False


# ════════════════════════════════════════════════════════════════
#  Signals (thread-safe bridge from background threads → Qt GUI)
# ════════════════════════════════════════════════════════════════
if HAS_PYQT:
    class OverlaySignals(QObject):
        update_text = pyqtSignal(str, bool)      # (text, append)
        append_html = pyqtSignal(str)             # html block
        set_status  = pyqtSignal(str)             # status key
        toggle_vis  = pyqtSignal()                # panic hotkey -> hide/show
        toggle_op   = pyqtSignal()                # opacity hotkey -> sun/moon

    class DraggableWidget(QWidget):
        """QWidget that drags its top-level window on mouse press+move."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self._drag_pos = None

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPos() - self.window().frameGeometry().topLeft()
                event.accept()

        def mouseMoveEvent(self, event):
            if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
                self.window().move(event.globalPos() - self._drag_pos)
                event.accept()

        # Set by GhostOverlay so a finished drag can be remembered. A plain
        # attribute rather than a signal: these widgets are constructed before
        # the overlay has anything to connect to.
        on_release = None

        def mouseReleaseEvent(self, event):
            self._drag_pos = None
            if callable(self.on_release):
                self.on_release()
            event.accept()


    class ResizeGrip(QWidget):
        """
        Small square handle pinned to a corner of the window. Dragging one
        resizes the window from that corner.

        It deliberately sets no cursor of its own: the overlay pins a plain
        arrow everywhere so hovering never hints that an invisible window is
        there, which means the shape has to be the visible affordance.
        """
        SIZE = 12
        MIN_W = 320
        MIN_H = 140

        def __init__(self, corner, parent=None):
            super().__init__(parent)
            self.corner = corner            # 'tl' | 'tr' | 'bl' | 'br'
            self.setFixedSize(self.SIZE, self.SIZE)
            self.setToolTip("Drag to resize")
            self._press_global = None
            self._start_geo = None

        def paintEvent(self, event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            box = self.rect().adjusted(1, 1, -2, -2)
            p.setPen(QPen(QColor(255, 255, 255, 46), 1.0))
            p.setBrush(QBrush(QColor(24, 24, 27, 225)))
            p.drawRoundedRect(box, 3, 3)
            # tiny cross inside, so it reads as a handle and not a bullet
            p.setPen(QPen(QColor(255, 255, 255, 120), 1.2))
            c = box.center()
            p.drawLine(c.x() - 2, c.y(), c.x() + 2, c.y())
            p.drawLine(c.x(), c.y() - 2, c.x(), c.y() + 2)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._press_global = event.globalPos()
                self._start_geo = self.window().geometry()
                event.accept()

        def mouseMoveEvent(self, event):
            if self._press_global is None:
                return
            g = self._start_geo
            d = event.globalPos() - self._press_global
            x, y, w, h = g.x(), g.y(), g.width(), g.height()

            # Dragging a left/top corner moves the origin as well as the size.
            if "l" in self.corner:
                x, w = x + d.x(), w - d.x()
            else:
                w = w + d.x()
            if "t" in self.corner:
                y, h = y + d.y(), h - d.y()
            else:
                h = h + d.y()

            # Clamp, keeping the opposite edge anchored where the user put it.
            if w < self.MIN_W:
                if "l" in self.corner:
                    x -= self.MIN_W - w
                w = self.MIN_W
            if h < self.MIN_H:
                if "t" in self.corner:
                    y -= self.MIN_H - h
                h = self.MIN_H

            self.window().setGeometry(x, y, w, h)
            event.accept()

        on_release = None

        def mouseReleaseEvent(self, event):
            self._press_global = None
            if callable(self.on_release):
                self.on_release()
            event.accept()

    class HotkeyButton(QPushButton):
        def __init__(self, key_name, current_hotkey, on_changed, parent=None):
            super().__init__(format_combo(current_hotkey), parent)
            self.key_name = key_name
            self.current_hotkey = current_hotkey
            self.on_changed = on_changed
            self.listening = False
            self.first_press = None
            self.setStyleSheet("""
                QPushButton {
                    background: rgba(165,180,252,0.12);
                    color: #F4F4F5;
                    border: 1px solid rgba(165,180,252,0.35);
                    border-radius: 7px;
                    padding: 6px 12px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    font-weight: 600;
                    text-align: center;
                }
                QPushButton:hover  { background: rgba(165,180,252,0.22); }
                QPushButton:focus  { border: 1px solid rgba(165,180,252,0.80); background: rgba(165,180,252,0.16); }
            """)
            self.setCursor(Qt.ArrowCursor)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.listening = True
                self.first_press = None
                self.setText("Press hotkey...")
                self.setFocus()
            else:
                super().mousePressEvent(event)

        def focusOutEvent(self, event):
            if self.listening:
                self.listening = False
                self.setText(format_combo(self.current_hotkey))
            super().focusOutEvent(event)

        def keyPressEvent(self, event):
            if not self.listening:
                super().keyPressEvent(event)
                return

            key = event.key()
            if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
                return
            
            if key == Qt.Key_Escape:
                self.listening = False
                self.setText(format_combo(self.current_hotkey))
                return

            mods = event.modifiers()
            if not (mods & (Qt.ControlModifier | Qt.AltModifier |
                            Qt.ShiftModifier | Qt.MetaModifier)):
                # A bare key registers globally, so it would be swallowed in
                # every other app for as long as this one runs.
                self.first_press = None
                self.setText("Needs ctrl / alt / shift...")
                return

            seq = QKeySequence(key | int(mods)).toString(QKeySequence.PortableText).lower()
            # Qt spells the Windows key "meta"; the keyboard library that
            # actually registers the combo only knows it as "windows".
            seq = seq.replace("meta+", "windows+")
            if not seq:
                return

            if self.first_press is None:
                self.first_press = seq
                self.setText(f"Press {format_combo(seq)} again…")
            else:
                if self.first_press == seq:
                    # The owner registers it and says whether it took; only
                    # show the new combo once it is really bound.
                    result = self.on_changed(self.key_name, seq)
                    if isinstance(result, tuple):
                        accepted = result[0]
                    else:
                        accepted = True
                    if accepted:
                        self.current_hotkey = seq
                self.setText(format_combo(self.current_hotkey))
                self.first_press = None
                self.listening = False

    class SetupDialog(QDialog):
        """
        Setup panel: attach resumes/notes, and pick the language code answers
        should be written in.

        Kept capture-excluded like the overlay — a separate top-level window
        would otherwise be plainly visible in a screen share even though the
        overlay behind it is not. Same reason the file picker below is forced
        to Qt's own widget instead of the native Windows one: a native dialog
        is not our window, so we cannot hide it from capture.
        """

        # Where the picker last looked, so a second upload does not start
        # over at the top. Class-level: shared by every panel instance.
        _last_browse_dir = None

        LABEL_CSS = ("color:#A5B4FC;font-family:'Segoe UI',sans-serif;"
                     "font-size:11px;font-weight:700;letter-spacing:0.6px;"
                     "background:transparent;border:none;")
        BTN_CSS = """
            QPushButton {
                background: rgba(165,180,252,0.12);
                color: #F4F4F5;
                border: 1px solid rgba(165,180,252,0.35);
                border-radius: 7px;
                padding: 6px 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover  { background: rgba(165,180,252,0.22); }
            QPushButton:disabled { color:#71717A; border-color:rgba(255,255,255,0.10); }
        """

        def __init__(self, owner, on_changed, parent=None):
            super().__init__(parent)
            self.owner = owner                    # the GhostOverlay
            self.on_changed = on_changed          # (kind, value) -> None
            languages, current_language = owner.languages, owner.code_language
            styles, current_style = owner.answer_styles, owner.answer_style
            self.setWindowTitle(f"{owner.window_title} — Setup")
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint
                                | Qt.WindowStaysOnTopHint | Qt.Tool)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setFixedWidth(400)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(10, 10, 10, 10)

            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: rgba(20,20,23,0.97);
                    border: 1.5px solid rgba(255,255,255,0.10);
                    border-radius: 14px;
                }
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(32)
            shadow.setColor(QColor(0, 0, 0, 55))
            shadow.setOffset(0, 8)
            card.setGraphicsEffect(shadow)
            outer.addWidget(card)

            lay = QVBoxLayout(card)
            lay.setContentsMargins(16, 12, 16, 14)
            lay.setSpacing(8)

            # ── header (doubles as the drag handle) ──
            header = DraggableWidget()
            header.setFixedHeight(26)
            header.setStyleSheet("background:transparent;border:none;")
            hl = QHBoxLayout(header)
            hl.setContentsMargins(0, 0, 0, 0)
            title = QLabel("Setup")
            title.setStyleSheet("color:#F4F4F5;font-family:'Segoe UI',sans-serif;"
                                "font-size:14px;font-weight:700;background:transparent;border:none;")
            hl.addWidget(title)
            hl.addStretch()
            close_btn = QPushButton("✕")
            close_btn.setFixedSize(20, 20)
            close_btn.setStyleSheet("""
                QPushButton { background:transparent;border:none;color:#A1A1AA;
                              font-size:13px;font-weight:700;border-radius:5px; }
                QPushButton:hover { background:rgba(248,113,113,0.16);color:#F87171; }
            """)
            close_btn.clicked.connect(self.close)
            hl.addWidget(close_btn)
            lay.addWidget(header)

            # ── documents ──
            docs_label = QLabel("DOCUMENTS")
            docs_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(docs_label)

            self.file_list = QListWidget()
            self.file_list.setFixedHeight(104)
            self.file_list.setStyleSheet("""
                QListWidget {
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 8px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    color: #F4F4F5;
                    padding: 4px;
                }
                QListWidget::item { padding: 3px 4px; border-radius: 4px; }
                QListWidget::item:hover { background: rgba(255,255,255,0.06); }
                QListWidget::item:selected {
                    background: rgba(165,180,252,0.30);
                    border: 1px solid rgba(165,180,252,0.55);
                    color: #FFFFFF;
                }
            """)
            # Which row Remove will take has to be obvious, so the selection
            # drives the button's enabled state as well as the highlight.
            self.file_list.itemSelectionChanged.connect(self._sync_remove_enabled)
            lay.addWidget(self.file_list)

            row = QHBoxLayout()
            row.setSpacing(6)
            self.add_btn = QPushButton("Add files…")
            self.remove_btn = QPushButton("Remove")
            self.clear_btn = QPushButton("Clear all")
            for b in (self.add_btn, self.remove_btn, self.clear_btn):
                b.setStyleSheet(self.BTN_CSS)
                row.addWidget(b)
            row.addStretch()
            lay.addLayout(row)
            self.add_btn.clicked.connect(self._pick_files)
            self.remove_btn.clicked.connect(self._remove_selected)
            self.clear_btn.clicked.connect(self._clear_all)

            self.status = QLabel("")
            self.status.setWordWrap(True)
            self.status.setStyleSheet("color:#A1A1AA;font-family:'Segoe UI',sans-serif;"
                                      "font-size:11px;background:transparent;border:none;")
            lay.addWidget(self.status)

            # ── answer language ──
            lang_label = QLabel("ANSWER LANGUAGE")
            lang_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(lang_label)

            self.lang_combo = TopMostComboBox()
            self.lang_combo.setObjectName("answer language")
            self.lang_combo.addItems(languages)
            if current_language in languages:
                self.lang_combo.setCurrentText(current_language)
            self.lang_combo.setStyleSheet("""
                QComboBox {
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 8px;
                    padding: 5px 8px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    color: #F4F4F5;
                }
                QComboBox::drop-down { border: none; width: 18px; }
                QComboBox QAbstractItemView {
                    background: #18181B;
                    border: 1px solid rgba(255,255,255,0.10);
                    selection-background-color: rgba(165,180,252,0.20);
                    selection-color: #F4F4F5;
                    color: #F4F4F5;
                    outline: none;
                }
            """)
            self.lang_combo.currentTextChanged.connect(self._language_changed)
            lay.addWidget(self.lang_combo)

            hint = QLabel("Auto follows whatever the question implies.")
            hint.setStyleSheet("color:#A1A1AA;font-family:'Segoe UI',sans-serif;"
                               "font-size:11px;background:transparent;border:none;")
            lay.addWidget(hint)

            # ── answer style ──
            style_label = QLabel("ANSWER STYLE")
            style_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(style_label)

            self.style_combo = TopMostComboBox()
            self.style_combo.setObjectName("answer style")
            self.style_combo.addItems(styles)
            if current_style in styles:
                self.style_combo.setCurrentText(current_style)
            self.style_combo.setStyleSheet(self.lang_combo.styleSheet())
            self.style_combo.currentTextChanged.connect(self._style_changed)
            lay.addWidget(self.style_combo)

            self.style_hint = QLabel("")
            self.style_hint.setWordWrap(True)
            self.style_hint.setStyleSheet("color:#A1A1AA;font-family:'Segoe UI',sans-serif;"
                                          "font-size:11px;background:transparent;border:none;")
            lay.addWidget(self.style_hint)
            self._describe_style(self.style_combo.currentText())

            # ── audio device ──
            audio_label = QLabel("AUDIO SOURCE")
            audio_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(audio_label)

            self.audio_combo = TopMostComboBox()
            self.audio_combo.setObjectName("audio source")
            self.audio_combo.setStyleSheet(self.lang_combo.styleSheet())
            for device_id, device_label in owner.audio_devices:
                self.audio_combo.addItem(device_label, device_id)
            index = self.audio_combo.findData(owner.audio_device)
            if index >= 0:
                self.audio_combo.setCurrentIndex(index)
            self.audio_combo.currentIndexChanged.connect(self._audio_changed)
            lay.addWidget(self.audio_combo)

            audio_hint = QLabel("Pick the loopback device carrying the "
                                "interviewer's voice. Changing it restarts capture.")
            audio_hint.setWordWrap(True)
            audio_hint.setStyleSheet("color:#A1A1AA;font-family:'Segoe UI',sans-serif;"
                                     "font-size:11px;background:transparent;border:none;")
            lay.addWidget(audio_hint)

            # ── hotkeys ──
            if owner.hotkeys:
                hotkeys_label = QLabel("HOTKEYS")
                hotkeys_label.setStyleSheet(self.LABEL_CSS)
                lay.addWidget(hotkeys_label)

                for label_text, current_hotkey in owner.hotkeys:
                    row = QHBoxLayout()
                    lbl = QLabel(label_text)
                    lbl.setStyleSheet("color:#F4F4F5;font-family:'Segoe UI',sans-serif;"
                                      "font-size:12px;background:transparent;border:none;")
                    
                    btn = HotkeyButton(label_text, current_hotkey, self._hotkey_changed)
                    btn.setFixedWidth(160)

                    row.addWidget(lbl)
                    row.addStretch()
                    row.addWidget(btn)
                    lay.addLayout(row)

            self.refresh_files()

        # ── capture exclusion ──
        def showEvent(self, event):
            super().showEvent(event)
            exclude_from_capture(self)

        # ── file handling ──
        def refresh_files(self):
            self.file_list.clear()
            try:
                entries = file_context.list_files()
            except Exception as e:
                logger.exception("Could not list uploaded documents")
                entries = []
                self.status.setText(f"Documents folder unreadable: {e}")
            for stored, original, chars in entries:
                item = QListWidgetItem(f"{original}  ·  {chars:,} chars")
                item.setData(Qt.UserRole, stored)
                self.file_list.addItem(item)
            if not entries:
                placeholder = QListWidgetItem("No documents yet — add a resume to start.")
                placeholder.setFlags(Qt.NoItemFlags)
                self.file_list.addItem(placeholder)
            else:
                # Nothing was selected on open, so Remove was enabled but
                # answered every click with "select a document first" — which
                # reads exactly like a dead button. Start on the first row.
                self.file_list.setCurrentRow(0)
            self.clear_btn.setEnabled(bool(entries))
            self._sync_remove_enabled()

        def _sync_remove_enabled(self):
            """Remove is only meaningful with a row selected — say so by
            being disabled rather than by refusing the click afterwards."""
            item = self.file_list.currentItem()
            self.remove_btn.setEnabled(bool(item and item.data(Qt.UserRole)))

        def _pick_files(self):
            try:
                self._pick_files_inner()
            except Exception as e:
                logger.exception("File picker failed")
                self.status.setText(f"Could not add files: {e}")

        def _pick_files_inner(self):
            """
            Every step logs before it runs.

            A Python exception here is caught and shown, but a crash below
            Python — in Qt, in a shell extension the picker loads, or an
            antivirus killing the process — takes the app with it and leaves
            no traceback. The breadcrumbs are what tell us which step it was.
            """
            native = config.NATIVE_FILE_DIALOG or self.owner.force_native_picker
            logger.info(f"Upload: building the file picker (native={native})")
            # If this line is the last thing in the log, the picker crashed
            # the process and the next launch will switch to the native one.
            file_context.mark_picker_open("native" if native else "qt")

            dlg = QFileDialog(self, "Add resume or notes")
            dlg.setFileMode(QFileDialog.ExistingFiles)
            # The overlay and this panel are both WindowStaysOnTop. Without
            # the same hint the picker opens *under* them — and since it is
            # modal, every click on the panel above it is swallowed, so the
            # app looks frozen behind a dialog you cannot reach. Measured at
            # 62% of the picker covered, Open and Cancel among it.
            dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
            if not native:
                # Qt's own dialog, not the native one — see the class
                # docstring. The native one cannot be hidden from capture.
                dlg.setOption(QFileDialog.DontUseNativeDialog, True)
                # Keep the Windows shell out of it: no shell icon lookups, no
                # shell-populated sidebar, and start somewhere plain.
                dlg.setOption(QFileDialog.DontUseCustomDirectoryIcons, True)
                dlg.setIconProvider(BlankIconProvider())
                # Desktop and Documents are redirected into OneDrive on this
                # kind of setup, so they are not reachable by browsing down
                # from home the way someone expects — they have to be offered
                # directly. Downloads and Desktop are where a resume actually
                # lives, so lead with them.
                places, seen = [], set()
                for loc in (QStandardPaths.DownloadLocation,
                            QStandardPaths.DesktopLocation,
                            QStandardPaths.DocumentsLocation,
                            QStandardPaths.HomeLocation):
                    path = QStandardPaths.writableLocation(loc)
                    if path and path not in seen and os.path.isdir(path):
                        seen.add(path)
                        places.append(path)
                if places:
                    dlg.setSidebarUrls([QUrl.fromLocalFile(p) for p in places])
                    # Reopen where they left off; otherwise the first place
                    # that exists, which is Downloads on a normal machine.
                    start = SetupDialog._last_browse_dir
                    dlg.setDirectory(start if start and os.path.isdir(start)
                                     else places[0])
            dlg.setNameFilter("Documents (*.pdf *.docx *.txt *.md *.json *.csv);;All files (*)")

            # Exclude it once it is on screen. Queued rather than show()-then-
            # exec_(), which meant opening the dialog twice.
            def _surface():
                exclude_from_capture(dlg)
                # Both windows sit at the always-on-top level now, so settle
                # the order explicitly rather than trusting activation.
                dlg.raise_()
                dlg.activateWindow()

            QTimer.singleShot(0, _surface)
            QTimer.singleShot(80, exclude_process_windows)

            logger.info("Upload: opening the file picker")
            accepted = dlg.exec_()
            file_context.clear_picker_flag()
            # Remember the folder even on cancel — browsing somewhere and
            # backing out is still a hint about where the files are.
            try:
                SetupDialog._last_browse_dir = dlg.directory().absolutePath()
            except Exception as e:
                logger.debug(f"Could not read the picker's directory: {e}")
            logger.info(f"Upload: picker closed (accepted={bool(accepted)}, "
                        f"dir={SetupDialog._last_browse_dir})")
            if not accepted:
                return

            selected = dlg.selectedFiles()
            logger.info(f"Upload: {len(selected)} file(s) chosen")

            added, errors = 0, []
            for path in selected:
                name = os.path.basename(path)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = -1
                logger.info(f"Upload: extracting {name} ({size} bytes)")
                try:
                    stored = file_context.add_file(path)
                    added += 1
                    logger.info(f"Upload: stored {name} as {stored}")
                except Exception as e:
                    logger.exception(f"Upload: {name} failed")
                    errors.append(f"{name}: {e}")

            logger.info("Upload: refreshing the list")
            self.refresh_files()
            if added:
                self.on_changed("files", added)
            msg = f"Added {added} file{'s' if added != 1 else ''}." if added else ""
            if errors:
                msg = (msg + "  " if msg else "") + "Skipped — " + "; ".join(errors)
            self.status.setText(msg)
            logger.info(f"Upload: done — {added} added, {len(errors)} failed")

        def _remove_selected(self):
            try:
                self._remove_selected_inner()
            except Exception as e:
                logger.exception("Remove failed")
                self.status.setText(f"Could not remove: {e}")

        def _remove_selected_inner(self):
            item = self.file_list.currentItem()
            stored = item.data(Qt.UserRole) if item else None
            if not stored:
                self.status.setText("Select a document first.")
                return
            file_context.remove_file(stored)
            self.refresh_files()
            self.on_changed("files", -1)
            self.status.setText("Removed.")

        def _clear_all(self):
            try:
                n = file_context.clear_files()
            except Exception as e:
                logger.exception("Clear failed")
                self.status.setText(f"Could not clear: {e}")
                return
            self.refresh_files()
            self.on_changed("files", 0)
            self.status.setText(f"Cleared {n} document{'s' if n != 1 else ''}.")

        STYLE_HINTS = {
            "Balanced": "A sentence of reasoning, then the smallest snippet.",
            "Snippet only": "Code and nothing else.",
            "Text only": "Spoken explanation, no code at all.",
            "Full walkthrough": "Full code, then the approach, then the decisions made.",
        }

        def _describe_style(self, text):
            self.style_hint.setText(self.STYLE_HINTS.get(text, ""))

        def _style_changed(self, text):
            logger.info(f"Setup: answer style chosen -> {text}")
            self._describe_style(text)
            self.on_changed("style", text)
            self.status.setText(f"Answer style: {text}.")

        def _audio_changed(self, index):
            device_id = self.audio_combo.itemData(index)
            logger.info(f"Setup: audio source chosen -> {device_id} "
                        f"({self.audio_combo.currentText()})")
            self.on_changed("audio_device", device_id)
            self.status.setText("Audio source: {}".format(self.audio_combo.currentText()))

        def _language_changed(self, text):
            logger.info(f"Setup: answer language chosen -> {text}")
            self.on_changed("language", text)
            self.status.setText(f"Code answers will use {text}."
                                if text != "Auto" else
                                "Code answers follow the question.")

        def _hotkey_changed(self, key_name, new_value):
            result = self.on_changed("hotkey", (key_name, new_value))
            if isinstance(result, tuple):
                ok, message = result
            else:
                ok, message = True, f"Hotkey for '{key_name}' updated."
            self.status.setText(message)
            return ok, message


# ════════════════════════════════════════════════════════════════
#  Status definitions
# ════════════════════════════════════════════════════════════════
# Text, foreground, fill, border. A dot instead of an emoji: the state reads
# at a glance without another piece of colour competing with the answer.
STATUS_MAP = {
    "ready":        ("● Ready",        T.TEXT_MUTE, "rgba(255,255,255,0.05)", T.BORDER),
    "listening":    ("● Listening",    T.ACCENT,    T.ACCENT_SOFT,            "rgba(165,180,252,0.28)"),
    "transcribing": ("● Transcribing", T.ACCENT,    T.ACCENT_SOFT,            "rgba(165,180,252,0.28)"),
    "answering":    ("● Answering",    T.ACCENT,    T.ACCENT_SOFT,            "rgba(165,180,252,0.28)"),
    "error":        ("● Error",        T.DANGER,    T.DANGER_SOFT,            "rgba(248,113,113,0.30)"),
    "offline":      ("● Offline",      T.DANGER,    T.DANGER_SOFT,            "rgba(248,113,113,0.30)"),
}


# ════════════════════════════════════════════════════════════════
#  GhostOverlay — main overlay controller
# ════════════════════════════════════════════════════════════════
class GhostOverlay:
    """
    Cluely-inspired frosted glass overlay.

    Public API used by main.py:
        init_window()          — create GUI on main thread
        exec()                 — run Qt event loop (blocks)
        set_status(key)        — update status pill (thread-safe)
        show_question(text)    — show question card (thread-safe)
        stream_answer(chunk)   — append answer text (thread-safe)
        show_latency(ms, ttft) — show latency footer (thread-safe)
        update_answer(text, append) — raw text update (thread-safe)
        stop()                 — quit
    """

    # ── Geometry defaults ──
    BAR_W = 420
    BAR_H = 38
    PANEL_W = 480
    PANEL_H = 380

    # ── Opacity defaults (Qt window opacity scale, 0.0-1.0) ──
    OPACITY_OPAQUE = 1.0
    OPACITY_TRANSLUCENT = 0.5

    # ── Title scramble effect (plays once at startup) ──
    SCRAMBLE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%&*+=<>?"
    SCRAMBLE_DURATION_MS = 800
    SCRAMBLE_TICK_MS = 40

    def __init__(self, **kwargs):
        self.BAR_W   = kwargs.get("bar_width",   self.BAR_W)
        self.BAR_H   = kwargs.get("bar_height",  self.BAR_H)
        self.PANEL_W = kwargs.get("panel_width",  self.PANEL_W)
        self.PANEL_H = kwargs.get("panel_height", self.PANEL_H)
        self.position = kwargs.get("position", "top-center")
        self.OPACITY_OPAQUE = kwargs.get("opacity_opaque", self.OPACITY_OPAQUE)
        self.OPACITY_TRANSLUCENT = kwargs.get("opacity_translucent", self.OPACITY_TRANSLUCENT)
        # List of (label, key-combo) pairs, e.g. [("Screen capture", "ctrl+shift+h")]
        self.hotkeys = kwargs.get("hotkeys", [])
        # Languages offered in the setup panel, and the one selected now.
        self.languages = kwargs.get("languages", ["Auto"])
        self.code_language = kwargs.get("code_language", "Auto")
        # Answer shapes offered in the setup panel, and the one selected now.
        self.answer_styles = kwargs.get("answer_styles", ["Balanced"])
        self.answer_style = kwargs.get("answer_style", "Balanced")
        # [(device_id, label)] for the setup panel, and the one in use.
        self.audio_devices = kwargs.get("audio_devices", [("Auto", "Auto — detect")])
        self.audio_device = kwargs.get("audio_device", "Auto")
        self.auto_scroll = kwargs.get("auto_scroll", True)
        # Flipped on when a previous run died with the Qt picker open.
        self.force_native_picker = kwargs.get("force_native_picker", False)
        # Called with the text of a typed question, and with no arguments to
        # re-answer the last one.
        self.on_question_typed = kwargs.get("on_question_typed", None)
        self.on_retry = kwargs.get("on_retry", None)
        # Called as on_setup_changed(kind, value) with kind "files"|"language".
        self.on_setup_changed = kwargs.get("on_setup_changed", None)

        self.app = None
        self.window = None
        self.bar = None
        self.panel = None
        self.text_widget = None
        self.status_pill = None
        self.opacity_btn = None
        self.info_btn = None
        self.ask_hint = None
        self.title_label = None
        self._scramble_timer = None

        self._grips = {}
        self._expanded_h = None
        self._setup_dialog = None
        self._hotkeys_popover = None
        self._capture_shield = None
        self._sweep_timer = None

        # Set once the window exists and Windows confirms the flag stuck.
        self.capture_hidden = False
        self.window_title = kwargs.get("window_title", "System Audio Helper")

        self._is_running = False
        self._expanded = True
        self._opaque = True
        self._question_count = 0
        self._hwnd = None

        if HAS_PYQT:
            self.signals = OverlaySignals()
            self.signals.update_text.connect(self._slot_update_text)
            self.signals.append_html.connect(self._slot_append_html)
            self.signals.set_status.connect(self._slot_set_status)
            self.signals.toggle_vis.connect(self._slot_toggle_visible)
            self.signals.toggle_op.connect(self._toggle_opacity)

    # ────────────────────────────────────────────────
    #  Window positioning
    # ────────────────────────────────────────────────
    def _get_position(self, sw, sh):
        w = max(self.BAR_W, self.PANEL_W)
        h = self.BAR_H + self.PANEL_H + 8
        positions = {
            "top-center":   ((sw - w) // 2, 30),
            "center":       ((sw - w) // 2, (sh - h) // 2),
            "top-left":     (20, 30),
            "top-right":    (sw - w - 20, 30),
            "bottom-left":  (20, sh - h - 60),
            "bottom-right": (sw - w - 20, sh - h - 60),
        }
        return positions.get(self.position, positions["top-center"])

    # ────────────────────────────────────────────────
    #  WDA (Windows Display Affinity)
    # ────────────────────────────────────────────────
    def _set_wda(self, exclude: bool):
        if not IS_WINDOWS or not self._hwnd:
            self.capture_hidden = False
            return
        if not config.CAPTURE_HIDING:
            # The diagnostic switch has to reach this path too, or the main
            # window stays excluded and the experiment answers nothing.
            self.capture_hidden = False
            logger.info("WDA skipped (CAPTURE_HIDING=0)")
            return
        try:
            flag = WDA_EXCLUDEFROMCAPTURE if exclude else WDA_NONE
            _U32.SetWindowDisplayAffinity(self._hwnd, flag)
            # Read it back: the call can fail quietly on Windows 10 before
            # build 19041, and "we asked for it" is not the same as "it is on".
            self.capture_hidden = is_excluded(self._hwnd) if exclude else False
            if exclude and not self.capture_hidden:
                logger.critical("WDA_EXCLUDEFROMCAPTURE did not take — this "
                                "window IS VISIBLE to screen capture")
            else:
                logger.info(f"WDA {'EXCLUDE (verified)' if exclude else 'NONE'}")
        except Exception as e:
            logger.error(f"WDA error: {e}")

    def _set_click_through(self, enabled: bool):
        if not IS_WINDOWS or not self._hwnd:
            return
        try:
            style = _U32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
            if enabled:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            _U32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, style)
        except Exception as e:
            logger.error(f"Click-through error: {e}")

    # ────────────────────────────────────────────────
    #  Build the UI
    # ────────────────────────────────────────────────
    def _create_window(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        _log_display_environment(self.app)

        # Catch every window the app opens — tooltips and dropdowns included.
        self._capture_shield = CaptureShield()
        self.app.installEventFilter(self._capture_shield)

        # Backstop: anything the event filter and its sweeps still miss gets
        # picked up within half a second. Cheap — EnumWindows over one process.
        # It doubles as the UI thread's pulse: it can only run if the Qt event
        # loop is still servicing timers, so a stale stamp means the interface
        # has stopped responding even though the process is still alive.
        self._sweep_timer = QTimer()
        self._sweep_timer.timeout.connect(_ui_tick)
        self._sweep_timer.start(500)
        if not config.CAPTURE_HIDING:
            logger.critical(
                "CAPTURE_HIDING=0 — this overlay IS VISIBLE in a screen share. "
                "Diagnostic mode only: it exists to find out whether hiding "
                "windows from capture is what crashes this machine.")

        total_w = max(self.BAR_W, self.PANEL_W) + 24
        total_h = self.BAR_H + self.PANEL_H + 32

        self.window = QWidget()
        # The OS-level title is readable by anything that enumerates windows;
        # the name shown inside the bar is a label, not this.
        self.window.setWindowTitle(self.window_title)
        self.window.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.window.setAttribute(Qt.WA_TranslucentBackground, True)

        screen = self.app.primaryScreen().geometry()
        x, y = self._get_position(screen.width(), screen.height())
        rect = self._restore_geometry(x, y, total_w, total_h)
        self.window.setGeometry(*rect)

        # ── Root layout ──
        root = QVBoxLayout(self.window)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(4)

        # ═══ COMMAND BAR ═══
        self.bar = DraggableWidget()
        self.bar.setFixedHeight(self.BAR_H)
        # A bare QWidget ignores a stylesheet background unless it is told to
        # draw one, and the id selector keeps the fill off every child.
        self.bar.setObjectName("commandBar")
        self.bar.setAttribute(Qt.WA_StyledBackground, True)
        self.bar.setStyleSheet(f"""
            #commandBar {{
                background-color: {T.BG};
                border: 1px solid {T.BORDER};
                border-radius: {self.BAR_H // 2}px;
            }}
        """)

        bar_shadow = QGraphicsDropShadowEffect()
        bar_shadow.setBlurRadius(40)
        bar_shadow.setColor(QColor(0, 0, 0, 130))
        bar_shadow.setOffset(0, 10)
        self.bar.setGraphicsEffect(bar_shadow)

        bar_layout = QHBoxLayout(self.bar)
        bar_layout.setContentsMargins(10, 0, 8, 0)
        bar_layout.setSpacing(8)

        # Wordmark first, the way Cluely leads with its name, then the tools.
        self.title_label = QLabel("Ghastly")
        self.title_label.setStyleSheet(f"""
            color: {T.TEXT};
            font-family: {T.FONT};
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.2px;
            background: transparent;
            border: none;
        """)
        bar_layout.addWidget(self.title_label)

        bar_layout.addWidget(self._bar_divider())

        # Opacity toggle (filled = opaque, hollow = translucent)
        self.opacity_btn = self._bar_button(
            "◐", "Opaque — click to make translucent", self._toggle_opacity)
        bar_layout.addWidget(self.opacity_btn)

        # Setup (documents + answer language)
        self.setup_btn = self._bar_button(
            "⚙", "Documents & answer language", self._open_setup)
        bar_layout.addWidget(self.setup_btn)

        # Retry — re-answers the last question
        self.retry_btn = self._bar_button(
            "↻", "Answer the last question again", self._on_retry_clicked)
        bar_layout.addWidget(self.retry_btn)

        # Info — click for the hotkey list. It used to be a hover tooltip,
        # which meant the list appeared when you were reaching for something
        # else and vanished the moment you tried to read it.
        self.info_btn = self._bar_button("ⓘ", "Hotkeys",
                                         self._toggle_hotkeys_popover)
        bar_layout.addWidget(self.info_btn)

        bar_layout.addStretch()

        # The headline shortcut, shown the way Cluely shows them — the one
        # hotkey worth knowing without opening anything.
        self.ask_hint = QLabel(format_combo(config.GRAB_HOTKEY))
        self.ask_hint.setToolTip("Answer what was just said")
        self.ask_hint.setStyleSheet(f"""
            QLabel {{
                color: {T.TEXT_MUTE};
                background: {T.BG_RAISED};
                border: 1px solid {T.BORDER};
                border-radius: 6px;
                padding: 2px 7px;
                font-family: {T.FONT};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
        """)
        bar_layout.addWidget(self.ask_hint)

        # Status pill
        self.status_pill = QLabel("⚡ Ready")
        self._apply_status_style("ready")
        bar_layout.addWidget(self.status_pill)

        bar_layout.addSpacing(6)

        # Close — a quiet glyph rather than a traffic light, which is the one
        # piece of chrome that always read as "some app is running here".
        close_btn = self._bar_button("✕", "Close", self._on_close,
                                     hover_bg=T.DANGER_SOFT, hover_fg=T.DANGER)
        bar_layout.addWidget(close_btn)

        # Click on bar toggles expand/collapse
        self.bar.mouseDoubleClickEvent = lambda e: self._toggle_panel()

        root.addWidget(self.bar)

        # ═══ ANSWER PANEL ═══
        self.panel = QFrame()
        self.panel.setObjectName("answerPanel")
        self.panel.setStyleSheet(f"""
            #answerPanel {{
                background-color: {T.BG};
                border: 1px solid {T.BORDER};
                border-radius: {T.RADIUS}px;
            }}
        """)

        panel_shadow = QGraphicsDropShadowEffect()
        panel_shadow.setBlurRadius(44)
        panel_shadow.setColor(QColor(0, 0, 0, 150))
        panel_shadow.setOffset(0, 12)
        self.panel.setGraphicsEffect(panel_shadow)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)

        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        self.text_widget.viewport().setCursor(QCursor(Qt.ArrowCursor))
        self.text_widget.setStyleSheet(f"""
            QTextEdit {{
                background: transparent;
                color: {T.TEXT};
                font-family: {T.FONT};
                font-size: 15px;
                font-weight: 400;
                line-height: 1.65;
                border: none;
                selection-background-color: {T.SELECT};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.16);
                min-height: 30px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.32);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        panel_layout.addWidget(self.text_widget)

        # ── Ask box ──
        # The whole pipeline depends on the interviewer's audio reaching a
        # loopback device. When that fails — wrong device, muted call, a term
        # Whisper mangles — this is the way back in.
        self.ask_input = QLineEdit()
        self.ask_input.setPlaceholderText("Ask anything…")
        self.ask_input.setFixedHeight(34)
        self.ask_input.setStyleSheet(f"""
            QLineEdit {{
                background: {T.BG_RAISED};
                border: 1px solid {T.BORDER};
                border-radius: 10px;
                padding: 5px 12px;
                font-family: {T.FONT};
                font-size: 13px;
                color: {T.TEXT};
                selection-background-color: {T.SELECT};
            }}
            QLineEdit:focus {{ border: 1px solid {T.BORDER_HI}; }}
        """)
        # Qt gives a line edit an I-beam; pin it like the answer panel so no
        # cursor shape ever hints that something is here.
        self.ask_input.setCursor(QCursor(Qt.ArrowCursor))
        self.ask_input.returnPressed.connect(self._on_question_submitted)
        panel_layout.addWidget(self.ask_input)

        root.addWidget(self.panel)

        # ── Resize grips ──
        # Four corner handles, repositioned whenever the window changes size.
        # They are children of the window rather than layout items, so they
        # float over the card instead of taking space in it.
        self._expanded_h = total_h
        self._grips = {c: ResizeGrip(c, self.window) for c in ("tl", "tr", "bl", "br")}
        # Moving or resizing is the user placing the window; remember it.
        self.bar.on_release = self.save_geometry
        for _grip in self._grips.values():
            _grip.on_release = self.save_geometry
        self.window.resizeEvent = lambda e: self._position_grips()
        self._position_grips()

        # ── Show ──
        # Force a plain arrow cursor everywhere over the overlay, in every
        # state — hovering a button, dragging the bar, over the text panel.
        # WDA_EXCLUDEFROMCAPTURE hides this window's pixels from screen
        # capture, but NOT the OS mouse cursor sprite, so any cursor shape
        # change would reveal that something interactive is here.
        QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))

        self.window.setWindowOpacity(self.OPACITY_OPAQUE)
        self.window.show()

        if IS_WINDOWS:
            self._hwnd = int(self.window.winId())
            # Always excluded from screen capture — not a toggle.
            self._set_wda(True)

        if not self.capture_hidden:
            reason = ("this platform has no equivalent of "
                      "WDA_EXCLUDEFROMCAPTURE" if not IS_WINDOWS else
                      "Windows refused it — build 19041 (Windows 10 2004) or "
                      "newer is required")
            self.append_html(
                f'<div style="background:{T.DANGER_SOFT};'
                f'border-left:2px solid {T.DANGER};border-radius:6px;'
                'padding:10px 14px;margin:10px 0;">'
                f'<span style="color:{T.DANGER};font-size:10px;font-weight:700;'
                'letter-spacing:0.8px;">VISIBLE TO SCREEN CAPTURE</span><br/>'
                f'<span style="color:{T.TEXT};font-size:13px;">'
                f'This overlay is NOT hidden — {reason}. '
                'Anyone you share your screen with will see it.</span></div>')

        self._start_title_scramble()

        logger.info(f"Cluely overlay created at {self.position} (always excluded from screen capture)")

    # ────────────────────────────────────────────────
    #  Title scramble effect (plays once at startup)
    # ────────────────────────────────────────────────
    def _start_title_scramble(self):
        """
        Animate the title label from scrambled noise into "Ghastly AI",
        resolving left-to-right. Runs once, at startup.
        """
        final_text = self.title_label.text()
        total_frames = self.SCRAMBLE_DURATION_MS // self.SCRAMBLE_TICK_MS
        n = len(final_text)

        # Stagger each character's resolve frame left-to-right, with a
        # little jitter so it doesn't look mechanically even. Spaces
        # resolve immediately so the word gap never visibly scrambles.
        resolve_frames = []
        for i, ch in enumerate(final_text):
            if ch == " ":
                resolve_frames.append(0)
                continue
            base = (i + 1) * total_frames / n
            resolve_frames.append(max(1, int(base + random.randint(-2, 2))))

        frame = 0

        def tick():
            nonlocal frame
            frame += 1
            chars = [
                ch if ch == " " or frame >= resolve_at else random.choice(self.SCRAMBLE_CHARS)
                for ch, resolve_at in zip(final_text, resolve_frames)
            ]
            self.title_label.setText("".join(chars))
            if frame >= total_frames:
                self.title_label.setText(final_text)
                self._scramble_timer.stop()

        self._scramble_timer = QTimer(self.title_label)
        self._scramble_timer.timeout.connect(tick)
        tick()  # set the first scrambled frame now, before the Qt event loop starts painting
        self._scramble_timer.start(self.SCRAMBLE_TICK_MS)

    # ────────────────────────────────────────────────
    #  Status pill styling
    # ────────────────────────────────────────────────
    def _apply_status_style(self, key: str):
        text, color, bg, border_c = STATUS_MAP.get(key, STATUS_MAP["ready"])
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: {bg};
                border: 1px solid {border_c};
                border-radius: 9px;
                padding: 2px 9px;
                font-family: {T.FONT};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }}
        """)

    # ────────────────────────────────────────────────
    #  Toggle handlers
    # ────────────────────────────────────────────────
    def _position_grips(self):
        """Pin the four handles to the window corners; hide them when collapsed."""
        if not self._grips:
            return
        w, h = self.window.width(), self.window.height()
        s, pad = ResizeGrip.SIZE, 4
        for corner, grip in self._grips.items():
            grip.move(pad if "l" in corner else w - s - pad,
                      pad if "t" in corner else h - s - pad)
            grip.setVisible(self._expanded)
            grip.raise_()

    def _toggle_panel(self):
        """Expand / collapse the answer panel."""
        self._expanded = not self._expanded
        self.panel.setVisible(self._expanded)
        w = self.window.width()

        if self._expanded:
            # Undo the collapsed height pin, otherwise the grips can never
            # grow the window again.
            self.window.setMinimumHeight(ResizeGrip.MIN_H)
            self.window.setMaximumHeight(16777215)
            self.window.resize(w, self._expanded_h or (self.BAR_H + self.PANEL_H + 32))
        else:
            # Remember whatever height the user resized to before collapsing.
            self._expanded_h = self.window.height()
            self.window.setFixedHeight(self.BAR_H + 28)

        self._position_grips()
        logger.info(f"Panel {'expanded' if self._expanded else 'collapsed'}")

    def _open_setup(self):
        """
        Open (or re-focus) the setup panel next to the overlay.

        Everything here is guarded: an exception raised inside a Qt slot
        aborts the whole process, so a folder the app cannot read must end as
        a message in the panel, not a vanished app.
        """
        try:
            logger.info("Setup: opening the panel")
            self._build_and_show_setup()
            logger.info("Setup: panel ready")
        except Exception as e:
            logger.exception("Could not open the setup panel")
            self._setup_dialog = None
            self.append_html(
                f'<div style="color:{T.DANGER};font-size:12px;padding-left:4px;">'
                f'Setup panel unavailable: {e}</div>')

    def _build_and_show_setup(self):
        if self._setup_dialog is None:
            self._setup_dialog = SetupDialog(self, self._on_setup_changed,
                                             parent=self.window)
        else:
            self._setup_dialog.refresh_files()

        # Beside the overlay rather than under it: stacked, the panel pushes
        # itself off the bottom of the screen on a short display and covers
        # the answer the overlay is there to show.
        dlg = self._setup_dialog
        dlg.adjustSize()
        geo = self.window.geometry()
        screen = self.app.primaryScreen().availableGeometry()
        gap = 8
        w, h = dlg.frameGeometry().width(), dlg.frameGeometry().height()

        x = geo.x() + geo.width() + gap                  # to the right
        if x + w > screen.right():
            x = geo.x() - w - gap                        # no room — go left
        x = max(screen.left() + 4, min(x, screen.right() - w - 4))
        y = max(screen.top() + 4, min(geo.y(), screen.bottom() - h - 4))
        dlg.move(x, y)
        side = "right of" if x > geo.x() else "left of"
        fits = y + h <= screen.bottom() and x + w <= screen.right()
        logger.info(f"Setup panel: {w}x{h} at ({x},{y}), {side} the overlay "
                    f"| work area {screen.width()}x{screen.height()} "
                    f"| fits on screen={fits}")
        if not fits:
            logger.warning("Setup panel does not fit the work area — some of "
                           "it is off-screen and there is no scroll area yet")
        self._setup_dialog.show()
        self._setup_dialog.raise_()
        self._setup_dialog.activateWindow()

    def _on_setup_changed(self, kind, value):
        """Relay a setup-panel change to whoever owns the context."""
        if kind == "language":
            self.code_language = value
        elif kind == "style":
            self.answer_style = value
        elif kind == "audio_device":
            self.audio_device = value
        if callable(self.on_setup_changed):
            try:
                self.on_setup_changed(kind, value)
            except Exception as e:
                logger.error(f"Setup callback failed: {e}")

    def _on_question_submitted(self):
        """Enter in the ask box — hand the text to whoever answers questions."""
        text = self.ask_input.text().strip()
        if not text:
            return
        self.ask_input.clear()
        if callable(self.on_question_typed):
            try:
                self.on_question_typed(text)
            except Exception as e:
                logger.error(f"Typed-question callback failed: {e}")
        else:
            logger.warning("No handler for typed questions")

    def _on_retry_clicked(self):
        if callable(self.on_retry):
            try:
                self.on_retry()
            except Exception as e:
                logger.error(f"Retry callback failed: {e}")

    def toggle_visibility(self):
        """Hide / show the whole overlay. Thread-safe (panic hotkey)."""
        if self._is_running and HAS_PYQT and hasattr(self, 'signals'):
            self.signals.toggle_vis.emit()

    def toggle_opacity(self):
        """Flip opaque / translucent. Thread-safe (global hotkey)."""
        if self._is_running and HAS_PYQT and hasattr(self, 'signals'):
            self.signals.toggle_op.emit()

    def _slot_toggle_visible(self):
        logger.info(f"Overlay visibility toggled (was "
                    f"{'visible' if self.window and self.window.isVisible() else 'hidden'})")
        if not self.window:
            return
        if self.window.isVisible():
            self.window.hide()
            if self._setup_dialog:
                self._setup_dialog.hide()
            logger.info("Overlay hidden (panic hotkey)")
        else:
            self.window.show()
            self.window.raise_()
            logger.info("Overlay shown")

    def _toggle_opacity(self):
        """
        Toggle overlay opacity between opaque and translucent.

        The glyph stays put and only the tooltip moves. It used to swap in
        ☀️/🌙 on the first click, which dropped a pair of colour emoji into an
        otherwise monochrome bar — and every other button here names its
        function rather than its state.
        """
        self._opaque = not self._opaque
        if self._opaque:
            self.opacity_btn.setToolTip("Opaque — click to make translucent")
            self.window.setWindowOpacity(self.OPACITY_OPAQUE)
            logger.info("Overlay opacity: opaque")
        else:
            self.opacity_btn.setToolTip("Translucent — click to make opaque")
            self.window.setWindowOpacity(self.OPACITY_TRANSLUCENT)
            logger.info("Overlay opacity: translucent")

    # ────────────────────────────────────────────────
    #  Command bar pieces
    # ────────────────────────────────────────────────
    def _bar_button(self, glyph: str, tooltip: str, handler=None,
                    hover_bg=None, hover_fg=None):
        """One monochrome glyph button — the bar's only button shape."""
        btn = QPushButton(glyph)
        btn.setFixedSize(26, 26)
        btn.setToolTip(tooltip)
        btn.setCursor(QCursor(Qt.ArrowCursor))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 7px;
                color: {T.TEXT_DIM};
                font-family: {T.FONT};
                font-size: 14px;
                padding: 0px;
            }}
            QPushButton:hover   {{ background: {hover_bg or T.HOVER};
                                   color: {hover_fg or T.TEXT}; }}
            QPushButton:pressed {{ background: {T.BORDER}; }}
        """)
        if handler is not None:
            btn.clicked.connect(handler)
        return btn

    def _bar_divider(self):
        """Hairline between the wordmark and the tools."""
        line = QWidget()
        line.setFixedSize(1, 16)
        line.setAttribute(Qt.WA_StyledBackground, True)
        line.setStyleSheet(f"background: {T.BORDER}; border: none;")
        return line

    # ────────────────────────────────────────────────
    #  Where the window was left
    # ────────────────────────────────────────────────
    @staticmethod
    def _geometry_path():
        """Beside the .exe when frozen, or wherever app data actually lands."""
        return file_context.writable_base() / "overlay.json"

    def save_geometry(self):
        """
        Remember where the window is and how big it is.

        The bar is draggable and there are four resize grips, and every launch
        used to throw all of that away and snap back to top-center at the
        default size. Never raises: a settings file that cannot be written is
        not worth losing the overlay over.
        """
        if not self.window or not self._expanded:
            # Collapsed height is the bar, not the size to reopen at.
            return
        try:
            g = self.window.geometry()
            self._geometry_path().write_text(json.dumps(
                {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}),
                encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save the overlay position: {e}")

    def _restore_geometry(self, x, y, w, h):
        """
        (x, y, w, h) to open at — the saved rect if there is a usable one.

        Clamped against the screens that exist now. A rect saved on a second
        monitor that has since been unplugged would otherwise put the window
        somewhere unreachable, and this app has no taskbar entry to get it
        back with.
        """
        try:
            saved = json.loads(self._geometry_path().read_text(encoding="utf-8"))
            sx, sy = int(saved["x"]), int(saved["y"])
            sw, sh = int(saved["w"]), int(saved["h"])
        except FileNotFoundError:
            logger.info("No saved overlay position — opening at the default")
            return x, y, w, h
        except Exception as e:
            logger.warning(f"Saved overlay position unreadable ({e}) — using the default")
            return x, y, w, h

        sw = max(ResizeGrip.MIN_W, sw)
        sh = max(ResizeGrip.MIN_H, sh)

        # The title bar has to land on a screen someone can see.
        rect = QRect(sx, sy, sw, sh)
        if not any(s.availableGeometry().intersects(rect) for s in self.app.screens()):
            logger.info("Saved overlay position is off-screen — using the default")
            return x, y, sw, sh

        area = self.app.primaryScreen().availableGeometry()
        sx = max(area.left(), min(sx, area.right() - 40))
        sy = max(area.top(), min(sy, area.bottom() - 40))
        logger.info(f"Restored overlay geometry: {sw}x{sh} at ({sx}, {sy})")
        return sx, sy, sw, sh

    def _toggle_hotkeys_popover(self):
        """
        Show the hotkey list under the info button, or hide it if it is
        already up. Built fresh each time, so a rebind is reflected without
        anything having to remember to refresh it.
        """
        if self._hotkeys_popover is not None and self._hotkeys_popover.isVisible():
            self._hotkeys_popover.close()
            self._hotkeys_popover = None
            return

        # Qt.Popup closes itself on the next click outside and on Escape.
        pop = QFrame(self.window, Qt.Popup | Qt.FramelessWindowHint)
        pop.setObjectName("hotkeyPopover")
        pop.setAttribute(Qt.WA_StyledBackground, True)
        pop.setStyleSheet(f"""
            #hotkeyPopover {{
                background-color: rgba(24, 24, 27, 0.98);
                border: 1px solid {T.BORDER};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(pop)
        lay.setContentsMargins(12, 10, 12, 11)
        lay.setSpacing(7)

        heading = QLabel("HOTKEYS")
        heading.setStyleSheet(f"color:{T.TEXT_MUTE};font-family:{T.FONT};"
                              "font-size:9px;font-weight:700;letter-spacing:1px;"
                              "background:transparent;border:none;")
        lay.addWidget(heading)

        if not self.hotkeys:
            empty = QLabel("None configured")
            empty.setStyleSheet(f"color:{T.TEXT_DIM};font-family:{T.FONT};"
                                "font-size:12px;background:transparent;border:none;")
            lay.addWidget(empty)

        for label, combo in self.hotkeys:
            row = QHBoxLayout()
            row.setSpacing(18)
            name = QLabel(label)
            name.setStyleSheet(f"color:{T.TEXT_DIM};font-family:{T.FONT};"
                               "font-size:12px;background:transparent;border:none;")
            chip = QLabel(format_combo(combo))
            chip.setStyleSheet(f"""
                QLabel {{
                    color: {T.TEXT};
                    background: {T.BG_RAISED};
                    border: 1px solid {T.BORDER};
                    border-radius: 5px;
                    padding: 2px 7px;
                    font-family: {T.FONT};
                    font-size: 10px;
                    font-weight: 600;
                }}
            """)
            row.addWidget(name)
            row.addStretch()
            row.addWidget(chip)
            lay.addLayout(row)

        pop.adjustSize()
        # Hang it under the info button, nudged left so a wide list stays on
        # screen rather than running off the right edge.
        anchor = self.info_btn.mapToGlobal(self.info_btn.rect().bottomLeft())
        x, y = anchor.x() - 8, anchor.y() + 8
        screen = self.app.primaryScreen().availableGeometry()
        x = max(screen.left() + 4, min(x, screen.right() - pop.width() - 4))
        pop.move(x, y)
        pop.show()
        exclude_from_capture(pop)
        self._hotkeys_popover = pop

    def set_hotkey(self, label: str, combo: str):
        """
        Record a rebound hotkey.

        The popover reads this list when it opens, so there is nothing to
        refresh there; only the chip in the bar holds its own copy.
        """
        for i, (l, _) in enumerate(self.hotkeys):
            if l == label:
                self.hotkeys[i] = (l, combo)
                break
        # The bar shows this one combo in full, so it has to move too.
        if label == "Answer what was just said" and self.ask_hint is not None:
            self.ask_hint.setText(format_combo(combo))

    def _on_close(self):
        logger.info("Close clicked")
        self.save_geometry()
        self.stop()

    # ────────────────────────────────────────────────
    #  Qt slots (execute on GUI thread)
    # ────────────────────────────────────────────────
    def _slot_update_text(self, text: str, append: bool):
        if not self.text_widget:
            return
        sb = self.text_widget.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 50
        if append:
            c = self.text_widget.textCursor()
            c.movePosition(QTextCursor.End)
            c.insertText(text)
        else:
            self.text_widget.setPlainText(text)
            at_bottom = True
        if at_bottom and self.auto_scroll:
            sb.setValue(sb.maximum())

    def _slot_append_html(self, html: str):
        if not self.text_widget:
            return
        sb = self.text_widget.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 50
        c = self.text_widget.textCursor()
        c.movePosition(QTextCursor.End)
        # A block is shared formatting, so an HTML card dropped into the tail
        # of streamed answer text takes that text's format with it — and the
        # text streamed in afterwards would take the card's. Fence it on both
        # sides with clean blocks.
        if c.block().text().strip():
            c.insertBlock(QTextBlockFormat(), QTextCharFormat())
        c.insertHtml(html)
        c.insertBlock(QTextBlockFormat(), QTextCharFormat())
        self.text_widget.setTextCursor(c)
        if at_bottom and self.auto_scroll:
            sb.setValue(sb.maximum())

    def _slot_set_status(self, key: str):
        self._apply_status_style(key)

    # ────────────────────────────────────────────────
    #  Public API (thread-safe — called from bg threads)
    # ────────────────────────────────────────────────
    def init_window(self):
        self._is_running = True
        if HAS_PYQT:
            self._create_window()

    def exec(self):
        if HAS_PYQT and self.app:
            self.app.exec_()

    def set_status(self, key: str):
        """Update status pill. Thread-safe."""
        if not self._is_running:
            return
        if HAS_PYQT and hasattr(self, 'signals'):
            self.signals.set_status.emit(key)

    def update_answer(self, text: str, append: bool = False):
        """Update text area. Thread-safe."""
        if not self._is_running:
            return
        if HAS_PYQT and hasattr(self, 'signals'):
            self.signals.update_text.emit(text, append)

    def append_html(self, html: str):
        """Append HTML block. Thread-safe."""
        if not self._is_running:
            return
        if HAS_PYQT and hasattr(self, 'signals'):
            self.signals.append_html.emit(html)

    def show_question(self, question: str):
        """Show question as a high-contrast frosted glass card."""
        self._question_count += 1
        q_html = f"""
        <div style="
            background: {T.ACCENT_SOFT};
            border-left: 2px solid {T.ACCENT};
            border-radius: 6px;
            padding: 10px 14px;
            margin-top: 14px;
            margin-bottom: 10px;
        ">
            <span style="color: {T.ACCENT}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;">QUESTION {self._question_count}</span><br/>
            <span style="color: {T.TEXT}; font-size: 14px; font-weight: 500; line-height: 1.5;">{question}</span>
        </div>
        """
        self.append_html(q_html)

    def notice(self, message: str):
        """
        A quiet line in the answer panel — status, not answer.

        Lives here so callers say what they mean and the palette stays in
        one file; main.py used to spell out a slate colour that vanished the
        moment the panel went dark.
        """
        self.append_html(
            f'<div style="color:{T.TEXT_MUTE};font-size:12px;'
            f'padding-left:4px;margin:6px 0;">{message}</div>')

    def stream_answer(self, text_chunk: str):
        """Append streaming answer text."""
        self.update_answer(text_chunk, append=True)

    def show_latency(self, latency_ms: float, ttft_ms: float = 0):
        """Show latency footer."""
        info_html = f"""
        <div style="
            color: {T.TEXT_MUTE};
            font-size: 11px;
            font-weight: 500;
            margin-top: 10px;
            margin-bottom: 12px;
            padding-left: 4px;
            padding-top: 6px;
        ">
            {latency_ms:.0f}ms &middot; first token {ttft_ms:.0f}ms
        </div>
        """
        self.append_html(info_html)

    def stop(self):
        self._is_running = False
        if IS_WINDOWS and self._hwnd:
            self._set_wda(False)
            self._set_click_through(False)
        if HAS_PYQT and self.app:
            self.app.quit()
        logger.info("Ghost overlay stopped")