"""
ghost_overlay.py — Cluely-Inspired Frosted Glass Overlay

Bright, clean, cloud-native aesthetic with:
  - Frosted glass command bar (draggable, collapsible)
  - Sun/moon icon toggles opacity (opaque / translucent)
  - Animated status pill (Ready / Listening / Transcribing / Answering / Error)
  - Scrollable answer panel with glass Q&A cards
  - Sky-blue accent color throughout

The cursor is pinned to a plain arrow over the entire overlay — hovering
buttons, dragging the bar, and the text panel all keep the default shape,
since WDA hides this window's pixels but not the OS cursor sprite.

On Windows:
  - WDA_EXCLUDEFROMCAPTURE is always active (invisible to screen capture)
"""

import sys
import os
import logging
import ctypes
import random

import file_context

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == 'win32'
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_NONE = 0x00000000
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020

# ── PyQt5 imports ──
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QTextEdit, QFrame, QGraphicsDropShadowEffect, QPushButton,
        QDialog, QListWidget, QListWidgetItem, QComboBox,
        QFileDialog, QLineEdit
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent
    from PyQt5.QtGui import (
        QColor, QTextCursor, QCursor, QPainter, QPen, QBrush
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 not installed")


# ════════════════════════════════════════════════════════════════
#  Screen-capture exclusion
# ════════════════════════════════════════════════════════════════
def exclude_from_capture(widget) -> bool:
    """
    Hide one top-level window's pixels from screen capture.

    WDA_EXCLUDEFROMCAPTURE applies per HWND, so every window the app puts on
    screen needs its own call — the affinity set on the overlay does not
    inherit to anything else.
    """
    if not IS_WINDOWS:
        return False
    try:
        ctypes.windll.user32.SetWindowDisplayAffinity(
            int(widget.winId()), WDA_EXCLUDEFROMCAPTURE)
        return True
    except Exception as e:
        logger.error(f"Capture exclusion failed for {widget}: {e}")
        return False


def exclude_process_windows() -> int:
    """
    Exclude every top-level window this process owns, Qt's or not.

    Windows puts its own drop-shadow window (class SysShadow) behind tooltips
    and menus. It belongs to us but is never created through Qt, so no event
    filter can see it — leaving a shadow-shaped box visible in a screen share
    exactly where the tooltip was, with the text correctly hidden inside it.

    Returns how many windows this call had to fix.
    """
    if not IS_WINDOWS:
        return 0

    u32 = ctypes.windll.user32
    pid_here = ctypes.windll.kernel32.GetCurrentProcessId()
    fixed = 0

    def visit(hwnd, _):
        nonlocal fixed
        pid = ctypes.c_ulong()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != pid_here or not u32.IsWindowVisible(hwnd):
            return True
        affinity = ctypes.c_ulong()
        if (u32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
                and affinity.value != WDA_EXCLUDEFROMCAPTURE):
            if u32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                fixed += 1
        return True

    try:
        callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                      ctypes.c_void_p)(visit)
        u32.EnumWindows(callback, 0)
    except Exception as e:
        logger.error(f"Window sweep failed: {e}")
    return fixed


if HAS_PYQT:
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
                exclude_from_capture(obj)
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

        def mouseReleaseEvent(self, event):
            self._drag_pos = None
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
            p.setPen(QPen(QColor(2, 132, 199, 220), 1.4))
            p.setBrush(QBrush(QColor(255, 255, 255, 235)))
            p.drawRoundedRect(box, 3, 3)
            # tiny cross inside, so it reads as a handle and not a bullet
            p.setPen(QPen(QColor(2, 132, 199, 170), 1.2))
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

        def mouseReleaseEvent(self, event):
            self._press_global = None
            event.accept()


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

        LABEL_CSS = ("color:#0369A1;font-family:'Segoe UI',sans-serif;"
                     "font-size:11px;font-weight:700;letter-spacing:0.6px;"
                     "background:transparent;border:none;")
        BTN_CSS = """
            QPushButton {
                background: rgba(2,132,199,0.10);
                color: #0F172A;
                border: 1px solid rgba(2,132,199,0.35);
                border-radius: 7px;
                padding: 6px 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover  { background: rgba(2,132,199,0.20); }
            QPushButton:disabled { color:#94A3B8; border-color:rgba(148,163,184,0.4); }
        """

        def __init__(self, owner, on_changed, parent=None):
            super().__init__(parent)
            self.owner = owner                    # the GhostOverlay
            self.on_changed = on_changed          # (kind, value) -> None
            languages, current_language = owner.languages, owner.code_language
            styles, current_style = owner.answer_styles, owner.answer_style
            self.setWindowTitle("Ghastly AI — Setup")
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint
                                | Qt.WindowStaysOnTopHint | Qt.Tool)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setFixedWidth(400)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(10, 10, 10, 10)

            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: rgba(255,255,255,0.97);
                    border: 1.5px solid rgba(203,213,225,0.9);
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
            title.setStyleSheet("color:#0F172A;font-family:'Segoe UI',sans-serif;"
                                "font-size:14px;font-weight:700;background:transparent;border:none;")
            hl.addWidget(title)
            hl.addStretch()
            close_btn = QPushButton("✕")
            close_btn.setFixedSize(20, 20)
            close_btn.setStyleSheet("""
                QPushButton { background:transparent;border:none;color:#64748B;
                              font-size:13px;font-weight:700;border-radius:5px; }
                QPushButton:hover { background:rgba(239,68,68,0.15);color:#DC2626; }
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
                    background: rgba(241,245,249,0.85);
                    border: 1px solid rgba(203,213,225,0.9);
                    border-radius: 8px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    color: #0F172A;
                    padding: 4px;
                }
                QListWidget::item { padding: 3px 4px; border-radius: 4px; }
                QListWidget::item:selected { background: rgba(2,132,199,0.18); color:#0F172A; }
            """)
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
            self.status.setStyleSheet("color:#334155;font-family:'Segoe UI',sans-serif;"
                                      "font-size:11px;background:transparent;border:none;")
            lay.addWidget(self.status)

            # ── answer language ──
            lang_label = QLabel("ANSWER LANGUAGE")
            lang_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(lang_label)

            self.lang_combo = QComboBox()
            self.lang_combo.addItems(languages)
            if current_language in languages:
                self.lang_combo.setCurrentText(current_language)
            self.lang_combo.setStyleSheet("""
                QComboBox {
                    background: rgba(241,245,249,0.85);
                    border: 1px solid rgba(203,213,225,0.9);
                    border-radius: 8px;
                    padding: 5px 8px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    color: #0F172A;
                }
                QComboBox::drop-down { border: none; width: 18px; }
                QComboBox QAbstractItemView {
                    background: #FFFFFF;
                    border: 1px solid rgba(203,213,225,0.9);
                    selection-background-color: rgba(2,132,199,0.18);
                    selection-color: #0F172A;
                    color: #0F172A;
                    outline: none;
                }
            """)
            self.lang_combo.currentTextChanged.connect(self._language_changed)
            lay.addWidget(self.lang_combo)

            hint = QLabel("Auto follows whatever the question implies.")
            hint.setStyleSheet("color:#64748B;font-family:'Segoe UI',sans-serif;"
                               "font-size:11px;background:transparent;border:none;")
            lay.addWidget(hint)

            # ── answer style ──
            style_label = QLabel("ANSWER STYLE")
            style_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(style_label)

            self.style_combo = QComboBox()
            self.style_combo.addItems(styles)
            if current_style in styles:
                self.style_combo.setCurrentText(current_style)
            self.style_combo.setStyleSheet(self.lang_combo.styleSheet())
            self.style_combo.currentTextChanged.connect(self._style_changed)
            lay.addWidget(self.style_combo)

            self.style_hint = QLabel("")
            self.style_hint.setWordWrap(True)
            self.style_hint.setStyleSheet("color:#64748B;font-family:'Segoe UI',sans-serif;"
                                          "font-size:11px;background:transparent;border:none;")
            lay.addWidget(self.style_hint)
            self._describe_style(self.style_combo.currentText())

            # ── audio device ──
            audio_label = QLabel("AUDIO SOURCE")
            audio_label.setStyleSheet(self.LABEL_CSS)
            lay.addWidget(audio_label)

            self.audio_combo = QComboBox()
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
            audio_hint.setStyleSheet("color:#64748B;font-family:'Segoe UI',sans-serif;"
                                     "font-size:11px;background:transparent;border:none;")
            lay.addWidget(audio_hint)

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
            self.remove_btn.setEnabled(bool(entries))
            self.clear_btn.setEnabled(bool(entries))

        def _pick_files(self):
            try:
                self._pick_files_inner()
            except Exception as e:
                logger.exception("File picker failed")
                self.status.setText(f"Could not add files: {e}")

        def _pick_files_inner(self):
            dlg = QFileDialog(self, "Add resume or notes")
            dlg.setFileMode(QFileDialog.ExistingFiles)
            # Qt's own dialog, not the native one — see the class docstring.
            dlg.setOption(QFileDialog.DontUseNativeDialog, True)
            dlg.setNameFilter("Documents (*.pdf *.docx *.txt *.md *.json *.csv);;All files (*)")
            dlg.show()
            exclude_from_capture(dlg)
            if not dlg.exec_():
                return

            added, errors = 0, []
            for path in dlg.selectedFiles():
                try:
                    file_context.add_file(path)
                    added += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")

            self.refresh_files()
            if added:
                self.on_changed("files", added)
            msg = f"Added {added} file{'s' if added != 1 else ''}." if added else ""
            if errors:
                msg = (msg + "  " if msg else "") + "Skipped — " + "; ".join(errors)
            self.status.setText(msg)

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
            self._describe_style(text)
            self.on_changed("style", text)
            self.status.setText(f"Answer style: {text}.")

        def _audio_changed(self, index):
            device_id = self.audio_combo.itemData(index)
            self.on_changed("audio_device", device_id)
            self.status.setText("Audio source: {}".format(self.audio_combo.currentText()))

        def _language_changed(self, text):
            self.on_changed("language", text)
            self.status.setText(f"Code answers will use {text}."
                                if text != "Auto" else
                                "Code answers follow the question.")


# ════════════════════════════════════════════════════════════════
#  Status definitions
# ════════════════════════════════════════════════════════════════
STATUS_MAP = {
    "ready":        ("⚡ Ready",        "#0EA5E9", "rgba(14,165,233,0.12)", "rgba(14,165,233,0.28)"),
    "listening":    ("🎤 Listening",    "#0EA5E9", "rgba(14,165,233,0.12)", "rgba(14,165,233,0.28)"),
    "transcribing": ("✨ Transcribing", "#0EA5E9", "rgba(14,165,233,0.12)", "rgba(14,165,233,0.28)"),
    "answering":    ("⚡ Answering",    "#0EA5E9", "rgba(14,165,233,0.12)", "rgba(14,165,233,0.28)"),
    "error":        ("⚠ Error",        "#EF4444", "rgba(239,68,68,0.12)",  "rgba(239,68,68,0.28)"),
    "offline":      ("🔴 Offline",     "#EF4444", "rgba(239,68,68,0.12)",  "rgba(239,68,68,0.28)"),
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
        self.title_label = None
        self._scramble_timer = None

        self._grips = {}
        self._expanded_h = None
        self._setup_dialog = None
        self._capture_shield = None
        self._sweep_timer = None

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
            return
        try:
            flag = WDA_EXCLUDEFROMCAPTURE if exclude else WDA_NONE
            ctypes.windll.user32.SetWindowDisplayAffinity(self._hwnd, flag)
            logger.info(f"WDA {'EXCLUDE' if exclude else 'NONE'}")
        except Exception as e:
            logger.error(f"WDA error: {e}")

    def _set_click_through(self, enabled: bool):
        if not IS_WINDOWS or not self._hwnd:
            return
        try:
            style = ctypes.windll.user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
            if enabled:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, style)
        except Exception as e:
            logger.error(f"Click-through error: {e}")

    # ────────────────────────────────────────────────
    #  Build the UI
    # ────────────────────────────────────────────────
    def _create_window(self):
        self.app = QApplication.instance() or QApplication(sys.argv)

        # Catch every window the app opens — tooltips and dropdowns included.
        self._capture_shield = CaptureShield()
        self.app.installEventFilter(self._capture_shield)

        # Backstop: anything the event filter and its sweeps still miss gets
        # picked up within half a second. Cheap — EnumWindows over one process.
        self._sweep_timer = QTimer()
        self._sweep_timer.timeout.connect(exclude_process_windows)
        self._sweep_timer.start(500)

        total_w = max(self.BAR_W, self.PANEL_W) + 24
        total_h = self.BAR_H + self.PANEL_H + 32

        self.window = QWidget()
        self.window.setWindowTitle("Ghastly AI")
        self.window.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.window.setAttribute(Qt.WA_TranslucentBackground, True)

        screen = self.app.primaryScreen().geometry()
        x, y = self._get_position(screen.width(), screen.height())
        self.window.setGeometry(x, y, total_w, total_h)

        # ── Root layout ──
        root = QVBoxLayout(self.window)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(4)

        # ═══ COMMAND BAR ═══
        self.bar = DraggableWidget()
        self.bar.setFixedHeight(self.BAR_H)
        self.bar.setStyleSheet("""
            QWidget {{
                background-color: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(255, 255, 255, 0.35);
                border-radius: 12px;
            }}
        """)

        bar_shadow = QGraphicsDropShadowEffect()
        bar_shadow.setBlurRadius(32)
        bar_shadow.setColor(QColor(0, 0, 0, 30))
        bar_shadow.setOffset(0, 8)
        self.bar.setGraphicsEffect(bar_shadow)

        bar_layout = QHBoxLayout(self.bar)
        bar_layout.setContentsMargins(10, 0, 8, 0)
        bar_layout.setSpacing(8)

        # Opacity toggle button (sun = opaque, moon = translucent)
        self.opacity_btn = QPushButton("☀️")
        self.opacity_btn.setFixedSize(28, 28)
        self.opacity_btn.setToolTip("Opaque — click to make translucent")
        self.opacity_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(14, 165, 233, 0.12);
            }
        """)
        self.opacity_btn.clicked.connect(self._toggle_opacity)
        bar_layout.addWidget(self.opacity_btn)

        # Info button (hover tooltip lists hotkeys)
        self.info_btn = QPushButton("ℹ️")
        self.info_btn.setFixedSize(28, 28)
        self.info_btn.setToolTip(self._build_hotkeys_tooltip())
        self.info_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(14, 165, 233, 0.12);
            }
        """)
        bar_layout.addWidget(self.info_btn)

        # Setup button (documents + answer language)
        self.setup_btn = QPushButton("📎")
        self.setup_btn.setFixedSize(28, 28)
        self.setup_btn.setToolTip("Documents & answer language")
        self.setup_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                font-size: 15px;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(14, 165, 233, 0.12);
            }
        """)
        self.setup_btn.clicked.connect(self._open_setup)
        bar_layout.addWidget(self.setup_btn)

        # Retry button — re-answers the last question
        self.retry_btn = QPushButton("↻")
        self.retry_btn.setFixedSize(28, 28)
        self.retry_btn.setToolTip("Answer the last question again")
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                color: #0F172A;
                font-size: 15px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton:hover { background: rgba(14, 165, 233, 0.12); }
        """)
        self.retry_btn.clicked.connect(self._on_retry_clicked)
        bar_layout.addWidget(self.retry_btn)

        # Title
        self.title_label = QLabel("Ghastly AI")
        self.title_label.setStyleSheet("""
            color: #0F172A;
            font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
            font-size: 14px;
            font-weight: 700;
            background: transparent;
            border: none;
        """)
        bar_layout.addWidget(self.title_label)

        bar_layout.addStretch()

        # Status pill
        self.status_pill = QLabel("⚡ Ready")
        self._apply_status_style("ready")
        bar_layout.addWidget(self.status_pill)

        bar_layout.addSpacing(6)

        # Window controls
        for color, hover, tip, handler in [
            ("#FF5F57", "#FF3B30", "Close", self._on_close),
        ]:
            btn = QPushButton()
            btn.setFixedSize(14, 14)
            btn.setToolTip(tip)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: none;
                    border-radius: 7px;
                }}
                QPushButton:hover {{
                    background-color: {hover};
                }}
            """)
            btn.clicked.connect(handler)
            bar_layout.addWidget(btn)

        # Click on bar toggles expand/collapse
        self.bar.mouseDoubleClickEvent = lambda e: self._toggle_panel()

        root.addWidget(self.bar)

        # ═══ ANSWER PANEL ═══
        self.panel = QFrame()
        self.panel.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.95);
                border: 1.5px solid rgba(203, 213, 225, 0.85);
                border-radius: 14px;
            }
        """)

        panel_shadow = QGraphicsDropShadowEffect()
        panel_shadow.setBlurRadius(32)
        panel_shadow.setColor(QColor(0, 0, 0, 45))
        panel_shadow.setOffset(0, 8)
        self.panel.setGraphicsEffect(panel_shadow)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)

        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        self.text_widget.viewport().setCursor(QCursor(Qt.ArrowCursor))
        self.text_widget.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #020617;
                font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
                font-size: 16px;
                font-weight: 500;
                line-height: 1.6;
                border: none;
                selection-background-color: rgba(14, 165, 233, 0.35);
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.20);
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 0.40);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        panel_layout.addWidget(self.text_widget)

        # ── Ask box ──
        # The whole pipeline depends on the interviewer's audio reaching a
        # loopback device. When that fails — wrong device, muted call, a term
        # Whisper mangles — this is the way back in.
        self.ask_input = QLineEdit()
        self.ask_input.setPlaceholderText("Type a question and press Enter…")
        self.ask_input.setFixedHeight(30)
        self.ask_input.setStyleSheet("""
            QLineEdit {
                background: rgba(241, 245, 249, 0.9);
                border: 1px solid rgba(203, 213, 225, 0.9);
                border-radius: 8px;
                padding: 4px 10px;
                font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
                font-size: 13px;
                color: #0F172A;
                selection-background-color: rgba(14, 165, 233, 0.35);
            }
            QLineEdit:focus { border: 1px solid rgba(2, 132, 199, 0.65); }
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
                border-radius: 10px;
                padding: 3px 10px;
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                font-size: 11px;
                font-weight: 600;
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
            self._build_and_show_setup()
        except Exception as e:
            logger.exception("Could not open the setup panel")
            self._setup_dialog = None
            self.append_html(
                '<div style="color:#DC2626;font-size:12px;padding-left:4px;">'
                f'Setup panel unavailable: {e}</div>')

    def _build_and_show_setup(self):
        if self._setup_dialog is None:
            self._setup_dialog = SetupDialog(self, self._on_setup_changed,
                                             parent=self.window)
        else:
            self._setup_dialog.refresh_files()

        geo = self.window.geometry()
        self._setup_dialog.move(geo.x(), geo.y() + geo.height() + 8)
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

    def _slot_toggle_visible(self):
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
        """Toggle overlay opacity between opaque and translucent (sun/moon icon)."""
        self._opaque = not self._opaque
        if self._opaque:
            self.opacity_btn.setText("☀️")
            self.opacity_btn.setToolTip("Opaque — click to make translucent")
            self.window.setWindowOpacity(self.OPACITY_OPAQUE)
            logger.info("Overlay opacity: opaque")
        else:
            self.opacity_btn.setText("🌙")
            self.opacity_btn.setToolTip("Translucent — click to make opaque")
            self.window.setWindowOpacity(self.OPACITY_TRANSLUCENT)
            logger.info("Overlay opacity: translucent")

    def _build_hotkeys_tooltip(self) -> str:
        """Build the info button's tooltip text listing all configured hotkeys."""
        if not self.hotkeys:
            return "No hotkeys configured"
        lines = ["Hotkeys:"]
        for label, combo in self.hotkeys:
            formatted = "+".join(part.capitalize() for part in combo.split("+"))
            lines.append(f"{label} — {formatted}")
        return "\n".join(lines)

    def _on_close(self):
        logger.info("Close clicked")
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
        c.insertHtml(html)
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
            background: rgba(2, 132, 199, 0.08);
            border-left: 4px solid #0284C7;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 14px;
            margin-bottom: 8px;
        ">
            <span style="color: #0369A1; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">QUESTION #{self._question_count}</span><br/>
            <span style="color: #0F172A; font-size: 15px; font-weight: 600; line-height: 1.5;">{question}</span>
        </div>
        <div style="color: #0284C7; font-size: 14px; font-weight: 700; padding-left: 4px; margin-bottom: 4px;">
            Answer
        </div>
        """
        self.append_html(q_html)

    def stream_answer(self, text_chunk: str):
        """Append streaming answer text."""
        self.update_answer(text_chunk, append=True)

    def show_latency(self, latency_ms: float, ttft_ms: float = 0):
        """Show latency footer."""
        info_html = f"""
        <div style="
            color: #334155;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
            margin-bottom: 12px;
            padding-left: 4px;
            border-top: 1px solid rgba(203, 213, 225, 0.7);
            padding-top: 6px;
        ">
            ⏱ {latency_ms:.0f}ms &middot; TTFT {ttft_ms:.0f}ms
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