"""
ghost_overlay.py — Cluely-Inspired Frosted Glass Overlay

Bright, clean, cloud-native aesthetic with:
  - Frosted glass command bar (draggable, collapsible)
  - Sun/moon icon toggles opacity (opaque / translucent)
  - Animated status pill (Ready / Listening / Transcribing / Answering / Error)
  - Scrollable answer panel with glass Q&A cards
  - Sky-blue accent color throughout

On Windows:
  - WDA_EXCLUDEFROMCAPTURE is always active (invisible to screen capture)
  - Cursor is forced to a plain arrow so hovering doesn't reveal the overlay
    in a screen capture (WDA hides this window's pixels, not the OS cursor)
"""

import sys
import os
import threading
import time
import logging
import ctypes
import random
from typing import Optional

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
        QSizePolicy
    )
    from PyQt5.QtCore import (
        Qt, QTimer, pyqtSignal, QObject, QPoint, QPropertyAnimation,
        QEasingCurve, QSize
    )
    from PyQt5.QtGui import QFont, QColor, QTextCursor, QCursor
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 not installed")


# ════════════════════════════════════════════════════════════════
#  Signals (thread-safe bridge from background threads → Qt GUI)
# ════════════════════════════════════════════════════════════════
if HAS_PYQT:
    class OverlaySignals(QObject):
        update_text = pyqtSignal(str, bool)      # (text, append)
        append_html = pyqtSignal(str)             # html block
        set_status  = pyqtSignal(str)             # status key

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

        self.app = None
        self.window = None
        self.bar = None
        self.panel = None
        self.text_widget = None
        self.status_pill = None
        self.opacity_btn = None
        self.title_label = None
        self._scramble_timer = None

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
        self.bar.setCursor(QCursor(Qt.OpenHandCursor))
        self.bar.setStyleSheet(f"""
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
        self.opacity_btn.setCursor(QCursor(Qt.PointingHandCursor))
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
            ("#FEBC2E", "#F0A500", "Minimize", self._on_minimize),
            ("#FF5F57", "#FF3B30", "Close",    self._on_close),
        ]:
            btn = QPushButton()
            btn.setFixedSize(14, 14)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
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

        root.addWidget(self.panel)

        # ── Show ──
        self.window.setWindowOpacity(self.OPACITY_OPAQUE)
        self.window.show()

        if IS_WINDOWS:
            self._hwnd = int(self.window.winId())
            # Always excluded from screen capture — not a toggle.
            self._set_wda(True)
            # Force a plain arrow cursor everywhere over the overlay —
            # WDA_EXCLUDEFROMCAPTURE hides this window's pixels from screen
            # capture, but NOT the OS mouse cursor sprite, so a widget-specific
            # cursor (e.g. the drag bar's OpenHandCursor) would otherwise
            # reveal that something interactive is here.
            QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))

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
    def _toggle_panel(self):
        """Expand / collapse the answer panel."""
        self._expanded = not self._expanded
        self.panel.setVisible(self._expanded)
        # Resize window
        if self._expanded:
            h = self.BAR_H + self.PANEL_H + 32
        else:
            h = self.BAR_H + 28
        w = self.window.width()
        self.window.setFixedHeight(h)
        self.window.resize(w, h)
        logger.info(f"Panel {'expanded' if self._expanded else 'collapsed'}")

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

    def _on_minimize(self):
        if self.window:
            self.window.showMinimized()

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
        if at_bottom:
            sb.setValue(sb.maximum())

    def _slot_append_html(self, html: str):
        if not self.text_widget:
            return
        sb = self.text_widget.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 50
        c = self.text_widget.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertHtml(html)
        if at_bottom:
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