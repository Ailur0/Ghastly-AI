"""
file_context.py — Resume / document uploads that feed the LLM context

The user picks files in the overlay's setup panel; we extract plain text
from each one and keep a copy under context/uploaded/. ContextManager reads
that folder on every load_context(), so an upload takes effect on the next
question without restarting the app.

Supported: .pdf (pypdf), .docx (stdlib zip + XML), and any plain-text file
(.txt/.md/.json/.csv and common source extensions).

Uploads are written next to the executable when frozen, never into the
PyInstaller temp bundle — that directory is deleted when the app exits.
"""

import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-file guard. A resume is a few thousand characters; anything far past
# this is a manual or a book and would only crowd out the question.
MAX_CHARS_PER_FILE = 20000

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".csv", ".tsv", ".log",
    ".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml",
    ".html", ".css", ".xml", ".ini", ".cfg", ".toml",
}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}


_writable_base = None


def _candidate_bases():
    """Where app data could live, best first."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Beside the .exe — sys._MEIPASS is a temp dir wiped on exit.
        yield Path(sys.executable).parent
    else:
        yield Path(__file__).parent

    # The .exe may sit somewhere unwritable: Program Files, a read-only
    # drive, or a folder Defender's controlled-folder-access is guarding.
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local:
        yield Path(local) / "GhastlyAI"
    yield Path(tempfile.gettempdir()) / "GhastlyAI"


def writable_base() -> Path:
    """
    A directory the app can actually write to, resolved once.

    Falls back rather than failing: a user who runs the .exe from a place it
    cannot write to should still get a working app, not a crash.
    """
    global _writable_base
    if _writable_base is not None:
        return _writable_base

    problems = []
    for base in _candidate_bases():
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as e:
            problems.append(f"{base}: {e}")
            continue
        if problems:
            logger.warning(f"Falling back to {base} for app data ("
                           f"{'; '.join(problems)})")
        _writable_base = base
        return base

    # Nothing was writable. Hand back the last candidate; every caller below
    # tolerates the operations failing.
    logger.error(f"No writable location found ({'; '.join(problems)})")
    _writable_base = Path(tempfile.gettempdir()) / "GhastlyAI"
    return _writable_base


def uploads_dir() -> Path:
    """Writable folder for uploaded documents."""
    d = writable_base() / "context" / "uploaded"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create {d}: {e}")
    return d


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "document"


# ────────────────────────────────────────────────
#  Extraction
# ────────────────────────────────────────────────
def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs pypdf (pip install pypdf)")

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise RuntimeError("PDF is password-protected")

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            logger.warning(f"PDF page skipped in {path.name}: {e}")
    text = "\n".join(pages).strip()
    if not text:
        raise RuntimeError("no selectable text (a scanned PDF needs OCR)")
    return text


def _extract_docx(path: Path) -> str:
    """Pull text out of a .docx without a dependency — it is a zip of XML."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    # Paragraph and line breaks become newlines, then drop every other tag.
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise RuntimeError("no text found in document")
    return text


def extract_text(path) -> str:
    """Return the plain text of a supported file. Raises RuntimeError otherwise."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError("file not found")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(p)
    elif suffix == ".docx":
        text = _extract_docx(p)
    elif suffix in TEXT_SUFFIXES or suffix == "":
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise RuntimeError("file is empty")
    elif suffix == ".doc":
        raise RuntimeError("legacy .doc is not readable — save it as .docx or PDF")
    else:
        raise RuntimeError(f"unsupported file type ({suffix or 'no extension'})")

    if len(text) > MAX_CHARS_PER_FILE:
        text = text[:MAX_CHARS_PER_FILE] + "\n[...truncated]"
        logger.warning(f"{p.name} truncated to {MAX_CHARS_PER_FILE} chars")
    return text


# ────────────────────────────────────────────────
#  Upload store
# ────────────────────────────────────────────────
def add_file(path) -> str:
    """
    Extract `path` and store it as an upload. Returns the stored name.
    Raises RuntimeError with a message fit to show the user.
    """
    src = Path(path)
    text = extract_text(src)
    dest = uploads_dir() / (_slugify(src.stem) + ".txt")
    dest.write_text(f"## {src.name}\n\n{text}\n", encoding="utf-8")
    logger.info(f"Upload stored: {dest.name} ({len(text)} chars from {src.name})")
    return dest.name


def _stored_files() -> list:
    """Upload files, oldest first. Never raises — an unreadable folder is
    reported as empty so the setup panel can still open."""
    try:
        return sorted(uploads_dir().glob("*.txt"), key=lambda p: p.stat().st_mtime)
    except Exception as e:
        logger.error(f"Uploads folder unreadable: {e}")
        return []


def list_files() -> list:
    """Stored uploads as (stored_name, original_name, char_count), oldest first."""
    out = []
    for f in _stored_files():
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first = body.split("\n", 1)[0]
        original = first[3:].strip() if first.startswith("## ") else f.stem
        out.append((f.name, original, len(body)))
    return out


def remove_file(stored_name: str) -> bool:
    target = uploads_dir() / stored_name
    try:
        target.unlink()
        logger.info(f"Upload removed: {stored_name}")
        return True
    except OSError as e:
        logger.error(f"Could not remove {stored_name}: {e}")
        return False


def clear_files() -> int:
    n = 0
    for f in _stored_files():
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    logger.info(f"Cleared {n} upload(s)")
    return n


# Documents whose name looks like a resume go into the context first, so a
# tight character cap eats the notes rather than the candidate's history.
RESUME_HINTS = ("resume", "cv", "curriculum", "profile")


def documents() -> list:
    """Uploads as (original_name, text), resumes first then oldest-first."""
    items = []
    for f in _stored_files():
        try:
            body = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            logger.error(f"Could not read upload {f.name}: {e}")
            continue
        if not body:
            continue
        first = body.split("\n", 1)[0]
        name = first[3:].strip() if first.startswith("## ") else f.stem
        items.append((name, body))
    # Stable sort keeps upload order within each group.
    items.sort(key=lambda it: 0 if any(h in it[0].lower() for h in RESUME_HINTS) else 1)
    return items


def combined_text() -> str:
    """All uploads concatenated, resumes first."""
    return "\n\n".join(text for _, text in documents())
