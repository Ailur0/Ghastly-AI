"""
context_manager.py — Context and state management for the interview agent

Handles:
1. Loading static context (interview-context.md) at startup — kept in memory
2. Reading/updating dynamic state (interview-state.json) per question
3. Detecting interviewer mood and persona from question text
4. Building the context string for the LLM prompt

Designed for speed: file I/O is minimal (<5ms), 
context string building is pure concatenation (<1ms).
"""

import json
import os
import sys
import time
import logging
from typing import Tuple

import file_context

logger = logging.getLogger(__name__)

# Mood detection keywords
MOOD_KEYWORDS = {
    "aggressive": ["why did you", "that's wrong", "that's not", "no that's", 
                   "i don't agree", "that doesn't make sense", "are you sure"],
    "friendly": ["great", "interesting", "nice", "good answer", "i like",
                 "that's good", "awesome", "well done"],
    "confused": ["i don't understand", "can you clarify", "what do you mean",
                 "i'm not sure i follow", "could you elaborate"],
    "impressed": ["excellent", "impressive", "wow", "that's exactly",
                  "perfect answer", "spot on"],
    "skeptical": ["really", "are you sure", "is that right", "i doubt",
                  "that seems unlikely"],
}

# Persona detection keywords
PERSONA_KEYWORDS = {
    "technical": ["implement", "code", "algorithm", "system design", "architecture",
                  "optimize", "debug", "complexity", "data structure", "api",
                  "database", "deploy", "scalable", "latency", "throughput"],
    "behavioral": ["tell me about a time", "how did you handle", "describe a situation",
                   "what would you do if", "give me an example", "conflict",
                   "team", "leadership", "challenge", "failure"],
    "casual": ["tell me about yourself", "why this company", "where do you see",
               "what are your hobbies", "why should we hire"],
    "managerial": ["how would you", "what's your approach to", "how do you manage",
                   "how do you prioritize", "how would you handle"],
}


from pathlib import Path

def resolve_writable_path(path_str: str) -> str:
    """
    Where app data that gets written back belongs.

    A frozen build must write beside the .exe: sys._MEIPASS is a temp folder
    Windows deletes on exit, so state saved there is gone the moment the app
    closes — which is what used to happen to the language and answer-style
    choices.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return str(Path(sys.executable).parent / path_str)
    return path_str


def resolve_app_path(path_str: str) -> str:
    """Resolve path relative to executable or base directory for PyInstaller compatibility."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        exe_dir = Path(sys.executable).parent
        target = exe_dir / path_str
        if target.exists():
            return str(target)
        base_dir = Path(sys._MEIPASS)
        return str(base_dir / path_str)
    return path_str

class ContextManager:
    def __init__(self, context_file="context/interview-context.md",
                 state_file="context/interview-state.json",
                 max_context_chars=8000):
        self.context_file = resolve_app_path(context_file)
        # State is written back, so it cannot live in the read-only bundle.
        self.state_file = resolve_writable_path(state_file)
        self.bundled_state_file = resolve_app_path(state_file)
        self.max_context_chars = max_context_chars
        
        # Static context loaded once at startup
        self.static_context = ""
        
        # Dynamic state kept in memory, persisted to file
        self.state = {
            "questions_asked": [],
            "answers_given": [],
            "interviewer_mood": "neutral",
            "interviewer_persona": "unknown",
            "current_topic": "intro",
            "question_count": 0,
            "session_start": None,
            "last_question_time": None,
            # Language the candidate wants code answers in; "Auto" follows
            # whatever the question implies.
            "code_language": "Auto",
            # Shape of the answer: Balanced / Snippet only / Text only /
            # Full walkthrough.
            "answer_style": "Balanced",
            # Capture device id, or "Auto" to let the capture layer pick.
            "audio_device": "Auto"
        }
    
    def load_context(self):
        """Load static context from MD file. Call ONCE at startup."""
        try:
            # utf-8, not the ANSI default: a resume pasted in here is full
            # of curly quotes and dashes that cp1252 chokes on.
            with open(self.context_file, 'r', encoding='utf-8', errors='replace') as f:
                self.static_context = f.read().strip()
            
            # Trim if too long
            if len(self.static_context) > self.max_context_chars:
                self.static_context = self.static_context[:self.max_context_chars]
                logger.warning(f"Context truncated to {self.max_context_chars} chars")
            
            logger.info(f"Loaded context: {len(self.static_context)} chars")
        except FileNotFoundError:
            logger.info(f"No context file at {self.context_file}")
            self.static_context = ""

        self._append_uploads()

        if not self.static_context:
            self.static_context = "No context available."
    
    def _append_uploads(self):
        """
        Fold uploaded documents into the static context, resumes first.

        Packs one document at a time so a tight cap leaves out the least
        important file whole, rather than slicing the tail off whichever one
        happened to be last — which used to cut a resume in half mid-sentence.
        """
        try:
            docs = file_context.documents()
        except Exception as e:
            logger.error(f"Could not read uploads: {e}")
            return
        if not docs:
            return

        header, gap = "Candidate documents:\n", "\n\n"
        base = self.static_context
        budget = self.max_context_chars - len(header) - (len(base) + len(gap) if base else 0)
        if budget <= 0:
            logger.warning("Context cap leaves no room for uploaded documents")
            return

        kept, dropped = [], []
        for name, text in docs:
            need = len(text) + (len(gap) if kept else 0)
            if need <= budget:
                kept.append(text)
                budget -= need
            elif not kept and budget > 500:
                # Nothing fits whole and this is the most important document,
                # so take as much of its head as there is room for.
                kept.append(text[:budget - 20] + "\n[...truncated]")
                dropped.append(f"{name} (truncated)")
                budget = 0
            else:
                dropped.append(name)

        if not kept:
            return

        block = header + gap.join(kept)
        self.static_context = base + gap + block if base else block

        if dropped:
            logger.warning(f"Context cap {self.max_context_chars} chars — "
                           f"left out: {', '.join(dropped)}")
        logger.info(f"Context with uploads: {len(self.static_context)} chars")

    def reload_context(self):
        """Re-read context + uploads after the user adds or removes a file."""
        self.load_context()
        return len(self.static_context)

    def set_code_language(self, language: str):
        """Set the language code answers should be written in."""
        self.state["code_language"] = language or "Auto"
        self.save_state()
        logger.info(f"Code language set to {self.state['code_language']}")

    def get_code_language(self) -> str:
        return self.state.get("code_language", "Auto")

    def set_answer_style(self, style: str):
        """Set how much of the answer is code vs. spoken explanation."""
        self.state["answer_style"] = style or "Balanced"
        self.save_state()
        logger.info(f"Answer style set to {self.state['answer_style']}")

    def get_answer_style(self) -> str:
        return self.state.get("answer_style", "Balanced")

    def set_audio_device(self, device_id: str):
        """Remember which capture device the user picked."""
        self.state["audio_device"] = device_id or "Auto"
        self.save_state()
        logger.info(f"Audio device set to {self.state['audio_device']}")

    def get_audio_device(self) -> str:
        return self.state.get("audio_device", "Auto")

    def load_state(self):
        """
        Load state from JSON. Call at startup.

        Falls back to the copy inside the bundle on first run, so a build that
        ships a pre-seeded state still picks it up before anything is saved
        beside the .exe.
        """
        for path in (self.state_file, self.bundled_state_file):
            try:
                # utf-8-sig tolerates a BOM, which Notepad and PowerShell
                # both add — this file sits next to the .exe where people
                # can edit it, and a BOM used to read as corruption.
                with open(path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                logger.warning(f"State file corrupted, ignoring: {path}")
                continue

            # Merge, so a state file written by an older build keeps the keys
            # it never knew about instead of dropping them.
            self.state.update(loaded)
            logger.info(f"Loaded state from {path}: "
                        f"{self.state.get('question_count', 0)} previous questions, "
                        f"language={self.get_code_language()}, "
                        f"style={self.get_answer_style()}")
            return

        logger.info("No state file found, starting fresh")
    
    def save_state(self):
        """Persist state to JSON file. Call after each question."""
        try:
            parent = os.path.dirname(os.path.abspath(self.state_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def detect_mood(self, text: str) -> str:
        """Detect interviewer mood from their question text."""
        text_lower = text.lower()
        
        for mood, keywords in MOOD_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return mood
        
        return self.state.get("interviewer_mood", "neutral")
    
    def detect_persona(self, text: str) -> str:
        """Detect interviewer style/persona from question text."""
        text_lower = text.lower()
        
        for persona, keywords in PERSONA_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return persona
        
        return self.state.get("interviewer_persona", "technical")
    
    def detect_topic(self, text: str) -> str:
        """Detect current interview topic from question."""
        text_lower = text.lower()
        
        topics = {
            "intro": ["yourself", "background", "experience", "who are you"],
            "ml": ["machine learning", "model", "training", "inference", "neural",
                   "tensorflow", "pytorch", "deep learning", "accuracy", "loss"],
            "mlops": ["mlops", "deployment", "pipeline", "ci/cd", "mlflow",
                      "docker", "kubernetes", "monitoring"],
            "python": ["python", "gIL", "decorator", "generator", "async",
                       "concurrent", "multiprocessing"],
            "database": ["sql", "nosql", "database", "query", "index",
                         "normalization", "join", "mongodb", "postgres"],
            "api": ["api", "rest", "fastapi", "flask", "endpoint", "http",
                    "authentication", "rate limit"],
            "system_design": ["design", "scalable", "architecture", "distributed",
                              "microservice", "load balanc", "cache", "queue"],
            "behavioral": ["team", "conflict", "failure", "challenge", "leadership",
                           "time management", "priority"],
            "data": ["data", "pandas", "numpy", "etl", "data pipeline",
                     "preprocessing", "feature engineering"],
            "rag": ["rag", "retrieval", "embedding", "vector", "langchain",
                    "llm", "document"],
        }
        
        for topic, keywords in topics.items():
            for kw in keywords:
                if kw in text_lower:
                    return topic
        
        return "general"
    
    def add_qa(self, question: str, answer: str):
        """
        Add a Q&A pair to state and persist.
        Call after each completed answer.
        """
        now = time.time()
        
        self.state["questions_asked"].append(question)
        self.state["answers_given"].append(answer)
        self.state["interviewer_mood"] = self.detect_mood(question)
        self.state["interviewer_persona"] = self.detect_persona(question)
        self.state["current_topic"] = self.detect_topic(question)
        self.state["question_count"] += 1
        
        if self.state["session_start"] is None:
            self.state["session_start"] = now
        
        self.state["last_question_time"] = now
        
        # Persist to file
        self.save_state()
        
        logger.info(f"Q&A added: #{self.state['question_count']} | "
                      f"mood={self.state['interviewer_mood']} | "
                      f"persona={self.state['interviewer_persona']} | "
                      f"topic={self.state['current_topic']}")
    
    def get_context_string(self) -> str:
        """
        Get the full context string for the LLM prompt.
        This is what gets prepended to the question in the prompt.
        """
        return self.static_context
    
    def get_state(self) -> dict:
        """Get current state dict for prompt building."""
        return self.state
    
    def reset_state(self):
        """Reset state for a new interview session, keeping preferences."""
        language = self.state.get("code_language", "Auto")
        style = self.state.get("answer_style", "Balanced")
        device = self.state.get("audio_device", "Auto")
        self.state = {
            "questions_asked": [],
            "answers_given": [],
            "interviewer_mood": "neutral",
            "interviewer_persona": "unknown",
            "current_topic": "intro",
            "question_count": 0,
            "session_start": time.time(),
            "last_question_time": None,
            "code_language": language,
            "answer_style": style,
            "audio_device": device
        }
        self.save_state()
        logger.info(f"State reset for new interview "
                    f"(language={language}, style={style})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    cm = ContextManager()
    cm.load_context()
    cm.load_state()
    
    # Test mood/persona/topic detection
    tests = [
        "Tell me about yourself",
        "What's the difference between SQL and NoSQL?",
        "How would you design a scalable ML inference system?",
        "Why did you leave your last role? Are you sure that's the real reason?",
        "That's a great answer! Can you tell me about a time you failed?",
    ]
    
    for t in tests:
        mood = cm.detect_mood(t)
        persona = cm.detect_persona(t)
        topic = cm.detect_topic(t)
        print(f"\nQ: {t}")
        print(f"  mood={mood} | persona={persona} | topic={topic}")
    
    # Test context string
    ctx = cm.get_context_string()
    print(f"\nContext string: {len(ctx)} chars")