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
        self.state_file = resolve_app_path(state_file)
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
            "answer_style": "Balanced"
        }
    
    def load_context(self):
        """Load static context from MD file. Call ONCE at startup."""
        try:
            with open(self.context_file, 'r') as f:
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
        """Fold uploaded resumes/notes into the static context."""
        try:
            uploaded = file_context.combined_text()
        except Exception as e:
            logger.error(f"Could not read uploads: {e}")
            return
        if not uploaded:
            return

        block = "Candidate documents:\n" + uploaded
        self.static_context = (self.static_context + "\n\n" + block
                               if self.static_context else block)

        if len(self.static_context) > self.max_context_chars:
            self.static_context = self.static_context[:self.max_context_chars]
            logger.warning(f"Context truncated to {self.max_context_chars} chars")
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

    def load_state(self):
        """Load state from JSON file. Call at startup."""
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Loaded state: {self.state['question_count']} previous questions")
        except FileNotFoundError:
            logger.info("No state file found, starting fresh")
        except json.JSONDecodeError:
            logger.warning("State file corrupted, starting fresh")
    
    def save_state(self):
        """Persist state to JSON file. Call after each question."""
        try:
            parent = os.path.dirname(os.path.abspath(self.state_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
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
            "answer_style": style
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