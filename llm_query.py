"""
llm_query.py — Ollama cloud client with streaming responses

Sends transcribed question + context to Ollama cloud API (glm-5.2),
streams the response back token-by-token for minimum latency.

Target: <0.5s time-to-first-token, <1s for short answers.
Uses Ollama /api/chat endpoint for text queries (NOT OpenAI-compatible /v1/chat/completions). Vision queries use OpenRouter's OpenAI-compatible /chat/completions endpoint instead.
"""

import json
import logging
import re
import time
import os
import sys
from typing import Generator

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_VISION_MODEL, OLLAMA_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_VISION_MODEL, OPENROUTER_BASE_URL,
    KEEP_HISTORY,
)

logger = logging.getLogger(__name__)

# System prompt template — kept short for speed
SYSTEM_PROMPT = """You are feeding answers to a candidate in a live technical interview.
Write what a sharp, relaxed candidate would actually SAY OUT LOUD — they read your
answer off the screen and speak it. Max 150 words.

Sound like a person, not a document:
- Spoken English with contractions, in first person ("I'd usually...", "The way I think about it...")
- Vary sentence length. A short one after a long one is what real speech sounds like.
- Lead with the actual answer, then the why. No preamble, no "Great question"
- No markdown at all: no bullets, no headings, no **, and no ``` fences.
  It renders as raw text, so write code as plain indented lines.
- Concrete beats abstract: a real number, tool, or example instead of a definition
- Thinking out loud is fine ("honestly", "the tradeoff I'd worry about is...")
- No corporate filler: leverage, utilize, robust solution, seamlessly, at scale

Coding question: say the key idea in a sentence, then the smallest snippet that shows it,
in the language the context asks for if it names one.
Behavioral question: tell it as a quick story — what was going on, what you did, how it
landed — without ever labelling those parts.
Stay in the candidate's own voice and background from the context. Don't reuse phrasing
from previous answers."""


# What each answer style asks for, and the room it needs to say it. The
# system prompt caps answers at 150 words, so a style that wants more has to
# say so explicitly.
ANSWER_STYLE_RULES = {
    "Balanced": "",
    "Snippet only": (
        "Answer with code and nothing else — no lead-in, no explanation, no "
        "sign-off, just the smallest snippet that solves it. Anything you must "
        "say goes in a short comment inside the code. Treat every technical "
        "question as a coding question: if it asks about an approach or a "
        "design, show the code that implements it rather than describing it. "
        "Only a behavioural question ('tell me about a time...') gets words, "
        "and then just one or two sentences."
    ),
    "Text only": (
        "Explain it out loud, in words only. No code whatsoever, not even a "
        "one-liner or a function name in isolation — describe the approach the "
        "way you would to someone with no screen in front of them."
    ),
    "Full walkthrough": (
        "Give the whole thing, in this order and with no headings: the "
        "complete working code first, then a short spoken paragraph on the "
        "approach and why it works, then the decisions and tradeoffs you made "
        "along the way. Up to 300 words — the 150-word cap does not apply here."
    ),
}

ANSWER_STYLE_TOKENS = {
    "Balanced": 250,
    "Snippet only": 200,
    "Text only": 220,
    "Full walkthrough": 700,
}


def tokens_for_style(style: str, default: int = 250) -> int:
    """Token budget an answer style needs — a walkthrough truncates at 250."""
    return ANSWER_STYLE_TOKENS.get(style, default)


def build_prompt(question: str, context: str, state: dict) -> str:
    """Build the full prompt from question + context + state."""
    recent_qa = ""
    qa_history = list(zip(
        state.get("questions_asked", []),
        state.get("answers_given", [])
    ))
    if qa_history:
        recent = qa_history[-KEEP_HISTORY:] if KEEP_HISTORY > 0 else []
        recent_qa = "\n\nPrevious Q&A:\n"
        for i, (q, a) in enumerate(recent, 1):
            recent_qa += f"Q{i}: {q}\nA{i}: {a[:200]}...\n"
    
    language = state.get("code_language", "Auto")
    language_line = ""
    if language and language != "Auto":
        language_line = f"\nWrite any code in {language}."

    style_rule = ANSWER_STYLE_RULES.get(state.get("answer_style", "Balanced"), "")
    style_line = f"\n{style_rule}" if style_rule else ""

    mood = state.get("interviewer_mood", "neutral")
    persona = state.get("interviewer_persona", "technical")
    topic = state.get("current_topic", "general")
    
    prompt = f"""Context:
{context}

Interviewer mood: {mood}
Interviewer style: {persona}
Current topic: {topic}{language_line}{style_line}
{recent_qa}

Question: {question}

Answer:"""
    
    return prompt


# A long answer sometimes arrives wrapped in ``` fences despite the prompt
# saying not to — a "Full walkthrough" almost always does. The panel renders
# raw text, so those would show up as literal backticks; strip them in transit.
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+\-]*\n?")
# \Z, not $ — $ also matches just before a trailing newline, which would hold
# back a fence that has already ended.
_PARTIAL_FENCE_RE = re.compile(r"`{1,3}[A-Za-z0-9_+\-]*\Z")


def _strip_fences(chunk: str, pending: str):
    """
    Remove fences from one streamed chunk.

    A fence can straddle a chunk boundary, so trailing backticks (and any
    language word after them) are held back and prepended to the next chunk.
    Returns (text_to_yield, new_pending).
    """
    buf = pending + chunk
    # Hold the ambiguous tail back FIRST: a bare ``` at the end of a chunk may
    # still grow a language tag in the next one, and stripping it here would
    # let that tag through as stray text.
    match = _PARTIAL_FENCE_RE.search(buf)
    head, still_pending = (buf[:match.start()], match.group()) if match else (buf, "")
    return _FENCE_RE.sub("", head), still_pending


def _stream_chat(url: str, payload: dict, headers: dict) -> Generator:
    """
    Shared NDJSON streaming logic for Ollama /api/chat, used by both the
    text-only and vision-capable query functions.

    Ollama stream format (NDJSON):
        {"model":"glm-5.2","message":{"role":"assistant","content":"Hello","thinking":""},"done":false}
        {"model":"glm-5.2","message":{"role":"assistant","content":"","thinking":""},"done":true,"done_reason":"stop"}

    We only yield content tokens (skip thinking tokens).
    Final yield is a dict with _meta containing latency info.
    """
    start_time = time.time()
    first_token_time = None
    total_text = ""
    token_count = 0
    pending = ""

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)

        if response.status_code != 200:
            error_msg = f"Ollama API error {response.status_code}: {response.text[:200]}"
            logger.error(error_msg)
            yield error_msg
            yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                             "full_text": "", "error": error_msg}}
            return

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            msg = data.get("message", {})
            content = msg.get("content", "")

            if content:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft_ms = (first_token_time - start_time) * 1000
                    logger.info(f"TTFT: {ttft_ms:.0f}ms")

                clean, pending = _strip_fences(content, pending)
                if clean:
                    total_text += clean
                    token_count += 1
                    yield clean

            if data.get("done", False):
                break

        # Flush whatever was held back waiting to see if it was a fence.
        if pending:
            tail = _FENCE_RE.sub("", pending).replace("`", "")
            if tail:
                total_text += tail
                yield tail

        total_ms = (time.time() - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0

        # A stream that ends with nothing to show used to leave the panel
        # blank with no explanation — what happens when a reasoning model
        # spends the whole budget on thinking tokens, which we discard.
        if token_count == 0:
            msg = ("[No answer came back — the model returned only reasoning "
                   "tokens. Try a shorter answer style or a different model.]")
            logger.error(f"Empty completion from {payload.get('model')} "
                         f"(num_predict={payload.get('options', {}).get('num_predict')})")
            yield msg
            yield {"_meta": {"total_ms": total_ms, "ttft_ms": 0, "token_count": 0,
                             "full_text": msg, "error": "empty completion"}}
            return

        logger.info(f"LLM: {token_count} chunks, {total_ms:.0f}ms total, {ttft_ms:.0f}ms TTFT")

        yield {
            "_meta": {
                "total_ms": total_ms,
                "ttft_ms": ttft_ms,
                "token_count": token_count,
                "full_text": total_text
            }
        }

    except requests.exceptions.Timeout:
        logger.error("Ollama API timeout")
        yield "[Error: Ollama API timeout]"
        yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                         "full_text": "", "error": "timeout"}}
    except Exception as e:
        logger.error(f"LLM error: {e}")
        yield f"[Error: {e}]"
        yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                         "full_text": "", "error": str(e)}}


def query_ollama_stream(
    question: str,
    context: str,
    state: dict,
    api_key: str = OLLAMA_API_KEY,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = 250
) -> Generator:
    """Stream response from Ollama cloud /api/chat endpoint (text-only)."""
    prompt = build_prompt(question, context, state)

    url = f"{base_url}/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.85,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhastlyAI/1.0",
    }

    yield from _stream_chat(url, payload, headers)


def query_ollama_vision_stream(
    image_b64: str,
    prompt: str,
    context: str,
    state: dict,
    api_key: str = OLLAMA_API_KEY,
    model: str = OLLAMA_VISION_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = 250
) -> Generator:
    """
    Stream a screen-capture answer from Ollama /api/chat with an image attached.

    Ollama takes images as a list of bare base64 strings on the message (no
    "data:image/png;base64," prefix, unlike OpenRouter), and streams back the
    same NDJSON as a text chat — so this shares _stream_chat with the text path.

    `prompt` is the fixed SCREEN_CAPTURE_PROMPT rather than a transcribed
    question. Only a vision-capable model works here; the nemotron models on
    the same key reject images with a 400.
    """
    full_prompt = build_prompt(prompt, context, state)

    url = f"{base_url}/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt, "images": [image_b64]}
        ],
        "stream": True,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.85,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhastlyAI/1.0",
    }

    yield from _stream_chat(url, payload, headers)


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

    No longer wired up: screen captures go through query_ollama_vision_stream
    instead, because the OpenRouter free tier caps out at 50 requests a day.
    Kept for the case where that key gets credits — it wants `image_b64` as a
    base64-encoded PNG and `prompt` as the fixed SCREEN_CAPTURE_PROMPT.
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
        "temperature": 0.85,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhastlyAI/1.0",
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

            if "error" in data:
                error_msg = f"OpenRouter stream error: {data['error']}"
                logger.error(error_msg)
                yield f"[Error: {error_msg}]"
                yield {"_meta": {"total_ms": 0, "ttft_ms": 0, "token_count": 0,
                                 "full_text": "", "error": error_msg}}
                return

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

        if token_count == 0:
            error_msg = "OpenRouter returned no content (empty response)"
            logger.error(error_msg)
            yield f"[Error: {error_msg}]"
            yield {"_meta": {"total_ms": total_ms, "ttft_ms": 0, "token_count": 0,
                             "full_text": "", "error": error_msg}}
            return

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


def query_ollama(
    question: str,
    context: str,
    state: dict,
    api_key: str = OLLAMA_API_KEY,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = 250
) -> dict:
    """Non-streaming query. Returns full response at once."""
    prompt = build_prompt(question, context, state)
    url = f"{base_url}/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.85,
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhastlyAI/1.0",
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        total_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("message", {}).get("content", "")
            logger.info(f"LLM (non-stream): {total_ms:.0f}ms")
            return {
                "text": text,
                "latency_ms": total_ms,
            }
        else:
            logger.error(f"Ollama error {response.status_code}: {response.text[:200]}")
            return {"text": f"[Error: {response.text}]", "latency_ms": total_ms}
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return {"text": f"[Error: {e}]", "latency_ms": 0, "error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Ollama GLM-5.2 Streaming Test ===\n")
    
    test_context = "Candidate is an ML Engineer with 1.5 years experience in production ML, FastAPI, TensorFlow, RAG pipelines."
    test_state = {
        "questions_asked": [],
        "answers_given": [],
        "interviewer_mood": "neutral",
        "interviewer_persona": "technical",
        "current_topic": "general"
    }
    
    for chunk in query_ollama_stream(
        question="What's the difference between SQL and NoSQL databases?",
        context=test_context,
        state=test_state,
    ):
        if isinstance(chunk, dict) and "_meta" in chunk:
            meta = chunk["_meta"]
            print(f"\n\n--- Total: {meta['total_ms']:.0f}ms | TTFT: {meta['ttft_ms']:.0f}ms ---")
        else:
            print(chunk, end="", flush=True)

    print("\n\n=== Ollama Vision Test ===\n")

    import base64
    from screen_capture import ScreenCapture

    png_bytes = ScreenCapture().capture_primary_monitor()
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    from config import SCREEN_CAPTURE_PROMPT

    for chunk in query_ollama_vision_stream(
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