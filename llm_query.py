"""
llm_query.py — Ollama cloud client with streaming responses

Sends transcribed question + context to Ollama cloud API (glm-5.2),
streams the response back token-by-token for minimum latency.

Target: <0.5s time-to-first-token, <1s for short answers.
Uses Ollama /api/chat endpoint for text queries (NOT OpenAI-compatible /v1/chat/completions). Vision queries use OpenRouter's OpenAI-compatible /chat/completions endpoint instead.
"""

import json
import logging
import time
import os
import sys
from typing import Generator

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_VISION_MODEL, OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)

# System prompt template — kept short for speed
SYSTEM_PROMPT = """You are an interview assistant. The interviewer asked a question during a technical interview.
Give a CONCISE, NATURAL answer (max 150 words) that the candidate can read and paraphrase.

Rules:
- Answer directly, no preamble ("Great question!", "That's a good point")
- Use simple language, not academic
- If it's a coding question, give the key concept + brief code snippet
- If it's a behavioral question, give a STAR-format answer
- Match the candidate's persona from context
- Don't repeat previous answers
- Be specific, not generic"""


def build_prompt(question: str, context: str, state: dict) -> str:
    """Build the full prompt from question + context + state."""
    recent_qa = ""
    qa_history = list(zip(
        state.get("questions_asked", []),
        state.get("answers_given", [])
    ))
    if qa_history:
        recent = qa_history[-3:]
        recent_qa = "\n\nPrevious Q&A:\n"
        for i, (q, a) in enumerate(recent, 1):
            recent_qa += f"Q{i}: {q}\nA{i}: {a[:200]}...\n"
    
    mood = state.get("interviewer_mood", "neutral")
    persona = state.get("interviewer_persona", "technical")
    topic = state.get("current_topic", "general")
    
    prompt = f"""Context:
{context}

Interviewer mood: {mood}
Interviewer style: {persona}
Current topic: {topic}
{recent_qa}

Question: {question}

Answer:"""
    
    return prompt


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

                total_text += content
                token_count += 1
                yield content

            if data.get("done", False):
                break

        total_ms = (time.time() - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0

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
            "temperature": 0.7,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhostInterviewAgent/1.0",
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

    Used by the screen capture feature: `prompt` is the fixed generic
    SCREEN_CAPTURE_PROMPT (not a transcribed question), `image_b64` is a
    base64-encoded PNG screenshot.
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
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhostInterviewAgent/1.0",
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
            "temperature": 0.7,
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "GhostInterviewAgent/1.0",
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

    print("\n\n=== OpenRouter Vision Test ===\n")

    import base64
    from screen_capture import ScreenCapture

    png_bytes = ScreenCapture().capture_primary_monitor()
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    from config import SCREEN_CAPTURE_PROMPT

    for chunk in query_openrouter_vision_stream(
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