#!/usr/bin/env python3
"""Free LLM and speech-to-text fallback integration for the ESP32 assistant.

Supports:
1. Google Gemini (Free tier, multimodal audio + text) - default
2. Groq Cloud (Free tier, Whisper-large-v3-turbo + Llama-3.3-70b)
3. Ollama (100% local and offline, text questions)

Uses only Python standard library (urllib, json, base64) with zero extra dependencies.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import urllib.error
import urllib.request
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are an offline smart assistant. Answer the user's question directly and concisely "
    "in 1 or 2 plain spoken sentences (maximum 50 words) suitable for speech synthesis. "
    "Never use markdown formatting, bolding, bullet points, asterisks, or emojis."
)

# In-memory overrides
_api_keys: dict[str, str] = {}
_forced_provider: str | None = None


def set_provider_key(provider: str, api_key: str) -> None:
    """Set an in-memory API key for a provider."""
    _api_keys[provider.lower()] = api_key.strip()


def set_active_provider(provider: str | None) -> None:
    """Override the active provider ('gemini', 'groq', 'ollama', or None for auto)."""
    global _forced_provider
    _forced_provider = provider.lower() if provider else None


def get_api_key(provider: str) -> str:
    """Retrieve the API key from in-memory store or environment variables."""
    provider = provider.lower()
    if provider in _api_keys and _api_keys[provider]:
        return _api_keys[provider]
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "").strip()
    if provider == "groq":
        return os.environ.get("GROQ_API_KEY", "").strip()
    return ""


def get_active_provider() -> str | None:
    """Determine the active LLM provider based on settings and available keys."""
    if _forced_provider:
        if _forced_provider in {"gemini", "groq", "ollama", "off"}:
            return None if _forced_provider == "off" else _forced_provider

    if get_api_key("gemini"):
        return "gemini"
    if get_api_key("groq"):
        return "groq"
    if os.environ.get("OLLAMA_HOST"):
        return "ollama"
    return None


def is_llm_available() -> bool:
    """Check if any LLM fallback provider is configured and available."""
    return get_active_provider() is not None


def get_status_description() -> str:
    """Return a human-readable description of current LLM fallback status."""
    provider = get_active_provider()
    if not provider:
        return (
            "Disabled (Set GEMINI_API_KEY or GROQ_API_KEY to enable free cloud fallback)"
        )
    if provider == "gemini":
        return f"Gemini ({DEFAULT_GEMINI_MODEL}) [Free Tier]"
    if provider == "groq":
        return f"Groq ({DEFAULT_GROQ_AUDIO_MODEL} + {DEFAULT_GROQ_CHAT_MODEL}) [Free Tier]"
    if provider == "ollama":
        return f"Ollama ({DEFAULT_OLLAMA_MODEL} at {os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)}) [Local]"
    return provider


# ==============================================================================
# Gemini Provider (Audio in -> Text out, or Text in -> Text out)
# ==============================================================================

def _query_gemini_audio(wav_bytes: bytes, api_key: str, timeout: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{DEFAULT_GEMINI_MODEL}:generateContent?key={api_key}"
    )
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": b64_audio,
                        }
                    },
                    {
                        "text": (
                            f"{SYSTEM_PROMPT} "
                            "Listen to the user's spoken audio and answer their question or request."
                        )
                    },
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 120,
            "temperature": 0.3,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "").strip()


def _query_gemini_text(question: str, api_key: str, timeout: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{DEFAULT_GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{SYSTEM_PROMPT}\n\nQuestion: {question}"
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 120,
            "temperature": 0.3,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "").strip()


# ==============================================================================
# Groq Provider (Whisper STT -> Llama Chat)
# ==============================================================================

def _query_groq_audio(wav_bytes: bytes, api_key: str, timeout: float) -> str:
    boundary = "----GroqBoundary" + secrets.token_hex(16)
    body = bytearray()
    body.extend(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{DEFAULT_GROQ_AUDIO_MODEL}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
    )
    body.extend(wav_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    stt_request = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(stt_request, timeout=timeout) as response:
        stt_result = json.loads(response.read().decode("utf-8"))
        transcription = stt_result.get("text", "").strip()

    if not transcription:
        return ""

    return _query_groq_text(transcription, api_key, timeout)


def _query_groq_text(question: str, api_key: str, timeout: float) -> str:
    payload = {
        "model": DEFAULT_GROQ_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()


# ==============================================================================
# Ollama Provider (Local text)
# ==============================================================================

def _query_ollama_text(question: str, host: str, timeout: float) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {question}",
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "").strip()


# ==============================================================================
# Public Dispatchers
# ==============================================================================

def query_audio(wav_bytes: bytes, timeout: float = 30.0) -> str:
    """Process spoken audio and return a concise text answer using the active provider."""
    provider = get_active_provider()
    if not provider:
        raise RuntimeError(
            "No LLM fallback provider is available. Set GEMINI_API_KEY or GROQ_API_KEY."
        )

    if provider == "gemini":
        key = get_api_key("gemini")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        return _query_gemini_audio(wav_bytes, key, timeout)

    if provider == "groq":
        key = get_api_key("groq")
        if not key:
            raise ValueError("GROQ_API_KEY is not configured.")
        return _query_groq_audio(wav_bytes, key, timeout)

    if provider == "ollama":
        raise RuntimeError(
            "Ollama does not accept raw audio directly. Use Gemini or Groq for voice fallback."
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def query_text(question: str, timeout: float = 30.0) -> str:
    """Process a text question and return a concise text answer using the active provider."""
    provider = get_active_provider()
    if not provider:
        raise RuntimeError(
            "No LLM fallback provider is available. Set GEMINI_API_KEY or GROQ_API_KEY."
        )

    if provider == "gemini":
        key = get_api_key("gemini")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        return _query_gemini_text(question, key, timeout)

    if provider == "groq":
        key = get_api_key("groq")
        if not key:
            raise ValueError("GROQ_API_KEY is not configured.")
        return _query_groq_text(question, key, timeout)

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        return _query_ollama_text(question, host, timeout)

    raise ValueError(f"Unknown LLM provider: {provider}")
