"""
llm_response.py — Phase Oct

Empathetic, context-aware response generation using an LLM (Mistral via
Ollama or a hosted API) conditioned on:
    - Translated ASL sentence (from model.py / serve.py)
    - Detected emotion label (from emotion_recognition.py)
    - Conversation history (rolling window)

Pipeline:
    1. Build a system prompt that instructs the LLM to respond empathetically
       given the user's emotional state.
    2. Inject the translated sentence + emotion label as user context.
    3. Stream the LLM response back via the FastAPI WebSocket (serve.py).

TODO (Oct):
    - Prompt engineering: system prompt templates per emotion.
    - Conversation history management (token budget ~2048).
    - Streaming response support via Ollama /api/chat endpoint.
    - Fallback to a rule-based response if Ollama is unreachable.
    - Unit tests with mocked Ollama responses.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

# httpx is used for async HTTP to Ollama; install via requirements.txt
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"

SYSTEM_PROMPT_TEMPLATE = """You are a compassionate communication assistant helping a
deaf or hard-of-hearing user. The user communicated via American Sign Language.
Their message has been translated to: "{sentence}"
Detected emotional tone: {emotion}

Respond empathetically and clearly in 1-3 sentences. Acknowledge their emotion if
appropriate. Do not mention that you are an AI unless asked directly."""


class LLMResponseGenerator:
    """Placeholder — not yet implemented."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_BASE_URL,
        max_history_turns: int = 6,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.max_history_turns = max_history_turns
        self._history: list[dict] = []

        if not _HTTPX_AVAILABLE:
            raise ImportError("Install httpx: pip install httpx")
        raise NotImplementedError(
            "LLMResponseGenerator is scheduled for Oct development."
        )

    async def generate(
        self,
        sentence: str,
        emotion: str = "neutral",
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Yields response tokens as they stream from Ollama.

        Args:
            sentence: translated ASL sentence
            emotion:  emotion label from EmotionRecognizer
            stream:   whether to stream tokens (True) or return full response

        Yields:
            str chunks of the response
        """
        raise NotImplementedError

    def reset_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
