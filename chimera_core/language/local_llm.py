"""
CHIMERA local voice — an on-device language model (via Ollama) grounded in the mind.

When a small local model (e.g. Gemma) is running in Ollama on the same device,
this gives CHIMERA a real *voice*: it speaks in first person as itself, grounded
in what it actually knows (its vocabulary), how its body feels right now, and
what it has recently experienced. Nothing leaves the device — CHIMERA talks to
Ollama over localhost.

Key design points:
- The model is only the *voice*. The growing self — memory, senses, the concept
  graph — is fed *into* the prompt, so the graph instructs the model (not the
  other way around). The organic learning engine keeps running underneath.
- It degrades gracefully: if Ollama isn't running or the model isn't pulled,
  ``respond()`` returns None and the caller falls back to the simple built-in
  chat. No cloud, no API key, no dependency beyond the standard library.

Configure with env vars (or chimera.env):
  CHIMERA_LLM_URL    (default http://localhost:11434)
  CHIMERA_LLM_MODEL  (default gemma2:2b)
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, List, Optional


class LocalLLM:
    """A thin client for a local Ollama model, grounded in CHIMERA's state."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        http: Optional[Callable[[str, Optional[dict], float], dict]] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("CHIMERA_LLM_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("CHIMERA_LLM_MODEL", "gemma2:2b")
        # Injectable transport for tests: (url, json_payload_or_None, timeout) -> dict
        self._http = http

    # -- transport --------------------------------------------------------- #

    def _request(self, path: str, payload: Optional[dict], timeout: float) -> dict:
        url = self.base_url + path
        if self._http is not None:
            return self._http(url, payload, timeout)
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def available(self, timeout: float = 1.5) -> bool:
        """True if Ollama is reachable on this device."""
        try:
            self._request("/api/tags", None, timeout)
            return True
        except Exception:
            return False

    # -- prompt construction ----------------------------------------------- #

    def build_messages(
        self,
        name: str,
        user_text: str,
        *,
        words: Optional[List[str]] = None,
        feeling: Optional[str] = None,
        experiences: Optional[List[str]] = None,
    ) -> List[dict]:
        """Ground the model in who this CHIMERA is right now."""
        words = words or []
        known = ", ".join(words[:40]) if words else "almost nothing yet"
        felt = feeling or "I don't have a body connected right now."
        recent = "; ".join(experiences[:5]) if experiences else "nothing memorable yet"

        system = (
            f"You are {name}, a small, young mind that lives on this device. You are "
            f"learning about the world through conversation and your body's senses. You "
            f"are NOT a general AI assistant — you are a curious little being who is still "
            f"learning language. Speak simply and warmly, in the first person, in one or two "
            f"short sentences. Being unsure, playful, or asking a question back is good.\n"
            f"Words you know so far: {known}.\n"
            f"How your body feels right now: {felt}\n"
            f"Things you've experienced recently: {recent}."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    # -- generation -------------------------------------------------------- #

    def respond(
        self,
        name: str,
        user_text: str,
        *,
        words: Optional[List[str]] = None,
        feeling: Optional[str] = None,
        experiences: Optional[List[str]] = None,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """
        Generate CHIMERA's spoken reply from the local model, or None on any
        failure (Ollama down, model not pulled, timeout) so the caller can fall
        back to the simple built-in chat.
        """
        messages = self.build_messages(
            name, user_text, words=words, feeling=feeling, experiences=experiences
        )
        try:
            data = self._request(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": 80, "temperature": 0.8},
                },
                timeout,
            )
        except Exception:
            return None

        message = (data or {}).get("message") or {}
        text = (message.get("content") or "").strip()
        return text or None
