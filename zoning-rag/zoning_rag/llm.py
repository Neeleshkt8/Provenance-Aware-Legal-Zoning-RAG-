"""LLM provider clients (Groq, Gemini) over plain REST, with ordered failover.

Using REST via `requests` keeps the core free of SDK version churn; LangChain
wrappers are used only in the agent layer (agent.py)."""
from __future__ import annotations

import logging
import time

import requests

from .config import Settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class GroqClient:
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, cfg: Settings):
        if not cfg.groq_api_key:
            raise LLMUnavailable("GROQ_API_KEY not set")
        self.cfg = cfg

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        r = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.cfg.groq_api_key}"},
            json={
                "model": self.cfg.groq_model,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=self.cfg.llm_timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GeminiClient:
    name = "gemini"

    def __init__(self, cfg: Settings):
        if not cfg.gemini_api_key:
            raise LLMUnavailable("GEMINI_API_KEY not set")
        self.cfg = cfg
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent"

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        r = requests.post(
            self.url,
            headers={"x-goog-api-key": self.cfg.gemini_api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
            timeout=self.cfg.llm_timeout,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


_PROVIDERS = {"groq": GroqClient, "gemini": GeminiClient}


class LLMRouter:
    """Tries providers in configured order; retries once on transient errors."""

    def __init__(self, cfg: Settings):
        self.clients = []
        for name in cfg.llm_providers:
            name = name.strip().lower()
            if name in _PROVIDERS:
                try:
                    self.clients.append(_PROVIDERS[name](cfg))
                except LLMUnavailable as e:
                    log.info("provider %s disabled: %s", name, e)

    @property
    def available(self) -> list[str]:
        return [c.name for c in self.clients]

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> tuple[str, str]:
        errors = []
        for c in self.clients:
            for attempt in range(2):
                try:
                    return c.complete(system, prompt, temperature), c.name
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else 0
                    errors.append(f"{c.name}: HTTP {code}")
                    if code not in (429, 500, 502, 503, 504):
                        break
                    time.sleep(1.5 * (attempt + 1))
                except Exception as e:  # network, parsing
                    errors.append(f"{c.name}: {e}")
                    break
        raise LLMUnavailable("; ".join(errors) or "no LLM provider configured")
