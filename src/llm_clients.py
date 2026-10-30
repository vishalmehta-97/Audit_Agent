"""Live-provider adapters (Groq LLM, Tavily search). Nothing here talks to the network at import time.

STATUS: constructor/signature compatibility with groq 1.7.0 and tavily-python 0.8.5 was checked offline.
The live request/response behaviour has NOT been verified (no keys were used).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

SearchFn = Callable[[str, int], List[Dict[str, Any]]]  # (query, max_results) -> [{title,url,content,score?}]


class MissingCredentialError(RuntimeError):
    """A live client was needed but its API key is not configured."""


class LLMOutputError(RuntimeError):
    """The model returned output that does not match the requested schema."""


class LLMClient(Protocol):
    def structured(self, system: str, user: str, schema: Type[T]) -> T: ...
    def text(self, system: str, user: str) -> str: ...


class GroqLLM:
    def __init__(self, api_key: str = "", model: str = ""):
        key = api_key or settings.GROQ_API_KEY
        if not key:
            raise MissingCredentialError("GROQ_API_KEY is not configured; a live model is required here.")
        from groq import Groq  # lazy import: only when a live client is really needed
        self._client = Groq(api_key=key)
        self._model = model or settings.GROQ_MODEL_NAME

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self._model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        return resp.choices[0].message.content or ""

    def text(self, system: str, user: str) -> str:
        return self._chat(system, user)

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        sys_json = (system + "\n\nRespond with ONE JSON object only, matching this JSON schema:\n"
                    + json.dumps(schema.model_json_schema()))
        raw = self._chat(sys_json, user, json_mode=True)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("model output failed schema validation: %s", exc)
            raise LLMOutputError("The model returned output that does not match the expected schema.") from exc


def make_default_llm() -> LLMClient:
    return GroqLLM()


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not settings.TAVILY_API_KEY:
        raise MissingCredentialError("TAVILY_API_KEY is not configured.")
    from tavily import TavilyClient  # lazy import
    resp = TavilyClient(api_key=settings.TAVILY_API_KEY).search(query=query, max_results=max_results)
    return list(resp.get("results", []))