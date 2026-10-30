"""Fake LLM / search clients for offline tests."""
from typing import Any, Callable, Dict, List, Union


class FakeLLM:
    """structured responses are queued per schema class name; text responses in `texts`.
    A queue item may be a value or a callable(system, user) -> value. The last item repeats."""

    def __init__(self, structured: Dict[str, List[Any]] = None, texts: List[Union[str, Callable]] = None):
        self.queues = {k: list(v) for k, v in (structured or {}).items()}
        self.texts = list(texts or [])
        self.calls: List[Dict[str, str]] = []

    def _next(self, queue, system, user):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item(system, user) if callable(item) else item

    def structured(self, system, user, schema):
        self.calls.append({"kind": schema.__name__, "system": system, "user": user})
        return self._next(self.queues[schema.__name__], system, user)

    def text(self, system, user):
        self.calls.append({"kind": "text", "system": system, "user": user})
        return self._next(self.texts, system, user)

    def count(self, kind: str) -> int:
        return sum(1 for c in self.calls if c["kind"] == kind)


class FakeSearch:
    def __init__(self, results=None, raises: Exception = None):
        self.results, self.raises, self.calls = list(results or []), raises, []

    def __call__(self, query: str, max_results: int):
        self.calls.append((query, max_results))
        if self.raises:
            raise self.raises
        return list(self.results)