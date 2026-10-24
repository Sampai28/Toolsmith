"""Local inference client: Ollama over plain HTTP.

There is no provider SDK here and no place to put an API key. `/api/chat`
returns token counts and nanosecond timings for every call, which is what lets
the harness reconcile its own accounting against the provider's rather than
estimating tokens from string length.

Two measurement details that change what the numbers mean:

  - Ollama caches the prompt prefix. Within a grid cell the system prompt and
    tool schemas are identical across tasks, so only the first call pays to
    process them. Measured here: 46.7 s cold versus 0.14 s warm for the same
    719-token prefix. The grid therefore runs cell-sequentially, and the
    per-cell cold call is reported separately from the warm ones instead of
    being averaged into a latency figure that describes neither.
  - `total_duration` is the model's own view and excludes HTTP overhead. Wall
    time is recorded alongside it; where they disagree the gap is transport,
    and on a local socket it is small.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import ModelConfig


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_wire(raw: dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        args = fn.get("arguments")
        # Ollama returns a dict; some builds and some models return a JSON
        # string. Accepting both here keeps the failure out of scoring, where
        # it would look like an argument error rather than a transport quirk.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        return ToolCall(name=str(fn.get("name") or ""), arguments=args or {})


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens)


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    model_total_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    load_s: float = 0.0
    # True when the model reprocessed the whole prompt rather than hitting its
    # cache. The first call in a cell does; later ones should not.
    cold_prompt: bool = False
    error: str | None = None

    @property
    def made_tool_call(self) -> bool:
        return bool(self.tool_calls)


class OllamaClient:
    def __init__(self, base_url: str, model: ModelConfig) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=httpx.Timeout(model.request_timeout_s))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- health ----------------------------------------------------------

    def wait_until_ready(self, timeout_s: float = 180.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                r = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        return False

    def has_model(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name", "") for m in (r.json().get("models") or [])}
        # Ollama reports qwen2.5:7b-instruct; a bare "qwen2.5" request matches
        # the same blob, so compare on the prefix before the tag too.
        return self.model.name in names or any(
            n.split(":")[0] == self.model.name.split(":")[0] for n in names
        )

    def warm(self) -> bool:
        """Load the model so the first measured call is not a load measurement."""
        try:
            # Warms with the same num_ctx the real calls use. A warm-up at a
            # different context size loads the model into a differently-shaped
            # slot and the first real call pays to rebuild it.
            self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model.name, "prompt": "ok", "stream": False,
                      "options": {"num_predict": 1,
                                  "num_ctx": self.model.context_length}},
                timeout=self.model.request_timeout_s,
            )
            return True
        except httpx.HTTPError:
            return False

    # -- chat ------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model.name,
            "messages": messages,
            "stream": False,
            # These options are byte-identical on every request, which is not
            # incidental: changing any of them mid-run invalidates Ollama's
            # prompt cache. Measured, a probe that altered num_batch dropped
            # prefix replay from 1,100 tok/s to 20.
            "options": {
                "temperature": self.model.temperature,
                "seed": self.model.seed,
                "num_predict": max_tokens or self.model.max_tokens,
                "num_ctx": self.model.context_length,
            },
        }
        if tools:
            body["tools"] = tools

        started = time.perf_counter()
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            return ChatResult(content="", wall_s=time.perf_counter() - started,
                              error=f"{type(exc).__name__}: {exc}")
        except json.JSONDecodeError as exc:
            return ChatResult(content="", wall_s=time.perf_counter() - started,
                              error=f"malformed response: {exc}")
        wall = time.perf_counter() - started

        msg = data.get("message") or {}
        calls = [ToolCall.from_wire(tc) for tc in (msg.get("tool_calls") or [])]
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        prompt_eval_s = float(data.get("prompt_eval_duration") or 0) / 1e9

        return ChatResult(
            content=msg.get("content") or "",
            tool_calls=calls,
            usage=Usage(prompt_tokens, int(data.get("eval_count") or 0)),
            wall_s=wall,
            model_total_s=float(data.get("total_duration") or 0) / 1e9,
            prompt_eval_s=prompt_eval_s,
            eval_s=float(data.get("eval_duration") or 0) / 1e9,
            load_s=float(data.get("load_duration") or 0) / 1e9,
            # Cold is judged on throughput, not elapsed time: a cached prefix
            # replays at ~1,100 tok/s on this box while a genuine evaluation
            # runs at ~20 tok/s, so the two differ by fifty-fold and any
            # threshold in between separates them. An absolute cutoff would
            # misclassify a large cached prompt as cold purely for being large.
            cold_prompt=(
                prompt_tokens > 0
                and prompt_eval_s > 0
                and (prompt_tokens / prompt_eval_s) < 200.0
            ),
            error=None,
        )
