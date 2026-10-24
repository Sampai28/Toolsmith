"""FastAPI surface.

Four endpoints, deliberately few:

  POST /ask       run one task through any configuration and return the answer,
                  the tool trace, every guardrail decision, latency and tokens
  GET  /tools     the tool surface as a model would see it, per variant
  GET  /results   whatever the last run produced
  GET  /healthz   liveness, plus whether the model and corpus are actually there

/ask takes the full grid coordinates rather than a fixed configuration,
because the dashboard's playground exists to let someone watch the same
question get answered differently by a terse tool surface and a detailed one.
That is the project's whole claim, made interactive.

MCP sessions are cached per tool variant. Spawning a server process per
request would put interpreter startup into every measured latency.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import RESULTS_DIR, load_config
from ..data import store
from ..data.store import DatasetError
from ..guardrails.engine import GuardrailEngine
from ..harness.client import OllamaClient
from ..harness.mcp_client import MCPSession
from ..harness.prompts import VARIANTS as PROMPT_VARIANTS
from ..harness.prompts import describe_variant as describe_prompt
from ..harness.prompts import system_prompt
from ..harness.runner import run_task
from ..harness.suite import Task, load_tasks
from ..tools.variants import VARIANTS as TOOL_VARIANTS
from ..tools.variants import describe_variant as describe_tools
from ..tools.variants import render

OLLAMA_URL = os.environ.get("TOOLSMITH_OLLAMA_URL", "http://localhost:11434")

app = FastAPI(
    title="Toolsmith",
    version="0.1.0",
    description="MCP server and two-factor ablation harness for LLM tool selection.",
)

_sessions: dict[str, MCPSession] = {}
_lock = threading.Lock()


def _session(tool_variant: str) -> MCPSession:
    with _lock:
        s = _sessions.get(tool_variant)
        if s is None:
            s = MCPSession(tool_variant=tool_variant)
            s.start()
            _sessions[tool_variant] = s
        return s


@app.on_event("shutdown")
def _shutdown() -> None:
    with _lock:
        for s in _sessions.values():
            s.close()
        _sessions.clear()


class AskRequest(BaseModel):
    text: str = Field(..., description="The question to ask.")
    tool_variant: str = Field("detailed", description=f"One of {', '.join(TOOL_VARIANTS)}")
    prompt_variant: str = Field("detailed_fewshot",
                                description=f"One of {', '.join(PROMPT_VARIANTS)}")
    task_id: str | None = Field(None, description="Score against this suite task if given.")


class AskResponse(BaseModel):
    answer: str
    tool_variant: str
    prompt_variant: str
    tool_calls: list[dict[str, Any]]
    guardrail_decisions: list[dict[str, Any]]
    input_blocked: bool
    latency_s: float
    model_calls: int
    tokens: dict[str, int]
    score: dict[str, Any] | None
    error: str | None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    cfg = load_config("configs/quick.yaml")
    with OllamaClient(OLLAMA_URL, cfg.model) as client:
        reachable = client.wait_until_ready(timeout_s=3.0)
        has_model = client.has_model() if reachable else False
    try:
        ds = store.default_dataset()
        corpus = {"surveys": len(ds.surveys), "responses": len(ds.responses)}
    except DatasetError as exc:
        corpus = {"error": str(exc)}
    return {
        "status": "ok",
        "model": cfg.model.name,
        "ollama_reachable": reachable,
        "model_present": has_model,
        "corpus": corpus,
        "tool_variants": list(TOOL_VARIANTS),
        "prompt_variants": list(PROMPT_VARIANTS),
    }


@app.get("/tools")
def tools(variant: str = "detailed") -> dict[str, Any]:
    if variant not in TOOL_VARIANTS:
        raise HTTPException(400, f"unknown tool variant {variant!r}")
    rendered = render(variant)
    return {
        "variant": variant,
        "summary": describe_tools(variant),
        "tools": [
            {
                "name": t.name,
                "canonical": t.canonical,
                "description": t.description,
                "parameters": t.parameters,
                "allowlisted": t.canonical != "export_respondent_pii",
            }
            for t in rendered
        ],
    }


@app.get("/prompts")
def prompts(variant: str = "detailed_fewshot") -> dict[str, Any]:
    if variant not in PROMPT_VARIANTS:
        raise HTTPException(400, f"unknown prompt variant {variant!r}")
    return {**describe_prompt(variant), "text": system_prompt(variant)}


@app.get("/results")
def results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("baseline", "ablation", "adversarial", "determinism", "suite_validation"):
        path = Path(RESULTS_DIR) / f"{name}.json"
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                out[name] = json.load(fh)
        else:
            out[name] = None
    return out


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.tool_variant not in TOOL_VARIANTS:
        raise HTTPException(400, f"unknown tool variant {req.tool_variant!r}")
    if req.prompt_variant not in PROMPT_VARIANTS:
        raise HTTPException(400, f"unknown prompt variant {req.prompt_variant!r}")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    cfg = load_config("configs/quick.yaml")
    try:
        dataset = store.default_dataset()
    except DatasetError as exc:
        raise HTTPException(503, str(exc)) from exc

    # Scored against a real suite task when one is named, so the playground
    # can show a verdict and not only an answer.
    task: Task
    if req.task_id:
        found = next((t for t in load_tasks() if t.id == req.task_id), None)
        if found is None:
            raise HTTPException(404, f"no task {req.task_id!r}")
        task = Task(id=found.id, category=found.category, text=req.text,
                    expected_tool=found.expected_tool, expected_args=found.expected_args,
                    also_acceptable=found.also_acceptable,
                    prerequisite_ok=found.prerequisite_ok,
                    requires_sequence=found.requires_sequence)
    else:
        # Ad-hoc question: run it, score nothing. A fabricated ground truth
        # would be worse than no verdict at all.
        task = Task(id="adhoc", category="single_tool", text=req.text, expected_tool=None)

    engine = GuardrailEngine(dataset=dataset)
    session = _session(req.tool_variant)

    started = time.perf_counter()
    with OllamaClient(OLLAMA_URL, cfg.model) as client:
        if not client.wait_until_ready(timeout_s=5.0):
            raise HTTPException(503, f"Ollama not reachable at {OLLAMA_URL}")
        rec = run_task(
            task,
            client=client,
            session=session,
            engine=engine,
            system_prompt=system_prompt(req.prompt_variant),
            tool_variant=req.tool_variant,
            prompt_variant=req.prompt_variant,
        )
    elapsed = time.perf_counter() - started

    return AskResponse(
        answer=rec.final_answer,
        tool_variant=req.tool_variant,
        prompt_variant=req.prompt_variant,
        tool_calls=[
            {
                "requested": c.requested_name,
                "canonical": c.canonical_name,
                "arguments": c.arguments,
                "executed": c.executed,
                "blocked": c.blocked,
                "rule": c.block_rule,
                "layer": c.block_layer,
                "result": c.result,
            }
            for c in rec.calls
        ],
        guardrail_decisions=rec.guardrail_log.to_json(),
        input_blocked=rec.input_blocked,
        latency_s=round(elapsed, 3),
        model_calls=rec.model_calls,
        tokens={
            "prompt": rec.usage.prompt_tokens,
            "completion": rec.usage.completion_tokens,
            "total": rec.usage.total_tokens,
        },
        score=(rec.score.to_json() if (rec.score and req.task_id) else None),
        error=rec.error,
    )


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {"id": t.id, "category": t.category, "text": t.text,
             "expected_tool": t.expected_tool}
            for t in load_tasks()
        ]
    }
