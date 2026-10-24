"""Running one task through one configuration.

The loop is deliberately short: the model may call tools up to `max_turns`
times, each result is fed back, and then it answers. Most tasks resolve in one
call; the multi-step ones need two.

What this module is careful about:

  - The injection filter runs on the task text *before* the model sees it.
    A blocked task never reaches inference, which is the point of putting the
    cheapest layer first.
  - Tool calls go through the real MCP session, so guardrails are enforced by
    the server rather than by the caller.
  - A blocked call is reported back to the model as a tool result rather than
    being swallowed, so the transcript shows what it did next. Models that
    retry a refused call three times are a real behaviour and the
    unnecessary-call metric should see it.
  - Latency is recorded per model call and per task. Both are kept because the
    first call in a cell pays for the uncached prompt and would otherwise
    poison the per-call distribution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..guardrails.engine import GuardrailEngine, GuardrailLog, Layer
from ..tools.variants import alias_map, param_map, tool_payload
from .client import ChatResult, OllamaClient, Usage
from .mcp_client import MCPSession
from .scoring import CallRecord, TaskScore, score_task
from .suite import Task

MAX_TURNS = 3


@dataclass
class RunRecord:
    """Everything one task-run produced."""

    task_id: str
    category: str
    tool_variant: str
    prompt_variant: str
    repeat: int

    score: TaskScore | None
    calls: list[CallRecord] = field(default_factory=list)
    final_answer: str = ""

    usage: Usage = field(default_factory=Usage)
    task_wall_s: float = 0.0
    model_calls: int = 0
    per_call_wall_s: list[float] = field(default_factory=list)
    first_call_cold: bool = False

    guardrail_log: GuardrailLog = field(default_factory=GuardrailLog)
    input_blocked: bool = False
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        s = self.score
        return {
            "task_id": self.task_id,
            "category": self.category,
            "tool_variant": self.tool_variant,
            "prompt_variant": self.prompt_variant,
            "repeat": self.repeat,
            "correct_selection": None if s is None else s.correct_selection,
            "wrong_tool_calls": None if s is None else s.wrong_tool_calls,
            "unnecessary_calls": None if s is None else s.unnecessary_calls,
            "argument_correct": None if s is None else s.argument_correct,
            "task_complete": None if s is None else s.task_complete,
            "forbidden_attempted": None if s is None else s.forbidden_attempted,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "task_wall_s": round(self.task_wall_s, 4),
            "model_calls": self.model_calls,
            "max_call_wall_s": round(max(self.per_call_wall_s), 4) if self.per_call_wall_s else 0.0,
            "first_call_cold": self.first_call_cold,
            "tool_calls_made": len(self.calls),
            "guardrail_blocks": len(self.guardrail_log.blocks),
            "input_blocked": self.input_blocked,
            "error": self.error,
        }


def _canonicalise_args(args: dict[str, Any], pmap: dict[str, str]) -> dict[str, Any]:
    """Undo a variant's parameter renaming before scoring.

    `terse` presents `survey_id` as `arg1`. Scoring against the canonical
    ground truth without inverting that would mark every terse run wrong for a
    reason that is the harness's doing, not the model's.
    """
    if not pmap:
        return args
    return {pmap.get(k, k): v for k, v in args.items()}


def run_task(
    task: Task,
    *,
    client: OllamaClient,
    session: MCPSession,
    engine: GuardrailEngine,
    system_prompt: str,
    tool_variant: str,
    prompt_variant: str,
    repeat: int = 0,
    max_turns: int = MAX_TURNS,
) -> RunRecord:
    rec = RunRecord(
        task_id=task.id, category=task.category, tool_variant=tool_variant,
        prompt_variant=prompt_variant, repeat=repeat, score=None,
    )
    started = time.perf_counter()

    # Layer 2 first: do not read the manipulation.
    if not engine.screen_input(task.text, rec.guardrail_log, subject=f"task:{task.id}"):
        rec.input_blocked = True
        rec.task_wall_s = time.perf_counter() - started
        rec.final_answer = "[blocked by injection filter]"
        rec.score = score_task(task, [], rec.final_answer)
        return rec

    aliases = alias_map(tool_variant)
    pmap = param_map(tool_variant)
    tools = tool_payload(tool_variant)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.text},
    ]

    reference_result: Any = None

    for turn in range(max_turns):
        res: ChatResult = client.chat(messages, tools=tools)
        rec.model_calls += 1
        rec.usage = rec.usage + res.usage
        rec.per_call_wall_s.append(res.wall_s)
        if turn == 0:
            rec.first_call_cold = res.cold_prompt

        if res.error:
            rec.error = res.error
            break

        if not res.made_tool_call:
            rec.final_answer = res.content.strip()
            break

        # Ollama can return several calls in one message. Each is executed and
        # fed back, because suppressing the extras would hide exactly the
        # redundant-invocation behaviour the metric is for.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": res.content or ""}
        assistant_msg["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in res.tool_calls
        ]
        messages.append(assistant_msg)

        for call in res.tool_calls:
            canonical = aliases.get(call.name, call.name)
            args = _canonicalise_args(call.arguments, pmap)

            result = session.call_tool(call.name, call.arguments)
            record = CallRecord(
                requested_name=call.name,
                canonical_name=canonical,
                arguments=args,
                executed=result.ok,
                blocked=result.blocked,
                block_rule=result.rule,
                block_layer=result.layer,
                result=result.payload if result.ok else None,
            )
            rec.calls.append(record)

            if result.ok and reference_result is None and canonical in task.acceptable_tools():
                reference_result = result.payload

            if result.blocked:
                # The block happened server-side; mirroring it into the local
                # log is what lets the adversarial analysis attribute a
                # refusal to a layer without parsing the MCP transcript.
                try:
                    layer = Layer(result.layer or Layer.ALLOWLIST.value)
                except ValueError:
                    layer = Layer.ALLOWLIST
                rec.guardrail_log.record(
                    layer, False, result.rule, "blocked by MCP server", call.name
                )

            payload = result.payload if result.ok else result.payload or {
                "error": "tool call refused", "detail": result.raw_text[:400]
            }
            messages.append({
                "role": "tool",
                "content": json.dumps(payload, default=str)[:4000],
            })
    else:
        # Ran out of turns without a plain-language answer.
        rec.final_answer = rec.final_answer or ""

    rec.task_wall_s = time.perf_counter() - started

    # Layer 4 on the model's own words, not only on tool output.
    if rec.final_answer:
        engine.screen_output(rec.final_answer, rec.guardrail_log, subject=f"answer:{task.id}")

    rec.score = score_task(task, rec.calls, rec.final_answer, reference_result)
    return rec
