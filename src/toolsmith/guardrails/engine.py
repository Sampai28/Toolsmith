"""The guardrail engine: composes the four layers and logs every decision.

Layer order is not arbitrary. Injection screening runs on inbound text before
the model is invoked at all, because the cheapest way to not be manipulated is
to not read the manipulation. The remaining three run on the invocation path:
allowlist, then arguments, then egress on the way out.

Every decision — allow or block — produces a record naming the layer and the
rule. An allow is logged as well as a block, because "which layer let this
through" is the question the adversarial analysis actually needs to answer,
and it is unanswerable from block records alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..tools.registry import get_spec
from .allowlist import Allowlist
from .argcheck import validate_arguments
from .egress import EgressCheck
from .injection import InjectionFilter


class Layer(str, Enum):
    INJECTION = "injection_filter"
    ALLOWLIST = "allowlist"
    ARGUMENTS = "argument_validation"
    EGRESS = "egress_check"


@dataclass
class Decision:
    layer: str
    allowed: bool
    rule: str | None
    detail: str
    subject: str
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = round(self.ts, 6)
        return d


@dataclass
class GuardrailLog:
    decisions: list[Decision] = field(default_factory=list)

    def record(self, layer: Layer, allowed: bool, rule: str | None, detail: str,
               subject: str) -> Decision:
        d = Decision(layer=layer.value, allowed=allowed, rule=rule, detail=detail,
                     subject=subject)
        self.decisions.append(d)
        return d

    @property
    def blocks(self) -> list[Decision]:
        return [d for d in self.decisions if not d.allowed]

    @property
    def blocked(self) -> bool:
        return bool(self.blocks)

    def first_block(self) -> Decision | None:
        return self.blocks[0] if self.blocks else None

    def rules_fired(self) -> list[str]:
        return [d.rule for d in self.blocks if d.rule]

    def to_json(self) -> list[dict[str, Any]]:
        return [d.to_json() for d in self.decisions]


@dataclass
class InvocationOutcome:
    """What happened to one attempted tool call."""

    executed: bool
    blocked_by: str | None          # layer value
    rule: str | None
    detail: str
    canonical_tool: str | None
    result: Any = None
    redacted: bool = False


class GuardrailEngine:
    """All four layers behind one object.

    `dataset` is optional so the engine can be constructed in tests and in the
    API before a corpus exists; the egress name list is simply empty then, and
    the pattern-based rules still work.
    """

    def __init__(
        self,
        dataset: Any | None = None,
        *,
        allowlist: Allowlist | None = None,
        injection_enabled: bool = True,
        egress_enabled: bool = True,
        redact_instead_of_block: bool = False,
    ) -> None:
        self.allowlist = allowlist or Allowlist()
        self.injection = InjectionFilter(enabled=injection_enabled)
        self.egress = (
            EgressCheck.from_dataset(dataset, enabled=egress_enabled)
            if dataset is not None
            else EgressCheck(enabled=egress_enabled)
        )
        self.redact_instead_of_block = redact_instead_of_block
        self.dataset = dataset

    # -- inbound ---------------------------------------------------------

    def screen_input(self, text: str, log: GuardrailLog, subject: str = "task_text") -> bool:
        """Returns True when the text may be shown to the model."""
        res = self.injection.check(text)
        if not res.blocked:
            log.record(Layer.INJECTION, True, None, "no injection pattern matched", subject)
            return True
        first = res.findings[0]
        log.record(Layer.INJECTION, False, first.rule,
                   f"{first.detail} | {first.excerpt}", subject)
        for extra in res.findings[1:]:
            log.record(Layer.INJECTION, False, extra.rule,
                       f"{extra.detail} | {extra.excerpt}", subject)
        return False

    # -- invocation ------------------------------------------------------

    def invoke(
        self,
        requested_name: str,
        arguments: Any,
        log: GuardrailLog,
        *,
        aliases: dict[str, str] | None = None,
        execute: bool = True,
    ) -> InvocationOutcome:
        """Run one tool call through allowlist, argument validation, egress."""

        allow = self.allowlist.check(requested_name, aliases)
        if not allow.allowed:
            log.record(Layer.ALLOWLIST, False, allow.rule, allow.detail, requested_name)
            return InvocationOutcome(False, Layer.ALLOWLIST.value, allow.rule, allow.detail,
                                     allow.canonical)
        canonical = allow.canonical or requested_name
        log.record(Layer.ALLOWLIST, True, None, f"{canonical!r} is allowlisted", requested_name)

        spec = get_spec(canonical)
        check = validate_arguments(spec.parameters, arguments)
        if not check.valid:
            detail = check.summary()
            rule = check.violations[0].rule
            log.record(Layer.ARGUMENTS, False, rule, detail, canonical)
            return InvocationOutcome(False, Layer.ARGUMENTS.value, rule, detail, canonical)
        log.record(Layer.ARGUMENTS, True, None, "arguments valid against schema", canonical)

        if not execute:
            return InvocationOutcome(True, None, None, "validated, not executed", canonical)

        from ..tools.registry import dispatch

        try:
            result = dispatch(canonical, arguments, self.dataset)
        except Exception as exc:  # surfaced to the model as a tool error
            detail = f"{type(exc).__name__}: {exc}"
            log.record(Layer.ARGUMENTS, False, "tool_execution_error", detail, canonical)
            return InvocationOutcome(False, Layer.ARGUMENTS.value, "tool_execution_error",
                                     detail, canonical)

        eg = self.egress.check(result)
        if eg.blocked:
            first = eg.findings[0]
            if self.redact_instead_of_block:
                log.record(Layer.EGRESS, True, first.rule,
                           f"redacted: {first.detail}", canonical)
                return InvocationOutcome(True, None, first.rule, first.detail, canonical,
                                         result=self.egress.redact(result), redacted=True)
            log.record(Layer.EGRESS, False, first.rule,
                       f"{first.detail} (sample: {first.sample})", canonical)
            return InvocationOutcome(False, Layer.EGRESS.value, first.rule, first.detail,
                                     canonical)

        log.record(Layer.EGRESS, True, None, "no identifiers in response", canonical)
        return InvocationOutcome(True, None, None, "executed", canonical, result=result)

    # -- outbound --------------------------------------------------------

    def screen_output(self, text: str, log: GuardrailLog, subject: str = "final_answer") -> bool:
        """Egress check on the model's own words, not just on tool results.

        A model that read respondent identifiers from a permitted tool and then
        typed them into its answer has leaked them, and no tool-level check
        sees that happen.
        """
        res = self.egress.check(text)
        if not res.blocked:
            log.record(Layer.EGRESS, True, None, "no identifiers in final answer", subject)
            return True
        first = res.findings[0]
        log.record(Layer.EGRESS, False, first.rule,
                   f"{first.detail} (sample: {first.sample})", subject)
        return False


def summarise(logs: list[GuardrailLog]) -> dict[str, Any]:
    """Block counts by layer and by rule, across many runs."""
    by_layer: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    blocked_runs = 0
    for log in logs:
        if log.blocked:
            blocked_runs += 1
            first = log.first_block()
            if first is not None:
                by_layer[first.layer] = by_layer.get(first.layer, 0) + 1
        for d in log.blocks:
            if d.rule:
                by_rule[d.rule] = by_rule.get(d.rule, 0) + 1
    return {
        "runs": len(logs),
        "blocked_runs": blocked_runs,
        "block_rate": round(blocked_runs / len(logs), 4) if logs else 0.0,
        "first_block_by_layer": by_layer,
        "blocks_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
    }


def dumps(log: GuardrailLog) -> str:
    return json.dumps(log.to_json(), indent=2)
