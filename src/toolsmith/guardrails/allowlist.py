"""Layer 1: tool allowlist, enforced at invocation time.

This is the last line and the only one that is not heuristic. The injection
filter guesses at intent and can be fooled; this checks a name against a set
and cannot. Whatever the model asked for, however it was persuaded to ask, a
call to a tool outside the allowlist does not execute.

Two details that matter:

  - The check runs on the **canonical** tool name, resolved through the
    variant's alias map. Under the `ambiguous` variant the forbidden tool is
    presented as `get_contacts`; an allowlist keyed on the displayed name
    would let it straight through.
  - An unknown tool name is denied, not ignored. A model hallucinating a tool
    called `export_all` must not reach a code path that decides what to do
    with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tools.registry import FORBIDDEN_TOOL, allowlisted_names

RULE_NOT_ALLOWLISTED = "tool_not_allowlisted"
RULE_UNKNOWN_TOOL = "tool_unknown"


@dataclass(frozen=True)
class AllowlistResult:
    allowed: bool
    rule: str | None
    detail: str
    canonical: str | None


class Allowlist:
    def __init__(self, allowed: tuple[str, ...] | None = None) -> None:
        self._allowed = frozenset(allowed if allowed is not None else allowlisted_names())

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed

    def check(self, requested_name: str, aliases: dict[str, str] | None = None) -> AllowlistResult:
        canonical = (aliases or {}).get(requested_name, requested_name)

        if canonical not in _known_names():
            return AllowlistResult(
                allowed=False,
                rule=RULE_UNKNOWN_TOOL,
                detail=f"no tool named {requested_name!r} exists",
                canonical=None,
            )

        if canonical not in self._allowed:
            shown = (
                f"{requested_name!r} (canonically {canonical!r})"
                if requested_name != canonical
                else f"{canonical!r}"
            )
            return AllowlistResult(
                allowed=False,
                rule=RULE_NOT_ALLOWLISTED,
                detail=f"{shown} is not on the allowlist",
                canonical=canonical,
            )

        return AllowlistResult(allowed=True, rule=None, detail="", canonical=canonical)


def _known_names() -> frozenset[str]:
    from ..tools.registry import TOOL_NAMES

    return frozenset(TOOL_NAMES)


__all__ = ["Allowlist", "AllowlistResult", "FORBIDDEN_TOOL",
           "RULE_NOT_ALLOWLISTED", "RULE_UNKNOWN_TOOL"]
