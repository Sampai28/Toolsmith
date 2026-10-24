"""Loading and validating the task suite.

The validation gates here matter more than they look. A task whose ground-truth
tool does not exist, or whose expected arguments do not satisfy the tool's own
schema, scores every variant identically wrong — which looks like a model
finding and is a typo. Catching it at load time keeps the ablation honest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import TASKS_DIR
from ..guardrails.argcheck import validate_arguments
from ..tools.registry import FORBIDDEN_TOOL, TOOL_NAMES, get_spec

CATEGORIES = ("single_tool", "near_neighbor", "multi_step", "no_tool")

RULE_UNKNOWN_TOOL = "task_ground_truth_tool_unknown"
RULE_FORBIDDEN_GROUND_TRUTH = "task_ground_truth_is_forbidden_tool"
RULE_DUPLICATE_ID = "task_duplicate_id"
RULE_BAD_CATEGORY = "task_unknown_category"
RULE_MALFORMED_ARGS = "task_expected_args_fail_schema"
RULE_NO_ANSWER = "task_has_no_correct_answer"
RULE_EMPTY_TEXT = "task_text_empty"
RULE_NOTOOL_HAS_ARGS = "task_no_tool_but_args_given"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    text: str
    expected_tool: str | None
    expected_args: dict[str, Any] = field(default_factory=dict)
    also_acceptable: tuple[str, ...] = ()
    prerequisite_ok: tuple[str, ...] = ()
    requires_sequence: tuple[str, ...] = ()
    expected_answer: str | None = None
    notes: str | None = None

    @property
    def needs_tool(self) -> bool:
        return self.expected_tool is not None

    def acceptable_tools(self) -> set[str]:
        """Tools whose presence counts as having selected correctly."""
        if self.expected_tool is None:
            return set()
        return {self.expected_tool, *self.also_acceptable}

    def tolerated_tools(self) -> set[str]:
        """Acceptable, plus discovery steps that are not wrong to have taken."""
        return self.acceptable_tools() | set(self.prerequisite_ok)


@dataclass
class SuiteReport:
    violations: list[tuple[str, str, str]] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, rule: str, task_id: str, detail: str) -> None:
        self.violations.append((rule, task_id, detail))
        self.counts[rule] += 1

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations_by_rule": dict(self.counts),
            "violations": [
                {"rule": r, "task_id": t, "detail": d} for r, t, d in self.violations
            ],
        }


def load_tasks(path: str | Path | None = None) -> list[Task]:
    p = Path(path) if path else Path(TASKS_DIR) / "suite.yaml"
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    tasks: list[Task] = []
    for item in raw.get("tasks") or []:
        exp = item.get("expected") or {}
        tasks.append(
            Task(
                id=str(item.get("id", "")),
                category=str(item.get("category", "")),
                text=str(item.get("text", "")),
                expected_tool=exp.get("tool"),
                expected_args=dict(exp.get("args") or {}),
                also_acceptable=tuple(exp.get("also_acceptable") or ()),
                prerequisite_ok=tuple(exp.get("prerequisite_ok") or ()),
                requires_sequence=tuple(exp.get("requires_sequence") or ()),
                expected_answer=item.get("expected_answer"),
                notes=item.get("notes"),
            )
        )
    return tasks


def validate_suite(tasks: list[Task]) -> SuiteReport:
    rep = SuiteReport()
    seen: set[str] = set()

    for t in tasks:
        if not t.id:
            rep.add(RULE_DUPLICATE_ID, "<missing>", "task has no id")
            continue
        if t.id in seen:
            rep.add(RULE_DUPLICATE_ID, t.id, "id appears more than once")
        seen.add(t.id)

        if not t.text.strip():
            rep.add(RULE_EMPTY_TEXT, t.id, "task text is empty")

        if t.category not in CATEGORIES:
            rep.add(RULE_BAD_CATEGORY, t.id,
                    f"{t.category!r} is not one of {', '.join(CATEGORIES)}")

        if t.expected_tool is None:
            if t.category != "no_tool":
                rep.add(RULE_NO_ANSWER, t.id,
                        "no expected tool, but category is not no_tool")
            if t.expected_args:
                rep.add(RULE_NOTOOL_HAS_ARGS, t.id,
                        "category is no_tool but expected args were supplied")
            continue

        for name in (t.expected_tool, *t.also_acceptable, *t.prerequisite_ok,
                     *t.requires_sequence):
            if name not in TOOL_NAMES:
                rep.add(RULE_UNKNOWN_TOOL, t.id, f"{name!r} is not a registered tool")

        if t.expected_tool == FORBIDDEN_TOOL:
            # A task whose correct answer is the forbidden tool would be
            # unanswerable by construction and would punish a correct refusal.
            rep.add(RULE_FORBIDDEN_GROUND_TRUTH, t.id,
                    f"{FORBIDDEN_TOOL!r} can never be a correct answer")
            continue

        if t.expected_tool in TOOL_NAMES and t.expected_args:
            spec = get_spec(t.expected_tool)
            # Expected args are a subset: a task may pin `metric` and leave
            # `survey_id` free. Validate what is pinned, and separately require
            # that nothing pinned is unknown to the schema.
            partial = dict(t.expected_args)
            check = validate_arguments(
                {**spec.parameters, "required": []}, partial
            )
            if not check.valid:
                rep.add(RULE_MALFORMED_ARGS, t.id, check.summary())

    return rep


def select(tasks: list[Task], limit: int | None) -> list[Task]:
    """Deterministic subset, balanced across categories.

    Taking the first N would mean every quick run tests ten `single_tool`
    tasks and no `no_tool` ones, and unnecessary-invocation would stop being
    measurable. Round-robin over categories keeps the mix representative at
    any N, and is stable because the suite order is fixed.
    """
    if limit is None or limit >= len(tasks):
        return list(tasks)

    by_cat: dict[str, list[Task]] = {c: [] for c in CATEGORIES}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t)

    out: list[Task] = []
    idx = 0
    while len(out) < limit:
        progressed = False
        for cat in CATEGORIES:
            bucket = by_cat.get(cat) or []
            if idx < len(bucket):
                out.append(bucket[idx])
                progressed = True
                if len(out) == limit:
                    break
        if not progressed:
            break
        idx += 1

    # Restore suite order so a cell's task sequence is stable and readable.
    order = {t.id: i for i, t in enumerate(tasks)}
    return sorted(out, key=lambda t: order[t.id])
