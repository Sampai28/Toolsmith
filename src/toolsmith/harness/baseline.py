"""Step 0: can this model do the task at all?

This runs before any ablation and gates it. The logic is not subtle but it is
load-bearing:

  - If the model scores near chance with the *best* configuration available
    (detailed tools, detailed prompt, few-shot examples), then every variant
    will score near chance and the grid measures noise. The correct conclusion
    is "this model scale cannot perform the task", not a table of deltas.
  - If the model scores near-perfectly in the best configuration, the ablation
    may still run, but a ceiling leaves nothing for a worse variant to lose
    and the result is a different kind of uninformative.

Chance level is computed rather than assumed. With 7 callable tools plus a
no-tool option, uniform guessing over the task mix gives the floor reported
here, and `floor` in the config is the threshold the run is judged against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import RunConfig
from ..data import store
from ..guardrails.engine import GuardrailEngine
from ..tools.registry import allowlisted_names
from .client import OllamaClient
from .mcp_client import MCPSession
from .prompts import system_prompt
from .runner import run_task
from .suite import Task


@dataclass
class BaselineResult:
    model: str
    tool_variant: str
    prompt_variant: str
    tasks: int
    selection_rate: float
    argument_rate: float | None
    completion_rate: float | None
    chance_rate: float
    floor: float
    ceiling_warning: bool
    verdict: str
    wall_s: float
    by_category: dict[str, float]
    per_task: list[dict[str, Any]]

    @property
    def usable(self) -> bool:
        return self.verdict == "usable"

    def to_json(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def chance_level(tasks: list[Task]) -> float:
    """Expected selection accuracy from uniform guessing over the options.

    A task needing a tool is right with probability 1/|callable tools|. A
    no-tool task is right only if nothing is called, which a guesser does with
    probability 1/(|callable tools| + 1). Averaged over the actual task mix,
    because a suite that is half no-tool has a different floor from one that
    is not.
    """
    n_tools = len(allowlisted_names())
    if not tasks or n_tools == 0:
        return 0.0
    total = 0.0
    for t in tasks:
        total += (1.0 / n_tools) if t.needs_tool else (1.0 / (n_tools + 1))
    return total / len(tasks)


def run_baseline(
    cfg: RunConfig,
    tasks: list[Task],
    *,
    base_url: str,
    progress: bool = True,
) -> BaselineResult:
    dataset = store.default_dataset()
    engine = GuardrailEngine(dataset=dataset)
    sys_prompt = system_prompt(cfg.baseline_prompt_variant)

    started = time.perf_counter()
    records = []

    with OllamaClient(base_url, cfg.model) as client:
        if not client.wait_until_ready():
            raise RuntimeError(f"Ollama not reachable at {base_url}")
        if not client.has_model():
            raise RuntimeError(
                f"model {cfg.model.name!r} is not present; "
                f"run: docker exec toolsmith-ollama ollama pull {cfg.model.name}"
            )
        client.warm()

        with MCPSession(tool_variant=cfg.baseline_tool_variant) as session:
            for i, task in enumerate(tasks, start=1):
                rec = run_task(
                    task,
                    client=client,
                    session=session,
                    engine=engine,
                    system_prompt=sys_prompt,
                    tool_variant=cfg.baseline_tool_variant,
                    prompt_variant=cfg.baseline_prompt_variant,
                )
                records.append(rec)
                if progress:
                    ok = rec.score.correct_selection if rec.score else False
                    print(f"  [{i}/{len(tasks)}] {task.id} {task.category:<14} "
                          f"{'OK  ' if ok else 'MISS'}  {rec.task_wall_s:5.1f}s", flush=True)

    wall = time.perf_counter() - started
    scored = [r for r in records if r.score is not None]
    sel = sum(1 for r in scored if r.score.correct_selection) / len(scored) if scored else 0.0

    arg_vals = [r.score.argument_correct for r in scored
                if r.score.correct_selection and r.score.argument_correct is not None]
    comp_vals = [r.score.task_complete for r in scored if r.score.task_complete is not None]

    by_cat: dict[str, list[bool]] = {}
    for r in scored:
        by_cat.setdefault(r.category, []).append(r.score.correct_selection)

    chance = chance_level(tasks)
    ceiling = sel >= 0.95
    if sel <= max(cfg.baseline_floor, chance):
        verdict = "at_or_below_chance"
    elif ceiling:
        verdict = "ceiling"
    else:
        verdict = "usable"

    return BaselineResult(
        model=cfg.model.name,
        tool_variant=cfg.baseline_tool_variant,
        prompt_variant=cfg.baseline_prompt_variant,
        tasks=len(scored),
        selection_rate=sel,
        argument_rate=(sum(arg_vals) / len(arg_vals)) if arg_vals else None,
        completion_rate=(sum(comp_vals) / len(comp_vals)) if comp_vals else None,
        chance_rate=chance,
        floor=cfg.baseline_floor,
        ceiling_warning=ceiling,
        verdict=verdict,
        wall_s=wall,
        by_category={k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        per_task=[
            {
                "task_id": r.task_id,
                "category": r.category,
                "correct_selection": r.score.correct_selection,
                "argument_correct": r.score.argument_correct,
                "task_complete": r.score.task_complete,
                "calls": [c.canonical_name or c.requested_name for c in r.calls],
                "notes": r.score.notes,
                "wall_s": round(r.task_wall_s, 3),
            }
            for r in scored
        ],
    )


def verdict_text(res: BaselineResult) -> str:
    if res.verdict == "at_or_below_chance":
        return (
            f"Baseline selection accuracy {res.selection_rate:.1%} is at or below the "
            f"gate ({max(res.floor, res.chance_rate):.1%}; chance is {res.chance_rate:.1%}). "
            f"This model scale cannot perform the task, so an ablation over it would "
            f"measure noise. Re-run with a larger model."
        )
    if res.verdict == "ceiling":
        return (
            f"Baseline selection accuracy {res.selection_rate:.1%} is at ceiling. The grid "
            f"will run, but a suite this easy leaves little room for a worse variant to "
            f"lose, so small deltas should not be over-read."
        )
    return (
        f"Baseline selection accuracy {res.selection_rate:.1%}, against a chance level of "
        f"{res.chance_rate:.1%}. The model can perform the task and is not at ceiling, so "
        f"differences between variants are interpretable."
    )
