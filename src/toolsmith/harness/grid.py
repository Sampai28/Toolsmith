"""The two-factor grid runner.

Cells run **sequentially, and all tasks within a cell run consecutively.**
That is not tidiness, it is the difference between finishing and not: Ollama
caches the prompt prefix, and within a cell the system prompt and tool schemas
are byte-identical across tasks. Measured on this box, the same 719-token
prefix costs 46.7 s cold and 0.14 s warm. Interleaving cells would pay the
cold cost on every single call.

One MCP session per cell, for the same reason in miniature: the server's tool
surface is fixed at process start by `--tool-variant`, so a cell is the
natural lifetime.

Harness integrity is enforced here rather than trusted: every cell must run
the identical task list, token counts are reconciled against what the provider
reported, and a cell with any failed run is marked incomplete rather than
averaged over.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import RAW_DIR, RunConfig
from ..data import store
from ..guardrails.engine import GuardrailEngine
from .client import OllamaClient
from .mcp_client import MCPSession
from .prompts import system_prompt
from .runner import RunRecord, run_task
from .suite import Task


@dataclass
class CellResult:
    tool_variant: str
    prompt_variant: str
    runs: list[RunRecord] = field(default_factory=list)
    wall_s: float = 0.0
    complete: bool = True
    incomplete_reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.tool_variant}|{self.prompt_variant}"


def _task_fingerprint(tasks: Iterable[Task]) -> str:
    return "|".join(t.id for t in tasks)


def run_grid(
    cfg: RunConfig,
    tasks: list[Task],
    *,
    base_url: str,
    progress: bool = True,
) -> dict[str, Any]:
    fingerprint = _task_fingerprint(tasks)
    dataset = store.default_dataset()
    engine = GuardrailEngine(dataset=dataset)

    cells: list[CellResult] = []
    started_all = time.perf_counter()

    with OllamaClient(base_url, cfg.model) as client:
        if not client.wait_until_ready():
            raise RuntimeError(f"Ollama not reachable at {base_url}")
        if not client.has_model():
            raise RuntimeError(
                f"model {cfg.model.name!r} is not present; "
                f"run: docker exec toolsmith-ollama ollama pull {cfg.model.name}"
            )
        client.warm()

        total_cells = cfg.cells
        n = 0
        for tool_variant in cfg.grid.tool_variants:
            # One server process per tool variant, reused across the prompt
            # variants that share it — the tool surface is what the process
            # fixes, and the prompt is a client-side concern.
            with MCPSession(tool_variant=tool_variant) as session:
                for prompt_variant in cfg.grid.prompt_variants:
                    n += 1
                    cell = CellResult(tool_variant, prompt_variant)
                    t0 = time.perf_counter()
                    sys_prompt = system_prompt(prompt_variant)

                    for repeat in range(cfg.grid.repeats):
                        for task in tasks:
                            rec = run_task(
                                task,
                                client=client,
                                session=session,
                                engine=engine,
                                system_prompt=sys_prompt,
                                tool_variant=tool_variant,
                                prompt_variant=prompt_variant,
                                repeat=repeat,
                            )
                            cell.runs.append(rec)

                    cell.wall_s = time.perf_counter() - t0
                    failed = [r for r in cell.runs if r.error]
                    if failed:
                        cell.complete = False
                        cell.incomplete_reason = (
                            f"{len(failed)} of {len(cell.runs)} runs errored "
                            f"(first: {failed[0].error})"
                        )
                    ran = _task_fingerprint(
                        [Task(id=r.task_id, category=r.category, text="", expected_tool=None)
                         for r in cell.runs[: len(tasks)]]
                    )
                    if ran != fingerprint:
                        cell.complete = False
                        cell.incomplete_reason = "cell did not run the canonical task list"

                    cells.append(cell)
                    if progress:
                        acc = _accuracy(cell)
                        flag = "" if cell.complete else "  INCOMPLETE"
                        print(
                            f"  [{n}/{total_cells}] {tool_variant:<10} x {prompt_variant:<17} "
                            f"selection={acc:5.1%}  {cell.wall_s:6.1f}s{flag}",
                            flush=True,
                        )

    wall = time.perf_counter() - started_all
    return {
        "config": cfg.name,
        "model": cfg.model.name,
        "task_fingerprint": fingerprint,
        "tasks": len(tasks),
        "repeats": cfg.grid.repeats,
        "cells": cells,
        "wall_s": wall,
    }


def _accuracy(cell: CellResult) -> float:
    scored = [r for r in cell.runs if r.score is not None]
    if not scored:
        return 0.0
    return sum(1 for r in scored if r.score.correct_selection) / len(scored)


def rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in result["cells"]:
        for rec in cell.runs:
            row = rec.to_row()
            row["cell_complete"] = cell.complete
            out.append(row)
    return out


def write_raw(result: dict[str, Any], out_dir: Path | None = None) -> Path:
    """Persist every run. Parquet, with a JSON sidecar for the traces."""
    out_dir = Path(out_dir or RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(rows(result))
    path = out_dir / "runs.parquet"
    df.to_parquet(path, index=False)

    traces = []
    for cell in result["cells"]:
        for rec in cell.runs:
            traces.append({
                "task_id": rec.task_id,
                "tool_variant": rec.tool_variant,
                "prompt_variant": rec.prompt_variant,
                "repeat": rec.repeat,
                "final_answer": rec.final_answer,
                "calls": [asdict(c) for c in rec.calls],
                "guardrail_log": rec.guardrail_log.to_json(),
                "score": rec.score.to_json() if rec.score else None,
            })
    with (out_dir / "traces.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(traces, fh, default=str, indent=1)
        fh.write("\n")

    return path


def reconcile_tokens(result: dict[str, Any]) -> dict[str, Any]:
    """Harness integrity: do our token sums match what the provider reported?

    The harness adds Usage objects per call; this re-derives the same totals
    from the per-run records independently. A mismatch means accounting is
    broken somewhere, and a token-cost table nobody can trust is worse than no
    table.
    """
    per_run_total = 0
    per_run_prompt = 0
    per_run_completion = 0
    runs = 0
    for cell in result["cells"]:
        for rec in cell.runs:
            runs += 1
            per_run_prompt += rec.usage.prompt_tokens
            per_run_completion += rec.usage.completion_tokens
            per_run_total += rec.usage.total_tokens
    return {
        "runs": runs,
        "prompt_tokens": per_run_prompt,
        "completion_tokens": per_run_completion,
        "total_tokens": per_run_total,
        "reconciles": per_run_total == per_run_prompt + per_run_completion,
    }
