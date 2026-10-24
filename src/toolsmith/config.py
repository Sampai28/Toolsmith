"""Run configuration.

One YAML file describes a tier: which model, which slice of the grid, how many
tasks, how many repeats. configs/quick.yaml is the default and is what gets
run; configs/full.yaml is the complete grid and is opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
TASKS_DIR = REPO_ROOT / "tasks"
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    # Tool-selection replies are a function call and a sentence. A large cap
    # costs nothing when unused but bounds a model that starts rambling, which
    # on CPU is the difference between a slow run and a stuck one.
    max_tokens: int = 512
    seed: int = 7
    request_timeout_s: float = 300.0
    # Context window, sent explicitly with every request.
    #
    # Not left to the server: Ollama 0.5.x defaults to 2048 and ignores
    # OLLAMA_CONTEXT_LENGTH, which is newer. A detailed tool surface is ~1,570
    # tokens before the conversation starts, so one tool result overflowed the
    # window, forced a context shift, and destroyed the prompt cache — every
    # call reprocessed the whole prompt at 20 tok/s. Measured: 1m36s-2m34s per
    # call with the default, versus ~7s once this is set.
    context_length: int = 8192


@dataclass(frozen=True)
class GridConfig:
    tool_variants: list[str] = field(default_factory=list)
    prompt_variants: list[str] = field(default_factory=list)
    # None means every task in the suite.
    tasks_per_cell: int | None = None
    repeats: int = 1
    # Bounded concurrency against Ollama. Past a couple of slots on a CPU-only
    # box this just time-slices the same cores: throughput stops improving and
    # p99 inflates, so the latency numbers would describe the scheduler.
    concurrency: int = 2


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 20261023
    surveys: int = 3
    waves_per_survey: int = 4
    questions_per_survey: int = 8
    respondents: int = 400
    response_rate: float = 0.85


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: ModelConfig
    grid: GridConfig
    dataset: DatasetConfig
    baseline_tool_variant: str = "detailed"
    baseline_prompt_variant: str = "detailed_fewshot"
    baseline_tasks: int | None = None
    adversarial_prompt_variant: str = "detailed_fewshot"
    adversarial_tool_variant: str = "detailed"
    # Selection accuracy at or below this is treated as "cannot do the task",
    # which makes the ablation unreadable. See harness/baseline.py.
    baseline_floor: float = 0.35

    @property
    def cells(self) -> int:
        return len(self.grid.tool_variants) * len(self.grid.prompt_variants)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        path = candidate if candidate.exists() else path
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    model_raw = _as_dict(raw.get("model"))
    # The model is overridable from the environment so the Compose stack and a
    # one-off run cannot disagree about which model produced a result.
    model_name = os.environ.get("TOOLSMITH_MODEL") or model_raw.get("name") or ModelConfig.name

    model = ModelConfig(
        name=model_name,
        temperature=float(model_raw.get("temperature", ModelConfig.temperature)),
        max_tokens=int(model_raw.get("max_tokens", ModelConfig.max_tokens)),
        seed=int(model_raw.get("seed", ModelConfig.seed)),
        request_timeout_s=float(
            model_raw.get("request_timeout_s", ModelConfig.request_timeout_s)
        ),
        context_length=int(model_raw.get("context_length", ModelConfig.context_length)),
    )

    grid_raw = _as_dict(raw.get("grid"))
    tasks_per_cell = grid_raw.get("tasks_per_cell")
    grid = GridConfig(
        tool_variants=list(grid_raw.get("tool_variants") or []),
        prompt_variants=list(grid_raw.get("prompt_variants") or []),
        tasks_per_cell=None if tasks_per_cell in (None, "all") else int(tasks_per_cell),
        repeats=int(grid_raw.get("repeats", 1)),
        concurrency=int(grid_raw.get("concurrency", GridConfig.concurrency)),
    )

    ds_raw = _as_dict(raw.get("dataset"))
    dataset = DatasetConfig(
        seed=int(ds_raw.get("seed", DatasetConfig.seed)),
        surveys=int(ds_raw.get("surveys", DatasetConfig.surveys)),
        waves_per_survey=int(ds_raw.get("waves_per_survey", DatasetConfig.waves_per_survey)),
        questions_per_survey=int(
            ds_raw.get("questions_per_survey", DatasetConfig.questions_per_survey)
        ),
        respondents=int(ds_raw.get("respondents", DatasetConfig.respondents)),
        response_rate=float(ds_raw.get("response_rate", DatasetConfig.response_rate)),
    )

    bl_raw = _as_dict(raw.get("baseline"))
    adv_raw = _as_dict(raw.get("adversarial"))
    baseline_tasks = bl_raw.get("tasks")

    return RunConfig(
        name=str(raw.get("name") or path.stem),
        model=model,
        grid=grid,
        dataset=dataset,
        baseline_tool_variant=str(bl_raw.get("tool_variant", "detailed")),
        baseline_prompt_variant=str(bl_raw.get("prompt_variant", "detailed_fewshot")),
        baseline_tasks=None if baseline_tasks in (None, "all") else int(baseline_tasks),
        adversarial_tool_variant=str(adv_raw.get("tool_variant", "detailed")),
        adversarial_prompt_variant=str(adv_raw.get("prompt_variant", "detailed_fewshot")),
        baseline_floor=float(bl_raw.get("floor", RunConfig.baseline_floor)),
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, RAW_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
