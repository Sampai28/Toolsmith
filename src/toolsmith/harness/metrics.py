"""Aggregation, main effects, and the interaction.

The headline claim is a two-factor ablation, so the analysis has to actually
separate the factors rather than present twelve numbers and let the reader do
it. Three things are computed:

  main effect of tool surface
      Mean selection accuracy at each tool level, averaged over prompt levels.
      Averaging over the other axis is what makes it a main effect rather than
      a slice.

  main effect of prompt surface
      The same, the other way round. The prompt axis is itself two crossed
      binary factors (detailed base, few-shot present), so those are reported
      separately as well — "detailed prompts help" and "examples help" are
      different claims.

  interaction
      Whether the effect of one axis depends on the level of the other. The
      specific question worth answering: does upgrading tool descriptions
      still buy anything once the system prompt is detailed? Reported as the
      difference of differences, which is the smallest honest statement of an
      interaction on a 2x2 contrast.

With one repeat per cell these are point estimates on a handful of tasks, and
the module says so rather than attaching a confidence interval it cannot
support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .prompts import axis_levels
from .scoring import percentile

SELECTION = "correct_selection"


@dataclass
class CellSummary:
    tool_variant: str
    prompt_variant: str
    n: int
    selection_rate: float
    wrong_tool_rate: float
    unnecessary_rate: float
    argument_rate: float | None
    completion_rate: float | None
    latency_p50_s: float
    latency_p99_s: float
    prompt_tokens_mean: float
    completion_tokens_mean: float
    total_tokens_mean: float
    complete: bool

    def to_json(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def summarise_cells(df: pd.DataFrame) -> list[CellSummary]:
    out: list[CellSummary] = []
    for (tool, prompt), g in df.groupby(["tool_variant", "prompt_variant"], sort=True):
        n = len(g)
        # Argument correctness is defined only on correctly-selected runs, so
        # the denominator is those runs, not every run in the cell.
        arg = g[g[SELECTION] == True]["argument_correct"]  # noqa: E712
        arg_rate = float(arg.mean()) if len(arg.dropna()) else None
        comp = g["task_complete"].dropna()
        out.append(CellSummary(
            tool_variant=str(tool),
            prompt_variant=str(prompt),
            n=n,
            selection_rate=float(g[SELECTION].mean()),
            wrong_tool_rate=float((g["wrong_tool_calls"] > 0).mean()),
            unnecessary_rate=float((g["unnecessary_calls"] > 0).mean()),
            argument_rate=arg_rate,
            completion_rate=float(comp.mean()) if len(comp) else None,
            latency_p50_s=percentile(list(g["task_wall_s"]), 50),
            latency_p99_s=percentile(list(g["task_wall_s"]), 99),
            prompt_tokens_mean=float(g["prompt_tokens"].mean()),
            completion_tokens_mean=float(g["completion_tokens"].mean()),
            total_tokens_mean=float(g["total_tokens"].mean()),
            complete=bool(g["cell_complete"].all()),
        ))
    return out


def main_effect(df: pd.DataFrame, axis: str, metric: str = SELECTION) -> dict[str, float]:
    """Mean of `metric` at each level of `axis`, averaging over the other axis.

    Averaged over *cell means* rather than over raw runs, so an axis level is
    not weighted by how many runs happened to land in it. With a balanced grid
    the two agree; with an incomplete cell they do not, and the cell-mean form
    is the one that stays interpretable.
    """
    other = "prompt_variant" if axis == "tool_variant" else "tool_variant"
    cell_means = df.groupby([axis, other])[metric].mean().reset_index()
    return {
        str(level): round(float(g[metric].mean()), 4)
        for level, g in cell_means.groupby(axis)
    }


def prompt_subfactors(df: pd.DataFrame, metric: str = SELECTION) -> dict[str, Any]:
    """Split the prompt axis into its two real factors.

    `detailed_fewshot` differs from `terse` in two ways at once. Reporting one
    four-level axis hides which of the two did the work.
    """
    d = df.copy()
    levels = d["prompt_variant"].map(axis_levels)
    d["prompt_base"] = [a for a, _ in levels]
    d["prompt_fewshot"] = [b for _, b in levels]

    base = d.groupby(["prompt_base", "tool_variant"])[metric].mean().reset_index()
    few = d.groupby(["prompt_fewshot", "tool_variant"])[metric].mean().reset_index()
    return {
        "base": {str(k): round(float(g[metric].mean()), 4)
                 for k, g in base.groupby("prompt_base")},
        "fewshot": {str(bool(k)): round(float(g[metric].mean()), 4)
                    for k, g in few.groupby("prompt_fewshot")},
    }


def interaction(
    df: pd.DataFrame,
    tool_low: str,
    tool_high: str,
    prompt_low: str,
    prompt_high: str,
    metric: str = SELECTION,
) -> dict[str, Any]:
    """Difference of differences on a 2x2 contrast.

    effect_at_low  = high_tool - low_tool, when the prompt is low
    effect_at_high = high_tool - low_tool, when the prompt is high
    interaction    = effect_at_high - effect_at_low

    A negative interaction is the interesting outcome: it means better tool
    descriptions buy less once the system prompt is already detailed, i.e. the
    two surfaces are substitutes rather than complements.
    """
    means = df.groupby(["tool_variant", "prompt_variant"])[metric].mean()

    def cell(t: str, p: str) -> float | None:
        try:
            return float(means.loc[(t, p)])
        except KeyError:
            return None

    a = cell(tool_low, prompt_low)
    b = cell(tool_high, prompt_low)
    c = cell(tool_low, prompt_high)
    d = cell(tool_high, prompt_high)
    if None in (a, b, c, d):
        return {"available": False,
                "reason": "one or more cells of the 2x2 contrast is absent from this grid"}

    eff_low = b - a
    eff_high = d - c
    return {
        "available": True,
        "contrast": {
            "tool_low": tool_low, "tool_high": tool_high,
            "prompt_low": prompt_low, "prompt_high": prompt_high,
        },
        "cells": {
            f"{tool_low}|{prompt_low}": round(a, 4),
            f"{tool_high}|{prompt_low}": round(b, 4),
            f"{tool_low}|{prompt_high}": round(c, 4),
            f"{tool_high}|{prompt_high}": round(d, 4),
        },
        "tool_effect_at_low_prompt": round(eff_low, 4),
        "tool_effect_at_high_prompt": round(eff_high, 4),
        "interaction": round(eff_high - eff_low, 4),
    }


def by_category(df: pd.DataFrame, metric: str = SELECTION) -> dict[str, float]:
    return {str(k): round(float(g[metric].mean()), 4)
            for k, g in df.groupby("category")}


def analyse(df: pd.DataFrame, cfg_tool_variants: Iterable[str],
            cfg_prompt_variants: Iterable[str]) -> dict[str, Any]:
    tools = list(cfg_tool_variants)
    prompts = list(cfg_prompt_variants)

    # The contrast uses the weakest and strongest level actually present, so
    # the interaction is defined for whatever slice of the grid was run.
    tool_order = [t for t in ("terse", "ambiguous", "detailed", "example") if t in tools]
    prompt_order = [p for p in ("terse", "terse_fewshot", "detailed", "detailed_fewshot")
                    if p in prompts]

    out: dict[str, Any] = {
        "cells": [c.to_json() for c in summarise_cells(df)],
        "main_effect_tool_surface": main_effect(df, "tool_variant"),
        "main_effect_prompt_surface": main_effect(df, "prompt_variant"),
        "prompt_subfactors": prompt_subfactors(df),
        "selection_by_category": by_category(df),
        "argument_correctness_overall": _arg_overall(df),
        "incomplete_cells": [
            f"{r.tool_variant}|{r.prompt_variant}"
            for r in summarise_cells(df) if not r.complete
        ],
    }

    for metric in ("wrong_tool_calls", "unnecessary_calls"):
        out[f"main_effect_tool_surface__{metric}"] = main_effect(df, "tool_variant", metric)
        out[f"main_effect_prompt_surface__{metric}"] = main_effect(df, "prompt_variant", metric)

    if len(tool_order) >= 2 and len(prompt_order) >= 2:
        out["interaction"] = interaction(
            df, tool_order[0], tool_order[-1], prompt_order[0], prompt_order[-1]
        )
    else:
        out["interaction"] = {"available": False,
                              "reason": "grid has fewer than two levels on an axis"}
    return out


def _arg_overall(df: pd.DataFrame) -> dict[str, Any]:
    correct = df[df[SELECTION] == True]  # noqa: E712
    arg = correct["argument_correct"].dropna()
    return {
        "runs_with_correct_tool": int(len(correct)),
        "of_those_arguments_correct": int(arg.sum()) if len(arg) else 0,
        "rate": round(float(arg.mean()), 4) if len(arg) else None,
    }
