"""Statistical power for the ablation design.

This module answers a question the rest of the harness quietly assumes: with
twelve cells and four tasks each, what size of effect could this design
actually detect? The answer turns the limitations paragraph from an apology
into a number.

Everything here is computed from the config, not hardcoded, so changing
`tasks_per_cell` or `repeats` changes the reported power rather than leaving
a stale claim in the README.

No scipy. `statistics.NormalDist` provides the normal CDF and its inverse and
`math.comb` provides exact binomial tails, which is everything these
calculations need.

Five things are reported:

  MDE           the smallest difference detectable at alpha/power
  required n    how many tasks per cell to detect a given difference
  interaction   the same, for a difference-of-differences, which needs
                roughly 4x the sample of a main effect of the same size
  pairing       the gain from McNemar on the shared task suite, which this
                design gets for free and currently throws away
  multiplicity  what happens to the error rate if all 66 pairwise cell
                comparisons are inspected
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

ND = NormalDist()

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def _z(p: float) -> float:
    return ND.inv_cdf(p)


# ---------------------------------------------------------------------------
# Two-proportion comparisons (unpaired)
# ---------------------------------------------------------------------------


def two_proportion_power(p1: float, p2: float, n_per_group: int,
                         alpha: float = DEFAULT_ALPHA) -> float:
    """Power of a two-sided test of p1 vs p2 with n per group.

    Normal approximation. At the sample sizes in the quick tier the
    approximation is poor in absolute terms — but it is *optimistic*, which is
    the safe direction for a claim of the form "this design cannot detect
    anything smaller than X".
    """
    if n_per_group <= 0:
        return 0.0
    diff = abs(p1 - p2)
    if diff == 0:
        return alpha

    pbar = (p1 + p2) / 2.0
    se_null = math.sqrt(2.0 * pbar * (1.0 - pbar) / n_per_group)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)
    if se_alt == 0:
        return 1.0

    crit = _z(1 - alpha / 2.0) * se_null
    # Two-sided: the far tail contributes negligibly but is included for
    # correctness at small effect sizes.
    upper = ND.cdf((diff - crit) / se_alt)
    lower = ND.cdf((-diff - crit) / se_alt)
    return max(0.0, min(1.0, upper + lower))


def required_n(p1: float, p2: float, alpha: float = DEFAULT_ALPHA,
               power: float = DEFAULT_POWER, cap: int = 100_000) -> int | None:
    """Smallest n per group reaching `power`. None if `cap` is not enough."""
    if p1 == p2:
        return None
    lo, hi = 1, 64
    while hi < cap and two_proportion_power(p1, p2, hi, alpha) < power:
        lo, hi = hi, hi * 2
    if two_proportion_power(p1, p2, hi, alpha) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if two_proportion_power(p1, p2, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def mde(n_per_group: int, baseline: float = 0.5, alpha: float = DEFAULT_ALPHA,
        power: float = DEFAULT_POWER, step: float = 0.005) -> float | None:
    """Minimum detectable effect: smallest |p2 - baseline| reaching `power`.

    Searched upward from zero rather than solved in closed form, because the
    closed form assumes a pooled variance that is wrong near the boundaries,
    and this design spends most of its time near a boundary.
    """
    d = step
    while baseline + d <= 1.0 or baseline - d >= 0.0:
        for p2 in ((baseline + d) if baseline + d <= 1.0 else None,
                   (baseline - d) if baseline - d >= 0.0 else None):
            if p2 is None:
                continue
            if two_proportion_power(baseline, p2, n_per_group, alpha) >= power:
                return round(d, 4)
        d += step
        if d > 1.0:
            break
    return None


def attainable_outcomes(n: int) -> list[float]:
    """The only values a cell's rate can take at this n.

    With four binary tasks a cell is one of {0, .25, .5, .75, 1}. Any claimed
    difference smaller than the gap between adjacent values is not merely
    underpowered, it is unobservable.
    """
    return [round(k / n, 4) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# Paired comparisons (McNemar) — what the shared task suite buys
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant counts b and c.

    b and c are the tasks that flipped one way and the other between two
    cells. Concordant tasks carry no information about a difference, which is
    exactly why pairing helps: the test conditions on n = b + c instead of on
    the full sample.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_power(p_discordant: float, p_flip_given_discordant: float,
                  n_tasks: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Approximate power of McNemar.

    `p_discordant` is how often the two cells disagree on a task at all;
    `p_flip_given_discordant` is how asymmetric those disagreements are. The
    test's effective sample is n_tasks * p_discordant, which is why a paired
    design with few tasks can still beat an unpaired one: it spends all of its
    sample on the informative cases.
    """
    n_disc = n_tasks * p_discordant
    if n_disc <= 0:
        return 0.0
    psi = p_flip_given_discordant
    if psi == 0.5:
        return alpha
    se_null = 0.5 * math.sqrt(n_disc)
    se_alt = math.sqrt(n_disc * psi * (1 - psi))
    if se_alt == 0:
        return 1.0
    crit = _z(1 - alpha / 2.0) * se_null
    centre = abs(psi - 0.5) * n_disc
    return max(0.0, min(1.0, ND.cdf((centre - crit) / se_alt)))


def pairing_gain(p1: float, p2: float, n_tasks: int,
                 correlation: float = 0.7,
                 alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Unpaired vs paired power on the same data.

    `correlation` is how much the two cells agree beyond chance — the same
    task is easy or hard in both. Higher correlation means fewer discordant
    pairs, which sounds bad and is not: it concentrates the signal.

    Derived rather than assumed where possible. The discordant rate is bounded
    below by |p1 - p2| and above by 1 - |correlation adjustment|; this uses a
    standard tetrachoric-style approximation adequate for a design note.
    """
    diff = abs(p1 - p2)
    # Minimum possible discordance is the observed difference; correlation
    # shrinks the excess above that minimum.
    max_extra = min(p1, p2, 1 - p1, 1 - p2) * 2.0
    p_disc = diff + (1.0 - correlation) * max_extra
    p_disc = max(diff, min(1.0, p_disc))
    psi = 0.5 if p_disc == 0 else min(1.0, max(0.0, 0.5 + (diff / p_disc) / 2.0))

    unpaired = two_proportion_power(p1, p2, n_tasks, alpha)
    paired = mcnemar_power(p_disc, psi, n_tasks, alpha)
    n_disc = n_tasks * p_disc

    # Below a handful of discordant pairs neither test can reject anything and
    # both sit at the alpha floor, where their difference is approximation
    # noise rather than a real ordering. Saying "pairing lost 2 points" there
    # would be reporting an artefact as a finding.
    meaningful = n_disc >= 5 and max(unpaired, paired) > 2 * alpha

    return {
        "p1": p1,
        "p2": p2,
        "n_tasks": n_tasks,
        "assumed_correlation": correlation,
        "implied_discordant_rate": round(p_disc, 4),
        "expected_discordant_pairs": round(n_disc, 2),
        "unpaired_power": round(unpaired, 4),
        "paired_power_mcnemar": round(paired, 4),
        "gain": round(paired - unpaired, 4) if meaningful else None,
        "meaningful": meaningful,
        "note": (
            "" if meaningful else
            f"only ~{n_disc:.1f} discordant pairs expected; both tests are at "
            f"the alpha floor and the comparison is not informative"
        ),
    }


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------


def main_effect_power(effect: float, n_per_cell: int, baseline: float = 0.5,
                      alpha: float = DEFAULT_ALPHA) -> float:
    """Power for a main effect estimated from a 2x2, averaging over the other axis.

    The contrast is (a1b1 + a1b2)/2 - (a2b1 + a2b2)/2, whose variance is
    (1/4) x 4 x var_cell = var_cell. That is the right comparator for
    `interaction_power` below, and it is *not* the same as a plain two-group
    test: averaging two cells per side halves the variance, so a two-group
    calculation understates how well the grid estimates a main effect and
    correspondingly understates the interaction's disadvantage.
    """
    if n_per_cell <= 0 or effect == 0:
        return alpha
    se = math.sqrt(baseline * (1 - baseline) / n_per_cell)
    crit = _z(1 - alpha / 2.0) * se
    return max(0.0, min(1.0, ND.cdf((abs(effect) - crit) / se)
                        + ND.cdf((-abs(effect) - crit) / se)))


def required_n_main_effect(effect: float, baseline: float = 0.5,
                           alpha: float = DEFAULT_ALPHA,
                           power: float = DEFAULT_POWER,
                           cap: int = 100_000) -> int | None:
    lo, hi = 1, 64
    while hi < cap and main_effect_power(effect, hi, baseline, alpha) < power:
        lo, hi = hi, hi * 2
    if main_effect_power(effect, hi, baseline, alpha) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if main_effect_power(effect, mid, baseline, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def interaction_power(effect: float, n_per_cell: int, baseline: float = 0.5,
                      alpha: float = DEFAULT_ALPHA) -> float:
    """Power to detect a difference-of-differences of size `effect`.

    An interaction is (d - c) - (b - a) across four cells, with variance
    4 x var_cell against the main effect contrast's var_cell — so its standard
    error is exactly twice as large, and matching power costs four times the
    sample. That is the origin of the "interactions need 4x" rule, and
    `analyse_design` reports the ratio it actually computes rather than
    citing the folklore.
    """
    if n_per_cell <= 0 or effect == 0:
        return alpha
    var_cell = baseline * (1 - baseline) / n_per_cell
    se = math.sqrt(4.0 * var_cell)
    crit = _z(1 - alpha / 2.0) * se
    return max(0.0, min(1.0, ND.cdf((abs(effect) - crit) / se)
                        + ND.cdf((-abs(effect) - crit) / se)))


def required_n_interaction(effect: float, baseline: float = 0.5,
                           alpha: float = DEFAULT_ALPHA,
                           power: float = DEFAULT_POWER,
                           cap: int = 100_000) -> int | None:
    lo, hi = 1, 64
    while hi < cap and interaction_power(effect, hi, baseline, alpha) < power:
        lo, hi = hi, hi * 2
    if interaction_power(effect, hi, baseline, alpha) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if interaction_power(effect, mid, baseline, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def family_wise_error(n_comparisons: int, alpha: float = DEFAULT_ALPHA) -> float:
    """P(at least one false positive) across independent comparisons."""
    return 1.0 - (1.0 - alpha) ** n_comparisons


def holm_threshold(n_comparisons: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The most stringent Holm threshold, i.e. what the smallest p must beat."""
    return alpha / n_comparisons if n_comparisons else alpha


def pairwise_count(cells: int) -> int:
    return cells * (cells - 1) // 2


# ---------------------------------------------------------------------------
# Variance from an observed repeat study
# ---------------------------------------------------------------------------


@dataclass
class VarianceReport:
    cell: str
    repeats: int
    tasks: int
    selection_flip_rate: float
    argument_flip_rate: float
    per_task_selection: dict[str, list[bool]]
    per_task_argument: dict[str, list[Any]]
    deterministic: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "repeats": self.repeats,
            "tasks": self.tasks,
            "selection_flip_rate": round(self.selection_flip_rate, 4),
            "argument_flip_rate": round(self.argument_flip_rate, 4),
            "deterministic": self.deterministic,
            "per_task_selection": {k: v for k, v in self.per_task_selection.items()},
            "per_task_argument": {k: v for k, v in self.per_task_argument.items()},
        }


def summarise_variance(rows: Iterable[dict[str, Any]], cell: str) -> VarianceReport:
    """Flip rate across repeats of identical input.

    A task "flips" when its outcome is not the same in every repeat. Reported
    as the fraction of tasks that flipped at all, because that is what decides
    whether `repeats: 1` is defensible — one unstable task in the suite is
    enough to make a single-repeat cell unreliable.
    """
    sel: dict[str, list[bool]] = {}
    arg: dict[str, list[Any]] = {}
    for r in rows:
        sel.setdefault(r["task_id"], []).append(bool(r["correct_selection"]))
        arg.setdefault(r["task_id"], []).append(r["argument_correct"])

    def flipped(values: list[Any]) -> bool:
        seen = [v for v in values if v is not None]
        return len(set(seen)) > 1

    sel_flips = sum(1 for v in sel.values() if flipped(v))
    arg_flips = sum(1 for v in arg.values() if flipped(v))
    n_tasks = len(sel)
    repeats = max((len(v) for v in sel.values()), default=0)

    return VarianceReport(
        cell=cell,
        repeats=repeats,
        tasks=n_tasks,
        selection_flip_rate=sel_flips / n_tasks if n_tasks else 0.0,
        argument_flip_rate=arg_flips / n_tasks if n_tasks else 0.0,
        per_task_selection=sel,
        per_task_argument=arg,
        deterministic=(sel_flips == 0 and arg_flips == 0),
    )


# ---------------------------------------------------------------------------
# The whole analysis, from a config
# ---------------------------------------------------------------------------


def analyse_design(
    *,
    cells: int,
    tasks_per_cell: int,
    repeats: int,
    baseline_rate: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    correlation: float = 0.7,
    label: str = "",
) -> dict[str, Any]:
    n = tasks_per_cell * repeats
    # A main effect averages over the other axis, so its n is the cell n times
    # the number of levels it averages over. For a 3x4 grid, a tool-surface
    # level is backed by 4 cells.
    levels_other_axis = 4 if cells % 4 == 0 else max(1, cells // 3)
    n_main = n * levels_other_axis

    out: dict[str, Any] = {
        "label": label,
        "cells": cells,
        "tasks_per_cell": tasks_per_cell,
        "repeats": repeats,
        "runs_per_cell": n,
        "total_runs": cells * n,
        "alpha": alpha,
        "target_power": power,
        "attainable_cell_rates": attainable_outcomes(n),
        "granularity": round(1.0 / n, 4) if n else None,
    }

    out["mde_per_cell"] = mde(n, baseline_rate, alpha, power)
    out["mde_main_effect"] = mde(n_main, baseline_rate, alpha, power)
    out["n_backing_a_main_effect_level"] = n_main

    # Two different "required n" questions, both worth reporting. The
    # two-group figure is what a single pair of cells needs; the main-effect
    # figure is what the grid needs when it averages over the other axis, and
    # is the honest comparator for the interaction below.
    out["required_n_two_group"] = {
        "10_points": required_n(baseline_rate, baseline_rate + 0.10, alpha, power),
        "20_points": required_n(baseline_rate, baseline_rate + 0.20, alpha, power),
        "30_points": required_n(baseline_rate, baseline_rate + 0.30, alpha, power),
    }
    out["required_n_per_cell"] = {
        "10_points": required_n_main_effect(0.10, baseline_rate, alpha, power),
        "20_points": required_n_main_effect(0.20, baseline_rate, alpha, power),
        "30_points": required_n_main_effect(0.30, baseline_rate, alpha, power),
    }
    out["main_effect_power_at_current_n"] = {
        "10_points": round(main_effect_power(0.10, n, baseline_rate, alpha), 4),
        "20_points": round(main_effect_power(0.20, n, baseline_rate, alpha), 4),
        "30_points": round(main_effect_power(0.30, n, baseline_rate, alpha), 4),
    }

    out["interaction"] = {
        "power_at_current_n": {
            "10_points": round(interaction_power(0.10, n, baseline_rate, alpha), 4),
            "20_points": round(interaction_power(0.20, n, baseline_rate, alpha), 4),
            "30_points": round(interaction_power(0.30, n, baseline_rate, alpha), 4),
        },
        "required_n_per_cell": {
            "10_points": required_n_interaction(0.10, baseline_rate, alpha, power),
            "20_points": required_n_interaction(0.20, baseline_rate, alpha, power),
            "30_points": required_n_interaction(0.30, baseline_rate, alpha, power),
        },
    }
    main_20 = out["required_n_per_cell"]["20_points"]
    int_20 = out["interaction"]["required_n_per_cell"]["20_points"]
    out["interaction"]["sample_penalty_vs_main_effect"] = (
        round(int_20 / main_20, 2) if (main_20 and int_20) else None
    )

    out["pairing"] = [
        pairing_gain(baseline_rate, baseline_rate + d, n, correlation, alpha)
        for d in (0.10, 0.20, 0.30)
    ]

    k = pairwise_count(cells)
    out["multiplicity"] = {
        "pairwise_comparisons": k,
        "family_wise_error_uncorrected": round(family_wise_error(k, alpha), 4),
        "holm_most_stringent_threshold": round(holm_threshold(k, alpha), 6),
        "prespecified_tests": 3,
        "family_wise_error_prespecified_only": round(family_wise_error(3, alpha), 4),
    }
    return out


def headline(design: dict[str, Any], full: dict[str, Any] | None = None) -> str:
    """The one-paragraph summary, in the form the deliverable asked for."""
    q_mde_cell = design["mde_per_cell"]
    q_mde_main = design["mde_main_effect"]

    if q_mde_cell is None:
        cell_txt = (
            "detects nothing at all within a single cell - no effect size, "
            "however large, reaches 80% power on "
            f"{design['runs_per_cell']} runs"
        )
    else:
        cell_txt = f"detects nothing smaller than {q_mde_cell * 100:.0f} points within a cell"

    main_txt = ("no effect size" if q_mde_main is None
                else f"{q_mde_main * 100:.0f} points")

    parts = [
        f"The quick tier ({design['cells']} cells x {design['tasks_per_cell']} tasks "
        f"x {design['repeats']} repeat, {design['total_runs']} runs) {cell_txt}, "
        f"and needs {main_txt} for a main effect averaged over the other axis. "
        f"A cell's rate can only take the values "
        f"{', '.join(f'{v:.2f}' for v in design['attainable_cell_rates'])}, so "
        f"anything finer than {design['granularity'] * 100:.0f} points is "
        f"unobservable rather than merely underpowered."
    ]

    if full:
        fi = full["interaction"]["power_at_current_n"]["20_points"]
        fm = full["main_effect_power_at_current_n"]["20_points"]
        parts.append(
            f"configs/full.yaml gives {fm:.0%} power for a 20-point main effect "
            f"and {fi:.0%} for a 20-point interaction - the interaction is the "
            f"binding constraint, needing four times the sample for the same size."
        )
        fpg = full["pairing"][1]
        if fpg.get("meaningful"):
            parts.append(
                f"Pairing across the shared task suite raises that 20-point "
                f"comparison from {fpg['unpaired_power']:.0%} to "
                f"{fpg['paired_power_mcnemar']:.0%} at no extra cost, since the "
                f"same tasks run in every cell."
            )
    return " ".join(parts)
