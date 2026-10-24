"""Harness entry points: `baseline`, `ablate`, `validate`, `determinism`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..config import RESULTS_DIR, ensure_dirs, load_config
from .baseline import run_baseline, verdict_text
from .grid import reconcile_tokens, rows, run_grid, write_raw
from .metrics import analyse
from .suite import load_tasks, select, validate_suite

OLLAMA_URL = os.environ.get("TOOLSMITH_OLLAMA_URL", "http://localhost:11434")


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, default=str, indent=2)
        fh.write("\n")
    return path


def cmd_validate(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    rep = validate_suite(tasks)
    print(f"validate-suite: {len(tasks)} tasks")
    if rep.ok:
        print("validate-suite: all gates passed")
    else:
        for rule, n in rep.counts.most_common():
            print(f"  {rule}: {n}")
        for rule, tid, detail in rep.violations[:20]:
            print(f"    {tid}: {rule} - {detail}")
    _write(RESULTS_DIR / "suite_validation.json", rep.to_json())
    return 0 if rep.ok else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = load_config(args.config)
    tasks = select(load_tasks(), cfg.baseline_tasks)

    rep = validate_suite(load_tasks())
    if not rep.ok:
        print("baseline: refusing to run, task suite failed validation", file=sys.stderr)
        for rule, tid, detail in rep.violations[:10]:
            print(f"  {tid}: {rule} - {detail}", file=sys.stderr)
        return 1

    print(f"baseline: {cfg.model.name}, {len(tasks)} tasks, "
          f"{cfg.baseline_tool_variant} tools + {cfg.baseline_prompt_variant} prompt")
    res = run_baseline(cfg, tasks, base_url=OLLAMA_URL)

    path = _write(RESULTS_DIR / "baseline.json", res.to_json())
    print()
    print(f"baseline: selection {res.selection_rate:.1%}  "
          f"arguments {('n/a' if res.argument_rate is None else f'{res.argument_rate:.1%}')}  "
          f"completion {('n/a' if res.completion_rate is None else f'{res.completion_rate:.1%}')}")
    print(f"baseline: by category {res.by_category}")
    print(f"baseline: {res.wall_s:.1f}s -> {path}")
    print()
    print(verdict_text(res))

    if res.verdict == "at_or_below_chance":
        # Hard gate. Refusing to proceed is the point of Step 0.
        return 2
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = load_config(args.config)

    baseline_path = RESULTS_DIR / "baseline.json"
    if not args.skip_gate:
        if not baseline_path.exists():
            print("ablate: no baseline.json; run `make baseline` first", file=sys.stderr)
            return 1
        with baseline_path.open(encoding="utf-8") as fh:
            bl = json.load(fh)
        if bl.get("verdict") == "at_or_below_chance":
            print("ablate: baseline is at or below chance; the ablation would measure "
                  "noise. Re-run with a larger model, or pass --skip-gate to override.",
                  file=sys.stderr)
            return 2
        print(f"ablate: baseline gate passed ({bl.get('selection_rate'):.1%} selection)")

    tasks = select(load_tasks(), cfg.grid.tasks_per_cell)
    total = cfg.cells * len(tasks) * cfg.grid.repeats
    print(f"ablate: {cfg.cells} cells x {len(tasks)} tasks x {cfg.grid.repeats} repeat(s) "
          f"= {total} runs, model {cfg.model.name}")

    result = run_grid(cfg, tasks, base_url=OLLAMA_URL)
    raw_path = write_raw(result)

    import pandas as pd

    df = pd.DataFrame(rows(result))
    report = analyse(df, cfg.grid.tool_variants, cfg.grid.prompt_variants)
    report["meta"] = {
        "config": cfg.name,
        "model": cfg.model.name,
        "tasks_per_cell": len(tasks),
        "repeats": cfg.grid.repeats,
        "total_runs": int(len(df)),
        "wall_s": round(result["wall_s"], 2),
        "task_fingerprint": result["task_fingerprint"],
        "token_reconciliation": reconcile_tokens(result),
    }
    path = _write(RESULTS_DIR / "ablation.json", report)

    print()
    print(f"ablate: {result['wall_s']:.1f}s total -> {raw_path}, {path}")
    print(f"ablate: main effect, tool surface   {report['main_effect_tool_surface']}")
    print(f"ablate: main effect, prompt surface {report['main_effect_prompt_surface']}")
    print(f"ablate: prompt subfactors           {report['prompt_subfactors']}")
    inter = report.get("interaction") or {}
    if inter.get("available"):
        print(f"ablate: interaction {inter['interaction']:+.3f} "
              f"(tool effect {inter['tool_effect_at_low_prompt']:+.3f} at low prompt, "
              f"{inter['tool_effect_at_high_prompt']:+.3f} at high prompt)")
    if report["incomplete_cells"]:
        print(f"ablate: INCOMPLETE cells {report['incomplete_cells']}")
    return 0


def cmd_determinism(args: argparse.Namespace) -> int:
    """Assert an identical config reproduces identical results.

    Scoped to the parts that must be deterministic by construction: corpus
    generation and task selection. Model sampling at temperature 0 is
    *expected* to be stable but is not guaranteed by the provider, so it is
    checked and reported rather than asserted.
    """
    from ..data.generate import generate

    cfg = load_config(args.config)
    import tempfile

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ma = generate(cfg.dataset, Path(a))
        mb = generate(cfg.dataset, Path(b))
        same = []
        for name in ("surveys", "questions", "respondents", "responses"):
            ta = (Path(a) / f"{name}.json").read_text(encoding="utf-8")
            tb = (Path(b) / f"{name}.json").read_text(encoding="utf-8")
            same.append(ta == tb)
        corpus_ok = all(same) and ma["counts"] == mb["counts"]

    tasks_a = [t.id for t in select(load_tasks(), cfg.grid.tasks_per_cell)]
    tasks_b = [t.id for t in select(load_tasks(), cfg.grid.tasks_per_cell)]
    tasks_ok = tasks_a == tasks_b

    print(f"determinism: corpus identical across two generations: {corpus_ok}")
    print(f"determinism: task selection stable: {tasks_ok} ({len(tasks_a)} tasks)")
    _write(RESULTS_DIR / "determinism.json", {
        "corpus_identical": corpus_ok,
        "task_selection_stable": tasks_ok,
        "task_ids": tasks_a,
        "seed": cfg.dataset.seed,
    })
    return 0 if (corpus_ok and tasks_ok) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="toolsmith.harness", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in (("validate", cmd_validate), ("baseline", cmd_baseline),
                     ("ablate", cmd_ablate), ("determinism", cmd_determinism)):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/quick.yaml")
        if name == "ablate":
            p.add_argument("--skip-gate", action="store_true",
                           help="run the grid even if the baseline gate failed")
        p.set_defaults(func=fn)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
