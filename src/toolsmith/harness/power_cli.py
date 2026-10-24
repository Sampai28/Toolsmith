"""`make power` and `make variance`.

`variance` is the measurement everything else leans on: one cell, the task
suite repeated N times on identical input, reporting how often an outcome
flips. Until that number exists, `repeats: 1` is an assumption rather than a
decision.

`power` computes the design analysis from the configs. It reads the variance
study if one exists and says so when it does not, rather than quietly
assuming determinism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..config import RESULTS_DIR, ensure_dirs, load_config
from ..data import store
from ..guardrails.engine import GuardrailEngine
from . import power as P
from .client import OllamaClient
from .mcp_client import MCPSession
from .prompts import system_prompt
from .runner import run_task
from .suite import load_tasks, select

import os

OLLAMA_URL = os.environ.get("TOOLSMITH_OLLAMA_URL", "http://localhost:11434")


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, default=str, indent=2)
        fh.write("\n")
    return path


def cmd_variance(args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = load_config(args.config)
    tasks = select(load_tasks(), cfg.grid.tasks_per_cell)
    tool_variant = args.tool_variant or cfg.grid.tool_variants[-1]
    prompt_variant = args.prompt_variant or cfg.grid.prompt_variants[-1]
    cell = f"{tool_variant}|{prompt_variant}"

    print(f"variance: {cell}, {len(tasks)} tasks x {args.repeats} repeats "
          f"= {len(tasks) * args.repeats} runs on identical input")

    dataset = store.default_dataset()
    engine = GuardrailEngine(dataset=dataset)
    sys_prompt = system_prompt(prompt_variant)
    rows: list[dict[str, Any]] = []

    with OllamaClient(OLLAMA_URL, cfg.model) as client:
        if not client.wait_until_ready():
            print(f"variance: Ollama not reachable at {OLLAMA_URL}", file=sys.stderr)
            return 1
        client.warm()
        with MCPSession(tool_variant=tool_variant) as session:
            for rep in range(args.repeats):
                for task in tasks:
                    rec = run_task(
                        task, client=client, session=session, engine=engine,
                        system_prompt=sys_prompt, tool_variant=tool_variant,
                        prompt_variant=prompt_variant, repeat=rep,
                    )
                    s = rec.score
                    rows.append({
                        "repeat": rep,
                        "task_id": task.id,
                        "correct_selection": bool(s.correct_selection) if s else False,
                        "argument_correct": (s.argument_correct if s else None),
                        "tool_called": [c.canonical_name or c.requested_name
                                        for c in rec.calls],
                        "wall_s": round(rec.task_wall_s, 3),
                    })
                sel = sum(1 for r in rows[-len(tasks):] if r["correct_selection"])
                print(f"  repeat {rep + 1}/{args.repeats}: selection "
                      f"{sel}/{len(tasks)}", flush=True)

    report = P.summarise_variance(rows, cell)
    payload = report.to_json()
    payload["model"] = cfg.model.name
    payload["temperature"] = cfg.model.temperature
    payload["seed"] = cfg.model.seed
    payload["rows"] = rows
    path = _write(Path(RESULTS_DIR) / "variance.json", payload)

    print()
    print(f"variance: selection flipped on {report.selection_flip_rate:.0%} of tasks, "
          f"arguments on {report.argument_flip_rate:.0%}")
    if report.deterministic:
        print("variance: fully deterministic across repeats at temperature "
              f"{cfg.model.temperature}. repeats: 1 is defensible for this cell.")
    else:
        unstable = [t for t, v in report.per_task_selection.items()
                    if len(set(v)) > 1]
        print(f"variance: NOT deterministic. Unstable tasks: {', '.join(unstable)}")
        print("variance: repeats: 1 understates uncertainty; see results/power.json "
              "for the repeats needed.")
    print(f"variance: -> {path}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    ensure_dirs()
    quick = load_config(args.config)
    full_cfg = None
    full_path = Path("configs/full.yaml")
    if full_path.exists():
        full_cfg = load_config(full_path)

    all_tasks = load_tasks()
    quick_tasks = len(select(all_tasks, quick.grid.tasks_per_cell))

    # The baseline rate the design is tested against. Taken from the observed
    # ablation if there is one, because power at p=0.5 is the most optimistic
    # case and this design does not sit at 0.5.
    baseline_rate = 0.5
    source = "0.50 (no ablation yet; this is the most optimistic case)"
    abl_path = Path(RESULTS_DIR) / "ablation.json"
    if abl_path.exists():
        with abl_path.open(encoding="utf-8") as fh:
            abl = json.load(fh)
        rates = [c["selection_rate"] for c in abl.get("cells", [])]
        if rates:
            baseline_rate = round(sum(rates) / len(rates), 4)
            source = f"{baseline_rate:.2f} (mean cell selection rate from results/ablation.json)"

    design = P.analyse_design(
        cells=quick.cells, tasks_per_cell=quick_tasks, repeats=quick.grid.repeats,
        baseline_rate=baseline_rate, alpha=args.alpha, power=args.power,
        correlation=args.correlation, label="quick",
    )

    full_design = None
    if full_cfg is not None:
        full_tasks = len(select(all_tasks, full_cfg.grid.tasks_per_cell))
        full_design = P.analyse_design(
            cells=full_cfg.cells, tasks_per_cell=full_tasks,
            repeats=full_cfg.grid.repeats, baseline_rate=baseline_rate,
            alpha=args.alpha, power=args.power, correlation=args.correlation,
            label="full",
        )

    variance = None
    var_path = Path(RESULTS_DIR) / "variance.json"
    if var_path.exists():
        with var_path.open(encoding="utf-8") as fh:
            v = json.load(fh)
        variance = {k: v[k] for k in ("cell", "repeats", "tasks",
                                      "selection_flip_rate", "argument_flip_rate",
                                      "deterministic", "model", "temperature")}

    payload = {
        "baseline_rate": baseline_rate,
        "baseline_rate_source": source,
        "alpha": args.alpha,
        "target_power": args.power,
        "assumed_task_correlation": args.correlation,
        "variance_study": variance,
        "quick": design,
        "full": full_design,
        "headline": P.headline(design, full_design),
    }
    path = _write(Path(RESULTS_DIR) / "power.json", payload)

    # ---- printed report -------------------------------------------------
    d = design
    print(f"power: alpha={args.alpha}  target power={args.power}  "
          f"baseline rate={source}")
    print()

    if variance is None:
        print("  run-to-run variance   NOT MEASURED - run `make variance` first")
    else:
        print(f"  run-to-run variance   selection flips on "
              f"{variance['selection_flip_rate']:.0%} of tasks, arguments on "
              f"{variance['argument_flip_rate']:.0%} "
              f"({variance['repeats']} repeats, {variance['cell']})")
        print(f"                        deterministic: {variance['deterministic']}")
    print()

    print(f"  QUICK  {d['cells']} cells x {d['tasks_per_cell']} tasks x "
          f"{d['repeats']} repeat = {d['total_runs']} runs")
    print(f"    attainable cell rates   {', '.join(f'{v:.2f}' for v in d['attainable_cell_rates'])}")
    print(f"    granularity             {d['granularity']:.2f} - differences smaller "
          f"than this are unobservable, not merely underpowered")
    mdec = d["mde_per_cell"]
    print(f"    MDE within a cell       "
          f"{'unreachable at any effect size' if mdec is None else f'{mdec*100:.0f} points'}")
    mdem = d["mde_main_effect"]
    print(f"    MDE for a main effect   "
          f"{'unreachable' if mdem is None else f'{mdem*100:.0f} points'}  "
          f"(n={d['n_backing_a_main_effect_level']} backing each level)")
    print()
    print("    required tasks/cell to detect, at 80% power:")
    for k, v in d["required_n_per_cell"].items():
        print(f"      main effect, {k.replace('_',' '):<12} n = "
              f"{'not reachable' if v is None else v}")
    for k, v in d["interaction"]["required_n_per_cell"].items():
        print(f"      interaction, {k.replace('_',' '):<12} n = "
              f"{'not reachable' if v is None else v}")
    pen = d["interaction"]["sample_penalty_vs_main_effect"]
    if pen:
        print(f"      interaction costs {pen}x the sample of the same-sized main effect")
    print()
    print("    power at the CURRENT n:")
    for k, v in d["main_effect_power_at_current_n"].items():
        print(f"      main effect, {k.replace('_',' '):<12} {v:.1%}")
    for k, v in d["interaction"]["power_at_current_n"].items():
        print(f"      interaction, {k.replace('_',' '):<12} {v:.1%}")
    print()
    print("    pairing (McNemar on the shared task suite):")
    for pg in d["pairing"]:
        diff = abs(pg["p2"] - pg["p1"]) * 100
        if not pg.get("meaningful"):
            print(f"      {diff:.0f}-point difference: {pg['note']}")
            continue
        print(f"      {diff:.0f}-point difference: "
              f"unpaired {pg['unpaired_power']:.0%} -> paired "
              f"{pg['paired_power_mcnemar']:.0%}  ({pg['gain']*100:+.0f} pts)")
    print()
    m = d["multiplicity"]
    print(f"    multiplicity: {m['pairwise_comparisons']} pairwise comparisons")
    print(f"      family-wise error if all are inspected  "
          f"{m['family_wise_error_uncorrected']:.1%}")
    print(f"      Holm threshold for the smallest p       "
          f"{m['holm_most_stringent_threshold']:.5f}")
    print(f"      prespecified tests only (2 main + 1 int) "
          f"{m['family_wise_error_prespecified_only']:.1%}")

    if full_design:
        f = full_design
        print()
        print(f"  FULL   {f['cells']} cells x {f['tasks_per_cell']} tasks x "
              f"{f['repeats']} repeats = {f['total_runs']} runs")
        print(f"    power, 20-point main effect   "
              f"{f['main_effect_power_at_current_n']['20_points']:.0%}")
        print(f"    power, 20-point interaction   "
              f"{f['interaction']['power_at_current_n']['20_points']:.0%}")
        fpg = f["pairing"][1]
        if fpg.get("meaningful"):
            print(f"    paired, 20-point difference   "
                  f"{fpg['unpaired_power']:.0%} -> {fpg['paired_power_mcnemar']:.0%}"
                  f"  ({fpg['gain']*100:+.0f} pts)")
        else:
            print(f"    paired, 20-point difference   {fpg['note']}")

    print()
    print(payload["headline"])
    print(f"power: -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="toolsmith.harness.power_cli")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("variance", help="measure run-to-run variance on one cell")
    v.add_argument("--config", default="configs/quick.yaml")
    v.add_argument("--repeats", type=int, default=10)
    v.add_argument("--tool-variant", default=None)
    v.add_argument("--prompt-variant", default=None)
    v.set_defaults(func=cmd_variance)

    p = sub.add_parser("power", help="design power analysis from the configs")
    p.add_argument("--config", default="configs/quick.yaml")
    p.add_argument("--alpha", type=float, default=P.DEFAULT_ALPHA)
    p.add_argument("--power", type=float, default=P.DEFAULT_POWER)
    p.add_argument("--correlation", type=float, default=0.7,
                   help="assumed per-task agreement between two cells")
    p.set_defaults(func=cmd_power)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
