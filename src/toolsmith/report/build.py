"""Build reports/evaluation.html from whatever results exist.

One self-contained file with Plotly inlined, committed with real results so a
visitor sees the findings without running anything. Nothing here estimates: a
section whose input is missing says so and renders nothing.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from ..config import REPORTS_DIR, RESULTS_DIR

INK = "#1f2328"
MUTED = "#636c76"
RULE = "#d8dee4"
GOOD = "#2f6f4f"
BAD = "#a3382c"

CSS = f"""
:root {{ --ink:{INK}; --muted:{MUTED}; --rule:{RULE}; --good:{GOOD}; --bad:{BAD}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2.4rem 1.4rem 5rem; background:#fbfbfa; color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.8rem; margin:0 0 .2rem; letter-spacing:-.015em; }}
h2 {{ font-size:1.15rem; margin:2.6rem 0 .5rem; padding-bottom:.3rem;
  border-bottom:1px solid var(--rule); }}
h3 {{ font-size:.82rem; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin:1.6rem 0 .4rem; }}
p {{ max-width:76ch; margin:.6rem 0; }}
.sub {{ color:var(--muted); margin:0 0 1.6rem; font-size:.9rem; }}
table {{ border-collapse:collapse; width:100%; margin:.7rem 0 1.2rem; font-size:.88rem; }}
th,td {{ text-align:right; padding:.38rem .6rem; border-bottom:1px solid var(--rule); }}
th:first-child, td:first-child {{ text-align:left; }}
thead th {{ font-weight:600; color:var(--muted); font-size:.76rem;
  text-transform:uppercase; letter-spacing:.05em; }}
tbody tr:hover {{ background:#f2f2f0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:.7rem; margin:1rem 0 1.4rem; }}
.card {{ background:#fff; border:1px solid var(--rule); border-radius:6px; padding:.75rem .9rem; }}
.card .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
.card .v {{ font-size:1.5rem; font-variant-numeric:tabular-nums; margin-top:.1rem; }}
.card .n {{ font-size:.75rem; color:var(--muted); }}
.note {{ background:#fff; border-left:3px solid var(--muted); padding:.65rem .9rem;
  margin:1rem 0; font-size:.9rem; }}
.bad {{ border-left-color:var(--bad); }}
.good {{ border-left-color:var(--good); }}
pre {{ background:#f4f4f2; border:1px solid var(--rule); border-radius:5px;
  padding:.6rem .8rem; overflow-x:auto; font-size:.78rem; line-height:1.45; }}
code {{ font:.82em ui-monospace,SFMono-Regular,Menlo,monospace; background:#f0f0ee;
  padding:.05rem .3rem; border-radius:3px; }}
.chart {{ background:#fff; border:1px solid var(--rule); border-radius:6px;
  padding:.5rem; margin:.8rem 0; }}
footer {{ margin-top:3.5rem; color:var(--muted); font-size:.8rem; }}
"""


def _load(name: str) -> dict[str, Any] | None:
    p = Path(RESULTS_DIR) / f"{name}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def esc(v: Any) -> str:
    return html.escape(str(v))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if isinstance(c, str) and c.startswith('<') else esc(c)}</td>"
                         for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def card(k: str, v: Any, n: str = "") -> str:
    return (f'<div class="card"><div class="k">{esc(k)}</div>'
            f'<div class="v">{esc(v)}</div><div class="n">{esc(n)}</div></div>')


def pct(x: float | None) -> str:
    return "--" if x is None else f"{x * 100:.1f}%"


def fig_html(fig: go.Figure, height: int = 380) -> str:
    fig.update_layout(
        height=height, margin=dict(l=60, r=24, t=34, b=46),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="-apple-system,Segoe UI,Roboto,sans-serif", size=12, color=INK),
    )
    return f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


# --------------------------------------------------------------------------


def _heatmap(ablation: dict[str, Any]) -> str:
    cells = ablation["cells"]
    tools = sorted({c["tool_variant"] for c in cells})
    prompts = sorted({c["prompt_variant"] for c in cells})
    z, text = [], []
    for p in prompts:
        row, trow = [], []
        for t in tools:
            c = next((x for x in cells if x["tool_variant"] == t and x["prompt_variant"] == p),
                     None)
            row.append(None if c is None else c["selection_rate"] * 100)
            trow.append("" if c is None else f"{c['selection_rate'] * 100:.0f}%")
        z.append(row)
        text.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z, x=tools, y=prompts, text=text, texttemplate="%{text}",
        colorscale=[[0, "#f7f4ef"], [0.5, "#9fc0a8"], [1, "#2f6f4f"]],
        zmin=0, zmax=100, colorbar=dict(title="selection %"),
    ))
    fig.update_layout(title="Correct tool selection by cell",
                      xaxis_title="tool surface", yaxis_title="prompt surface")
    return fig_html(fig, 340)


def _main_effects(ablation: dict[str, Any]) -> str:
    tool = ablation["main_effect_tool_surface"]
    prompt = ablation["main_effect_prompt_surface"]
    fig = go.Figure()
    fig.add_bar(x=list(tool), y=[v * 100 for v in tool.values()],
                name="tool surface", marker_color="#3a5a8c")
    fig.add_bar(x=list(prompt), y=[v * 100 for v in prompt.values()],
                name="prompt surface", marker_color="#7a3a6a")
    fig.update_layout(title="Main effects (averaged over the other axis)",
                      yaxis_title="selection accuracy %", barmode="group",
                      legend=dict(orientation="h", y=1.12))
    return fig_html(fig, 340)


def _interaction(ablation: dict[str, Any]) -> str:
    inter = ablation.get("interaction") or {}
    if not inter.get("available"):
        return ""
    c = inter["contrast"]
    cells = inter["cells"]
    fig = go.Figure()
    for tool in (c["tool_low"], c["tool_high"]):
        ys = [cells.get(f"{tool}|{c['prompt_low']}"), cells.get(f"{tool}|{c['prompt_high']}")]
        fig.add_scatter(x=[c["prompt_low"], c["prompt_high"]],
                        y=[(y or 0) * 100 for y in ys],
                        mode="lines+markers", name=f"{tool} tools", line=dict(width=3))
    fig.update_layout(title="Interaction: does the tool axis still matter at a better prompt?",
                      yaxis_title="selection accuracy %", xaxis_title="prompt surface",
                      legend=dict(orientation="h", y=1.12))
    return fig_html(fig, 340)


def _tradeoffs(ablation: dict[str, Any]) -> str:
    cells = ablation["cells"]
    out = []
    for xkey, xlabel, title in (
        ("latency_p50_s", "latency p50 (s)", "Accuracy against latency"),
        ("total_tokens_mean", "mean total tokens", "Accuracy against tokens"),
    ):
        fig = go.Figure()
        for t in sorted({c["tool_variant"] for c in cells}):
            sel = [c for c in cells if c["tool_variant"] == t]
            fig.add_scatter(
                x=[c[xkey] for c in sel], y=[c["selection_rate"] * 100 for c in sel],
                mode="markers+text", name=f"{t} tools",
                text=[c["prompt_variant"].replace("_fewshot", "+fs") for c in sel],
                textposition="top center", marker=dict(size=12),
            )
        fig.update_layout(title=title, xaxis_title=xlabel,
                          yaxis_title="selection accuracy %",
                          legend=dict(orientation="h", y=1.12))
        out.append(fig_html(fig, 360))
    return "".join(out)


# --------------------------------------------------------------------------


def build() -> Path:
    baseline = _load("baseline")
    ablation = _load("ablation")
    adversarial = _load("adversarial")
    determinism = _load("determinism")

    parts: list[str] = [
        "<!doctype html><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Toolsmith evaluation</title>",
        f"<style>{CSS}</style>",
        f"<script>{get_plotlyjs()}</script>",
        "<div class='wrap'>",
        "<h1>Toolsmith</h1>",
    ]

    model = (ablation or baseline or {}).get("model") or (
        (ablation or {}).get("meta", {}).get("model")
    )
    meta = (ablation or {}).get("meta", {})
    parts.append(
        f"<p class='sub'>Two-factor ablation on tool surface and prompt surface &middot; "
        f"{esc(model or 'model unknown')} &middot; local inference, no hosted API</p>"
    )

    # ---- capability baseline -------------------------------------------
    parts.append("<h2>Capability baseline</h2>")
    if not baseline:
        parts.append("<p class='note'>No baseline recorded. Run <code>make baseline</code>.</p>")
    else:
        verdict_cls = {"usable": "good", "at_or_below_chance": "bad",
                       "ceiling": ""}.get(baseline["verdict"], "")
        parts.append("<div class='cards'>")
        parts.append(card("selection", pct(baseline["selection_rate"]),
                          f"chance {pct(baseline['chance_rate'])}"))
        parts.append(card("arguments", pct(baseline.get("argument_rate")),
                          "given the right tool"))
        parts.append(card("completion", pct(baseline.get("completion_rate")), ""))
        parts.append(card("verdict", baseline["verdict"].replace("_", " "),
                          f"{baseline['tasks']} tasks"))
        parts.append("</div>")
        parts.append(
            f"<p class='note {verdict_cls}'>This gate runs before any ablation. If the model "
            f"cannot perform the task with the best configuration available, every variant "
            f"scores near chance and the grid measures noise.</p>"
        )
        parts.append(table(["category", "selection accuracy"],
                           [[k, pct(v)] for k, v in baseline["by_category"].items()]))

    # ---- the grid -------------------------------------------------------
    parts.append("<h2>Two-factor results</h2>")
    if not ablation:
        parts.append("<p class='note'>No ablation recorded. Run <code>make ablate</code>.</p>")
    else:
        te = ablation["main_effect_tool_surface"]
        pe = ablation["main_effect_prompt_surface"]
        inter = ablation.get("interaction") or {}
        parts.append("<div class='cards'>")
        parts.append(card("cells", len(ablation["cells"]),
                          f"{meta.get('total_runs','?')} runs"))
        parts.append(card("tool axis spread",
                          f"{(max(te.values()) - min(te.values())) * 100:.1f} pts",
                          f"{min(te, key=te.get)} to {max(te, key=te.get)}"))
        parts.append(card("prompt axis spread",
                          f"{(max(pe.values()) - min(pe.values())) * 100:.1f} pts",
                          f"{min(pe, key=pe.get)} to {max(pe, key=pe.get)}"))
        if inter.get("available"):
            parts.append(card("interaction", f"{inter['interaction'] * 100:+.1f} pts",
                              "tool effect, high minus low prompt"))
        parts.append("</div>")

        parts.append(_heatmap(ablation))
        parts.append(_main_effects(ablation))
        parts.append(_interaction(ablation))

        parts.append("<h3>Per cell, with the quality cost alongside</h3>")
        rows = []
        for c in sorted(ablation["cells"], key=lambda x: -x["selection_rate"]):
            rows.append([
                c["tool_variant"], c["prompt_variant"], c["n"],
                pct(c["selection_rate"]), pct(c.get("argument_rate")),
                pct(c.get("completion_rate")), pct(c["wrong_tool_rate"]),
                pct(c["unnecessary_rate"]),
                f"{c['latency_p50_s']:.1f}", f"{c['latency_p99_s']:.1f}",
                f"{c['prompt_tokens_mean']:.0f}", f"{c['total_tokens_mean']:.0f}",
                "yes" if c["complete"] else "NO",
            ])
        parts.append(table(
            ["tools", "prompt", "n", "selection", "arguments", "completion",
             "wrong tool", "unnecessary", "p50 s", "p99 s", "prompt tok",
             "total tok", "complete"],
            rows))

        sub = ablation["prompt_subfactors"]
        parts.append("<h3>The prompt axis is two factors, not one</h3>")
        parts.append(table(
            ["factor", "level", "selection accuracy"],
            [["base prose", k, pct(v)] for k, v in sub["base"].items()] +
            [["few-shot examples", k, pct(v)] for k, v in sub["fewshot"].items()]
        ))

        arg = ablation["argument_correctness_overall"]
        parts.append("<h3>Argument correctness, scored separately</h3>")
        parts.append(
            f"<p>Of {arg['runs_with_correct_tool']} runs that selected the right tool, "
            f"{arg['of_those_arguments_correct']} also got the arguments right "
            f"({pct(arg['rate'])}). This is scored only on correctly-selected runs so it "
            f"isolates the failure where a model picks correctly and then fills the call "
            f"in wrongly — a failure that a selection-accuracy number alone cannot see.</p>"
        )

        parts.append("<h3>Accuracy is not free</h3>")
        parts.append(_tradeoffs(ablation))

        if ablation.get("incomplete_cells"):
            parts.append(
                f"<p class='note bad'>Incomplete cells, excluded from conclusions: "
                f"{esc(', '.join(ablation['incomplete_cells']))}</p>"
            )

    # ---- guardrails -----------------------------------------------------
    parts.append("<h2>Guardrails under attack</h2>")
    if not adversarial:
        parts.append("<p class='note'>No adversarial run recorded. "
                     "Run <code>make adversarial</code>.</p>")
    else:
        a = adversarial
        parts.append("<div class='cards'>")
        parts.append(card("attacks", a["attacks"], ""))
        parts.append(card("blocked by a guardrail", a["blocked"], pct(a["block_rate"])))
        parts.append(card("refused by the model", a["model_refused"],
                          "not a guardrail win"))
        parts.append(card("bypassed", a["bypassed"],
                          "reached the objective"))
        parts.append(card("false positives", a["false_positives"].get("wrongly_blocked", "--"),
                          f"of {a['false_positives'].get('legitimate_tasks','?')} "
                          f"legitimate tasks"))
        parts.append("</div>")

        if a["bypassed"]:
            parts.append("<h3>What got through</h3>")
            parts.append(
                "<p>These are the result. Each one reached the attacker's objective; the "
                "layer that should have stopped it is named alongside.</p>"
            )
            for b in a["bypasses"]:
                parts.append(
                    f"<p><strong>{esc(b['id'])}</strong> &middot; {esc(b['category'])} "
                    f"&middot; expected to be caught by "
                    f"<code>{esc(b['expected_layer'])}</code></p>"
                )
                parts.append(f"<pre>{esc(b['text'])}</pre>")
                parts.append(f"<pre>answer: {esc((b['final_answer'] or '')[:700])}</pre>")
                if b["calls"]:
                    parts.append(f"<pre>calls: {esc(json.dumps(b['calls'], indent=1)[:900])}</pre>")
        else:
            parts.append(
                "<p class='note bad'>Nothing evaded the guardrails. That is more likely to "
                "mean the suite was insufficiently adversarial than that the defence is "
                "complete — a 100% block rate reads as an untested guardrail. The next "
                "version of this suite needs attacks that do not name a forbidden tool or "
                "use override phrasing at all.</p>"
            )

        parts.append("<h3>Where the load falls</h3>")
        parts.append(table(["layer", "attacks it stopped first"],
                           [[k, v] for k, v in a["blocked_by_layer"].items()]))
        parts.append(table(["rule", "times fired"],
                           [[k, v] for k, v in list(a["blocked_by_rule"].items())[:12]]))

        parts.append(table(
            ["category", "blocked", "model refused", "bypassed"],
            [[k, v["blocked"], v["model_refused"], v["bypassed"]]
             for k, v in a["by_category"].items()]))

        fp = a["false_positives"]
        if fp.get("blocked_tasks"):
            parts.append("<h3>Legitimate tasks wrongly blocked</h3>")
            parts.append(table(["task", "rule", "text"],
                               [[t["task_id"], t.get("rule"), t["text"][:90]]
                                for t in fp["blocked_tasks"]]))
        else:
            parts.append(
                f"<p>No legitimate task was wrongly blocked "
                f"({fp.get('legitimate_tasks','?')} checked). A filter that blocks "
                f"everything would score a perfect block rate and be useless, which is "
                f"why this is reported next to it.</p>"
            )

        if a.get("layer_misattribution"):
            parts.append("<h3>Caught by a different layer than expected</h3>")
            parts.append(
                "<p>Not failures — information about which layer is carrying the load.</p>"
            )
            parts.append(table(["attack", "expected", "actual"],
                               [[m["id"], m["expected_layer"], m["actual_layer"]]
                                for m in a["layer_misattribution"]]))

    # ---- integrity ------------------------------------------------------
    parts.append("<h2>Run integrity</h2>")
    rows = []
    if meta:
        tok = meta.get("token_reconciliation", {})
        rows += [
            ["tasks per cell", meta.get("tasks_per_cell", "--")],
            ["repeats", meta.get("repeats", "--")],
            ["total runs", meta.get("total_runs", "--")],
            ["wall clock", f"{meta.get('wall_s', 0):.0f}s"],
            ["token accounting reconciles", "yes" if tok.get("reconciles") else "NO"],
            ["total tokens", f"{tok.get('total_tokens', 0):,}"],
        ]
    if determinism:
        rows += [
            ["corpus identical across runs",
             "yes" if determinism.get("corpus_identical") else "NO"],
            ["task selection stable",
             "yes" if determinism.get("task_selection_stable") else "NO"],
        ]
    parts.append(table(["check", "value"], rows) if rows
                 else "<p class='note'>No integrity data recorded.</p>")

    parts.append(
        "<footer>Every figure on this page comes from a run recorded in results/. "
        "Nothing is estimated or projected. Inference is local Ollama; no hosted API "
        "was used and there is no key anywhere in this project.</footer>"
    )
    parts.append("</div>")

    out_dir = Path(REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    path = build()
    size = path.stat().st_size
    print(f"report: wrote {path} ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
