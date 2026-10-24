"""Streamlit dashboard.

Five views: the two-factor grid as a heatmap, main effects and the
interaction, the accuracy-versus-cost scatters, the guardrail panel with every
bypass trace, and a live playground that sends a task through any cell of the
grid and shows the tool selection happening.

The playground is the part worth having. The grid tells you detailed tool
descriptions are worth N points; watching the same question get answered twice,
once with terse tools and once with detailed, is what makes N mean something.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("TOOLSMITH_API_URL", "http://localhost:8000")
RESULTS = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="Toolsmith", layout="wide", page_icon="*")


@st.cache_data(ttl=10)
def load_results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("baseline", "ablation", "adversarial", "determinism"):
        p = RESULTS / f"{name}.json"
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


@st.cache_data(ttl=30)
def api_get(path: str, **params) -> Any:
    r = httpx.get(f"{API}{path}", params=params, timeout=30.0)
    r.raise_for_status()
    return r.json()


def pct(x: float | None) -> str:
    return "--" if x is None else f"{x * 100:.1f}%"


res = load_results()
ablation, baseline, adversarial = res["ablation"], res["baseline"], res["adversarial"]

st.title("Toolsmith")
st.caption("How tool surface and prompt surface change an LLM's tool selection. "
           "Local inference, no hosted API.")

tabs = st.tabs(["Two-factor grid", "Guardrails", "Baseline", "Playground", "Tools"])

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
with tabs[0]:
    if not ablation:
        st.info("No ablation recorded yet. Run `make ablate`.")
    else:
        meta = ablation.get("meta", {})
        te = ablation["main_effect_tool_surface"]
        pe = ablation["main_effect_prompt_surface"]
        inter = ablation.get("interaction") or {}

        c = st.columns(4)
        c[0].metric("cells", len(ablation["cells"]), f"{meta.get('total_runs','?')} runs")
        c[1].metric("tool axis spread",
                    f"{(max(te.values()) - min(te.values())) * 100:.1f} pts")
        c[2].metric("prompt axis spread",
                    f"{(max(pe.values()) - min(pe.values())) * 100:.1f} pts")
        if inter.get("available"):
            c[3].metric("interaction", f"{inter['interaction'] * 100:+.1f} pts",
                        help="Tool-surface effect at a detailed prompt minus the same "
                             "effect at a terse one. Negative means the two surfaces are "
                             "substitutes: better tool docs buy less once the prompt is good.")

        cells = ablation["cells"]
        tools = sorted({x["tool_variant"] for x in cells})
        prompts = sorted({x["prompt_variant"] for x in cells})

        metric = st.selectbox(
            "Heatmap metric",
            ["selection_rate", "argument_rate", "completion_rate",
             "wrong_tool_rate", "unnecessary_rate", "latency_p50_s", "total_tokens_mean"],
        )
        is_rate = metric.endswith("_rate")
        z, text = [], []
        for p in prompts:
            row, trow = [], []
            for t in tools:
                cell = next((x for x in cells
                             if x["tool_variant"] == t and x["prompt_variant"] == p), None)
                v = None if cell is None else cell.get(metric)
                row.append(None if v is None else (v * 100 if is_rate else v))
                trow.append("" if v is None else
                            (f"{v * 100:.0f}%" if is_rate else f"{v:.1f}"))
            z.append(row)
            text.append(trow)

        fig = go.Figure(go.Heatmap(
            z=z, x=tools, y=prompts, text=text, texttemplate="%{text}",
            colorscale="Greens" if is_rate else "Oranges",
        ))
        fig.update_layout(height=330, xaxis_title="tool surface",
                          yaxis_title="prompt surface", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Main effects")
            f = go.Figure()
            f.add_bar(x=list(te), y=[v * 100 for v in te.values()], name="tool surface")
            f.add_bar(x=list(pe), y=[v * 100 for v in pe.values()], name="prompt surface")
            f.update_layout(height=320, barmode="group", yaxis_title="selection %",
                            margin=dict(t=20), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(f, use_container_width=True)
        with right:
            st.subheader("Interaction")
            if inter.get("available"):
                cc = inter["contrast"]
                f2 = go.Figure()
                for tool in (cc["tool_low"], cc["tool_high"]):
                    ys = [inter["cells"].get(f"{tool}|{cc['prompt_low']}"),
                          inter["cells"].get(f"{tool}|{cc['prompt_high']}")]
                    f2.add_scatter(x=[cc["prompt_low"], cc["prompt_high"]],
                                   y=[(y or 0) * 100 for y in ys],
                                   mode="lines+markers", name=f"{tool} tools",
                                   line=dict(width=3))
                f2.update_layout(height=320, yaxis_title="selection %",
                                 margin=dict(t=20), legend=dict(orientation="h", y=1.15))
                st.plotly_chart(f2, use_container_width=True)
            else:
                st.info(inter.get("reason", "not available"))

        st.subheader("Accuracy is not free")
        s1, s2 = st.columns(2)
        for col, xkey, xlabel in ((s1, "latency_p50_s", "latency p50 (s)"),
                                  (s2, "total_tokens_mean", "mean total tokens")):
            f3 = go.Figure()
            for t in tools:
                sel = [x for x in cells if x["tool_variant"] == t]
                f3.add_scatter(
                    x=[x[xkey] for x in sel],
                    y=[x["selection_rate"] * 100 for x in sel],
                    mode="markers+text", name=f"{t}",
                    text=[x["prompt_variant"].replace("_fewshot", "+fs") for x in sel],
                    textposition="top center", marker=dict(size=13),
                )
            f3.update_layout(height=340, xaxis_title=xlabel, yaxis_title="selection %",
                             margin=dict(t=20), legend=dict(orientation="h", y=1.15))
            col.plotly_chart(f3, use_container_width=True)

        st.subheader("Every cell")
        st.dataframe(pd.DataFrame(cells), use_container_width=True, hide_index=True)

        if ablation.get("incomplete_cells"):
            st.error(f"Incomplete cells excluded from conclusions: "
                     f"{', '.join(ablation['incomplete_cells'])}")

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
with tabs[1]:
    if not adversarial:
        st.info("No adversarial run recorded yet. Run `make adversarial`.")
    else:
        a = adversarial
        c = st.columns(5)
        c[0].metric("attacks", a["attacks"])
        c[1].metric("blocked", a["blocked"], pct(a["block_rate"]))
        c[2].metric("model refused", a["model_refused"],
                    help="No guardrail fired; the model declined. Not a guardrail win — "
                         "that defence changes when the model changes.")
        c[3].metric("bypassed", a["bypassed"], delta_color="inverse")
        fp = a["false_positives"]
        c[4].metric("false positives", fp.get("wrongly_blocked", "--"),
                    f"of {fp.get('legitimate_tasks','?')} legitimate tasks")

        st.subheader("What got through")
        if a["bypassed"]:
            st.caption("These are the result. A 100% block rate reads as an untested "
                       "guardrail; the catalogue of evasions is what has information in it.")
            for b in a["bypasses"]:
                with st.expander(f"{b['id']} · {b['category']} · "
                                 f"expected {b['expected_layer']}", expanded=True):
                    st.markdown("**Prompt**")
                    st.code(b["text"], language=None)
                    st.markdown("**Final answer**")
                    st.code((b["final_answer"] or "")[:1500], language=None)
                    if b["calls"]:
                        st.markdown("**Tool calls**")
                        st.json(b["calls"])
                    st.markdown("**Guardrail decisions**")
                    st.json(b["guardrail_log"])
        else:
            st.warning(
                "Nothing evaded the guardrails. The honest reading is that the suite was "
                "insufficiently adversarial, not that the defence is complete."
            )

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Where the load falls")
            st.dataframe(
                pd.DataFrame([{"layer": k, "stopped first": v}
                              for k, v in a["blocked_by_layer"].items()]),
                use_container_width=True, hide_index=True)
            st.dataframe(
                pd.DataFrame([{"rule": k, "fired": v}
                              for k, v in a["blocked_by_rule"].items()]),
                use_container_width=True, hide_index=True)
        with g2:
            st.subheader("By attack category")
            st.dataframe(
                pd.DataFrame([{"category": k, **v} for k, v in a["by_category"].items()]),
                use_container_width=True, hide_index=True)
            if a.get("layer_misattribution"):
                st.caption("Caught by a different layer than expected")
                st.dataframe(pd.DataFrame(a["layer_misattribution"]),
                             use_container_width=True, hide_index=True)

        if fp.get("blocked_tasks"):
            st.subheader("Legitimate tasks wrongly blocked")
            st.dataframe(pd.DataFrame(fp["blocked_tasks"]),
                         use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
with tabs[2]:
    if not baseline:
        st.info("No baseline recorded yet. Run `make baseline`.")
    else:
        verdict = baseline["verdict"]
        c = st.columns(4)
        c[0].metric("selection", pct(baseline["selection_rate"]),
                    f"chance {pct(baseline['chance_rate'])}")
        c[1].metric("arguments", pct(baseline.get("argument_rate")))
        c[2].metric("completion", pct(baseline.get("completion_rate")))
        c[3].metric("verdict", verdict.replace("_", " "))

        if verdict == "at_or_below_chance":
            st.error("At or below chance. The ablation would measure noise; a larger "
                     "model is required before any variant comparison means anything.")
        elif verdict == "ceiling":
            st.warning("At ceiling. The grid can run, but a suite this easy leaves little "
                       "room for a worse variant to lose.")
        else:
            st.success("The model can perform the task and is not at ceiling, so "
                       "differences between variants are interpretable.")

        st.dataframe(pd.DataFrame(baseline["per_task"]),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Send a task through any configuration")
    st.caption("The grid says a tool surface is worth N points. This is where you watch "
               "the same question get answered differently and see what N means.")

    try:
        health = api_get("/healthz")
        tool_variants = health["tool_variants"]
        prompt_variants = health["prompt_variants"]
        st.caption(f"model {health['model']} · ollama reachable "
                   f"{health['ollama_reachable']} · model present {health['model_present']}")
    except Exception as exc:
        st.error(f"API unreachable at {API}: {exc}")
        tool_variants = ["terse", "detailed", "example", "ambiguous"]
        prompt_variants = ["terse", "terse_fewshot", "detailed", "detailed_fewshot"]

    try:
        suite = api_get("/tasks")["tasks"]
    except Exception:
        suite = []

    col1, col2 = st.columns([3, 2])
    with col1:
        preset = st.selectbox(
            "Start from a suite task (optional, enables scoring)",
            ["(free text)"] + [f"{t['id']} · {t['text'][:70]}" for t in suite],
        )
        default = ""
        task_id = None
        if preset != "(free text)":
            task_id = preset.split(" · ")[0]
            default = next(t["text"] for t in suite if t["id"] == task_id)
        text = st.text_area("Question", value=default, height=110,
                            placeholder="What is the mean satisfaction score in the SaaS tracker?")
    with col2:
        tv = st.selectbox("Tool surface", tool_variants,
                          index=tool_variants.index("detailed") if "detailed" in tool_variants else 0)
        pv = st.selectbox("Prompt surface", prompt_variants,
                          index=prompt_variants.index("detailed_fewshot")
                          if "detailed_fewshot" in prompt_variants else 0)
        compare = st.checkbox("Also run with terse tools, side by side")

    if st.button("Run", type="primary", disabled=not text.strip()):
        configs = [(tv, pv)] + ([("terse", pv)] if compare and tv != "terse" else [])
        cols = st.columns(len(configs))
        for col, (t_variant, p_variant) in zip(cols, configs):
            with col:
                st.markdown(f"**{t_variant} tools · {p_variant} prompt**")
                with st.spinner("running (CPU inference, ~6-60s)..."):
                    try:
                        out = httpx.post(f"{API}/ask", json={
                            "text": text, "tool_variant": t_variant,
                            "prompt_variant": p_variant, "task_id": task_id,
                        }, timeout=420.0).json()
                    except Exception as exc:
                        st.error(f"request failed: {exc}")
                        continue

                if out.get("input_blocked"):
                    st.error("Blocked by the injection filter before reaching the model.")

                m = st.columns(3)
                m[0].metric("latency", f"{out['latency_s']:.1f}s")
                m[1].metric("tokens", out["tokens"]["total"])
                m[2].metric("calls", len(out["tool_calls"]))

                st.markdown("**Tool selection**")
                if out["tool_calls"]:
                    for call in out["tool_calls"]:
                        tag = "BLOCKED" if call["blocked"] else "ok"
                        st.code(f"[{tag}] {call['requested']}"
                                + (f"  -> {call['canonical']}"
                                   if call["canonical"] != call["requested"] else "")
                                + f"\n{json.dumps(call['arguments'], indent=2)}",
                                language="json")
                        if call["blocked"]:
                            st.caption(f"{call['layer']} / {call['rule']}")
                else:
                    st.caption("no tool called")

                st.markdown("**Answer**")
                st.write(out["answer"] or "_(empty)_")

                if out.get("score"):
                    s = out["score"]
                    st.markdown("**Scored against ground truth**")
                    st.write({
                        "correct_selection": s["correct_selection"],
                        "argument_correct": s["argument_correct"],
                        "task_complete": s["task_complete"],
                        "detail": s["arg_detail"],
                    })

                with st.expander("Guardrail decisions"):
                    st.json(out["guardrail_decisions"])

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("The tool surface, as a model sees it")
    st.caption("Several of these overlap on purpose. If the tools were trivially "
               "distinct, description quality would have nothing to disambiguate and "
               "the ablation would have nothing to find.")
    try:
        variants = api_get("/healthz")["tool_variants"]
    except Exception:
        variants = ["terse", "detailed", "example", "ambiguous"]
    v = st.selectbox("Variant", variants, key="tools_variant")
    try:
        payload = api_get("/tools", variant=v)
        st.caption(f"{payload['summary']['payload_chars']:,} characters of tool schema "
                   f"sent with every request in this variant")
        for t in payload["tools"]:
            flag = "" if t["allowlisted"] else "  [NEVER ALLOWLISTED]"
            with st.expander(f"{t['name']}{flag}"
                             + (f"  (canonically {t['canonical']})"
                                if t["name"] != t["canonical"] else "")):
                st.write(t["description"])
                st.json(t["parameters"])
    except Exception as exc:
        st.error(f"could not load tools: {exc}")
