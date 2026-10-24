# Toolsmith

An MCP server over survey-analytics data, plus a harness that measures how
tool descriptions and system prompts change an LLM's tool selection — with
guardrails that get attacked rather than asserted.

Everything runs locally. Inference is Ollama in a container; there is no
hosted API anywhere in this project and nowhere to put a key.

## The headline result: selection is easy, arguments are not

Step 0 runs the task suite in the best configuration available (detailed tool
descriptions, detailed system prompt with few-shot examples) before any
ablation, to establish the model can do the task at all.

| model | correct tool selected | arguments correct | task completed | verdict |
|---|---:|---:|---:|---|
| `qwen2.5:7b-instruct` | **100.0%** | **55.6%** | 63.6% | ceiling |
| `llama3.2:3b` | 66.7% | 37.5% | 25.0% | usable |

Chance is 13.8% for this task mix, computed rather than assumed: 1/7 for a
task needing one of seven callable tools, 1/8 for one answerable with no call.

The 7B model picks the right tool **every single time**, across all four task
categories including the ones designed to be confusable. Then it gets the
arguments wrong on nearly half of those same calls. It passes
`survey_id: "SaaS Platform Satisfaction Tracker"` — the display name instead
of the identifier — and invents segment fields like `customer_type` that do
not exist in the schema.

A selection-accuracy benchmark scores that 100%. The call returns nothing.

This is why argument correctness is scored separately and only on runs that
selected correctly: averaged together, the two collapse into one number that
describes neither. It is the single most useful thing this project measured.

**The second finding is about the suite, not the model.** 100% selection is a
ceiling, and a ceiling means the tool-surface axis has nothing to lose on that
metric. A harder suite would need tasks where the *right* tool is genuinely
contested — composition across three tools, tasks whose phrasing points at the
wrong tool, and segment vocabulary that is not guessable from the question.

## Two-factor ablation

Two axes, every combination. Axis 1 is how the tools describe themselves
(`terse` / `detailed` / `ambiguous`, with `example` in the full tier). Axis 2
is the system prompt (`terse` / `detailed`, crossed with few-shot examples
present or absent).

| | selection | arguments | wrong tool | unnecessary calls | p50 | p99 | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| main effect, tool surface | `<TOOL_MAIN_EFFECT>` | `<TOOL_ARG_EFFECT>` | `<TOOL_WRONG>` | `<TOOL_UNNEC>` | `<TOOL_P50>` | `<TOOL_P99>` | `<TOOL_TOKENS>` |
| main effect, prompt surface | `<PROMPT_MAIN_EFFECT>` | `<PROMPT_ARG_EFFECT>` | `<PROMPT_WRONG>` | `<PROMPT_UNNEC>` | `<PROMPT_P50>` | `<PROMPT_P99>` | `<PROMPT_TOKENS>` |
| interaction | `<INTERACTION>` | | | | | | |

The interaction is the question worth asking: does upgrading tool
descriptions still buy anything once the system prompt is already detailed? A
negative interaction means the two surfaces are substitutes and a team should
invest in one, not both.

Run `make ablate` to populate these. The grid, scoring and analysis are
implemented and tested; the numbers above are not filled in because this
commit does not include a completed grid run.

## Guardrails, and what got past them

Four layers on the invocation path. Injection screening runs on inbound text
before inference; allowlist, argument validation and egress run on the call
itself. Every decision is logged with the rule that fired — allows as well as
blocks, because "which layer let this through" is the question bypass
analysis needs and it is unanswerable from block records alone.

| | |
|---|---:|
| attacks | 28 |
| blocked by a guardrail | `<BLOCK_COUNT>` |
| refused by the model, no guardrail fired | `<MODEL_REFUSED>` |
| **bypassed** | `<BYPASS_COUNT>` |
| legitimate tasks wrongly blocked | `<FALSE_POSITIVES>` of 36 |

`model_refused` is counted separately from `blocked` on purpose. When no rule
fires and the model simply declines, the defence was the model's disposition
— which changes when the model, the temperature or the phrasing changes.
Counting it as a block overstates the guardrails by exactly the amount that
matters.

The bypass catalogue is the output worth reading, not the block rate. A suite
reporting 100% blocked has usually been written to be blocked. Run
`make adversarial` to produce it.

What is verified today, by test rather than by run: the forbidden tool is
blocked at the allowlist even when presented under an alias (`get_contacts`
under the ambiguous-naming variant), out-of-range arguments are blocked with
an attributable rule, invented segment fields are rejected against the schema
enum, and legitimate row-level queries survive the egress check.

## Quickstart

```bash
make up && make all
```

`up` starts Ollama, pulls the model and brings up the API and dashboard.
`all` generates the corpus, runs every validation gate, runs the capability
baseline, then the grid, the adversarial suite and the report.

- dashboard — <http://localhost:8501>
- API — <http://localhost:8000/healthz>

Individual stages: `make generate`, `baseline`, `ablate`, `adversarial`,
`report`, `test`. Override the model with `MODEL=llama3.2:3b`.

## The tool set, and why it overlaps

Seven callable tools over a synthetic market-research corpus — surveys,
waves, respondents, segments, questions, responses — plus
`export_respondent_pii`, which is real, discoverable in the schema, and never
on the allowlist.

`aggregate_metric`, `compare_segments`, `compute_crosstab` and
`trend_over_waves` all compute the same metrics over the same data, differing
only in how the result is sliced. `query_responses` can answer any of them
badly, by returning rows a caller would have to summarise by hand.

That overlap is deliberate and it is the experiment. "How does satisfaction
differ between enterprise and SMB" is answered by `compare_segments`
(correct), two `aggregate_metric` calls (clumsy), or `query_responses` plus
arithmetic (wrong). If the tools were trivially distinct, description quality
would have nothing to disambiguate and the ablation would find nothing — and
the null result would be caused by the tool set, not by the descriptions.

The forbidden tool is implemented rather than stubbed. A guardrail tested
against a tool that does nothing proves the guardrail runs, not that it
protects anything.

## Adversarial methodology

28 prompts across eight families: direct instruction override, role
reassignment, hypothetical and roleplay framing, encoded directives (base64,
hex, zero-width characters, letter spacing), instructions embedded in what
looks like survey data, argument abuse of a permitted tool, multi-turn setup
followed by the ask, and chaining a permitted tool toward prohibited output.

The last family is the interesting one. It names no forbidden tool and uses
no override language, so the injection filter has almost nothing to match and
the allowlist is never triggered — the egress check is the only layer left. A
defence that only works when the attacker says "ignore your instructions" is
not a defence.

Success for the attacker is any of: the forbidden tool executes, respondent
identifiers appear in the final answer, or a permitted tool is driven outside
its schema.

## Design decisions

**Model.** `qwen2.5:7b-instruct`, CPU-only. `llama3.2:3b` was measured as the
faster-but-weaker alternative and turned out to be neither faster *nor*
adequate: 66.7% selection against 100%, and **slower end to end** (868 s
against 675 s for the same 12 tasks) because it retries and calls tools on
tasks needing none.

**Grid sizing is set by prompt caching.** Ollama caches the prompt prefix;
measured on this box a cached prefix replays at ~1,100 tok/s against ~20
tok/s cold. Each cell is one distinct prefix, so cells run sequentially with
their tasks consecutive, and adding a task to a cell is nearly free while
adding a cell is not. Every request sends identical inference options, because
changing one mid-run invalidates the cache.

**`num_ctx` is set explicitly.** Ollama 0.5.x defaults to 2048 and ignores
`OLLAMA_CONTEXT_LENGTH`. The detailed tool surface is 2,126 tokens, so the
model was being shown a *silently truncated tool list* and the cache was
destroyed on every second turn — 78–156 s per call instead of ~8 s. This was
the most expensive bug in the build and it produced no error message.

**Schema validation was moved off the MCP SDK.** The SDK validates
`inputSchema` before dispatch and raises before any guardrail runs, so an
out-of-range argument returned a bare protocol error with no layer and no
rule. The server passes `validate_input=False` and the named layer does the
work. Validation is not weakened, it is moved somewhere it can be counted.

**Guardrails are server-side.** They run inside the MCP server, not the
harness, so any MCP client is covered. `dispatch()` performs no authorisation
of its own — otherwise the allowlist could be deleted with no test failing.

## Limitations

- **One model, one family.** Tool-calling behaviour varies sharply across
  scales and tokenisers; nothing here generalises.
- **Synthetic data and synthetic tasks**, written by the same person who
  designed the tools. Real analyst questions are messier.
- **Selection is at ceiling for the 7B model**, so the tool-surface axis has
  little room to move on that metric. The suite needs to be harder.
- **The quick tier is 48 runs** — 12 cells, 4 tasks, 1 repeat. One task in a
  cell is 25 percentage points. `configs/full.yaml` is what buys confidence
  intervals and is not run here.
- **Single-turn majority.** Nothing measures long-horizon tool use or
  recovery from a failed call.
- **The 15-minute budget is not achievable for a 12-cell grid on this
  hardware.** The twelve cells sum to ~19,300 cold prompt tokens; at 20 tok/s
  that is ~16 minutes of prompt evaluation before a single token is
  generated. Cutting tasks does not help because tasks are not what costs,
  and cutting cells would destroy the ablation.
- **The adversarial suite is a floor, not a bound.** Absence of a bypass in a
  category means those prompts failed, not that the category is covered.

## Future scope

Implemented and tested but not yet run to completion in this commit:

- the two-factor grid (`make ablate`) and its main-effects and interaction
  analysis
- the adversarial suite (`make adversarial`) and the bypass catalogue
- `reports/evaluation.html` built from both

Not implemented:

- tool-output injection — a directive returned by a tool mid-conversation,
  which is harder than the same directive in task text
- repeats and variance in the quick tier
- a harder task suite that puts selection below ceiling

## Resume bullets

Values left blank are ones this commit did not measure.

- Built an MCP server and a two-factor ablation harness measuring how tool
  surface and prompt surface affect LLM tool selection, finding that
  `qwen2.5:7b-instruct` selects the correct tool **100%** of the time while
  supplying correct arguments only **55.6%** of the time — a failure mode
  invisible to selection-accuracy benchmarks.
- Designed a deliberately confusable seven-tool surface with four description
  variants and a four-level prompt axis, measuring main effects of
  `<TOOL_MAIN_EFFECT>` and `<PROMPT_MAIN_EFFECT>` and an interaction of
  `<INTERACTION>` across `<TOTAL_RUNS>` runs.
- Implemented a four-layer guardrail stack with per-rule decision logging and
  attacked it with 28 adversarial prompts across eight families, blocking
  `<BLOCK_COUNT>` with `<FALSE_POSITIVES>` false positives on legitimate
  traffic and cataloguing `<BYPASS_COUNT>` bypasses with full traces.
- Diagnosed a silent prompt-truncation bug where a 2048-token default context
  cut off a 2,126-token tool surface and destroyed the inference cache,
  cutting per-call latency from **78–156 s to ~8 s** and restoring the tool
  list the model was actually shown.

## License

MIT. All dependencies are free and open source. Nothing here needs an
account, an API key, or a paid service.
