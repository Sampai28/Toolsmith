# Toolsmith

An MCP server over survey-analytics data, plus a harness that measures how
tool descriptions and system prompts change an LLM's tool selection — with
guardrails that get attacked rather than asserted.

Everything runs locally. Inference is Ollama in a container; there is no
hosted API anywhere in this project and nowhere to put a key.

## What it found

Before running any ablation, the harness runs the task suite once in the best
configuration available — detailed tool descriptions, detailed system prompt,
few-shot examples — to establish the model can do the task at all.

| model | correct tool selected | arguments correct | task completed |
|---|---:|---:|---:|
| `qwen2.5:7b-instruct` | **100.0%** | **55.6%** | 63.6% |
| `llama3.2:3b` | 66.7% | 37.5% | 25.0% |

Chance is 13.8% for this task mix, computed rather than assumed: one in seven
for a task needing one of the seven callable tools, one in eight for a task
answerable with no call at all.

The 7B model picks the right tool every single time, across all four task
categories including the ones built to be confusable. Then it gets the
arguments wrong on nearly half of those same calls. It passes
`survey_id: "SaaS Platform Satisfaction Tracker"` — the display name instead
of the identifier — and invents segment fields like `customer_type` that do
not exist in the schema.

A selection-accuracy benchmark scores that a perfect 100%. The call returns
nothing.

That is the reason argument correctness is scored on its own, and only on
runs that already selected the right tool. Averaged together the two collapse
into a single number that describes neither.

The other thing the baseline says is about the suite rather than the model.
100% is a ceiling, and a ceiling leaves the tool-description axis nothing to
lose. A harder suite would need tasks where the right tool is genuinely
contested — composition across three tools, phrasing that points at the wrong
one, and segment vocabulary that cannot be guessed from the question.

## The grid

Two axes, every combination, scored on selection, arguments, wrong-tool
calls, unnecessary calls, completion, latency and tokens.

Axis one is how the tools describe themselves: `terse` one-liners with
genericised parameter names, `detailed` descriptions saying when to use each
tool and when not to, `example` adding a worked call, and `ambiguous` —
detailed prose under deliberately colliding names like `get_data` and
`get_results`. That last one is the control that makes the axis readable: a
gap between terse and detailed could be better names or better prose, and
holding the prose fixed while wrecking the names separates them.

Axis two is the system prompt: terse or detailed, crossed with few-shot
examples present or absent. "Spell out the rules" and "show worked examples"
are different claims, so they are reported separately as well as combined.

The analysis gives a main effect per axis plus the interaction, which is the
part worth having: if better tool descriptions stop buying anything once the
system prompt is already detailed, the two are substitutes and you should
invest in one rather than both.

```bash
make ablate
```

## Guardrails

Four layers. An injection filter screens inbound text before the model sees
it. An allowlist, argument validation and an egress check run on the
invocation path, inside the MCP server rather than the harness, so any MCP
client is covered.

Every decision is logged with the rule that fired — allows as well as blocks.
"Which layer let this through" is the question a bypass analysis needs, and
it cannot be answered from block records alone.

The adversarial suite is 28 prompts across eight families: direct instruction
override, role reassignment, hypothetical framing, encoded directives,
instructions buried in what looks like survey data, argument abuse,
multi-turn setup, and chaining a permitted tool toward prohibited output.

That last family is the interesting one. It names no forbidden tool and uses
no override language, so the injection filter has almost nothing to match and
the allowlist never fires — the egress check is the only thing left. A
defence that only works when the attacker says "ignore your instructions" is
not a defence.

An attack counts as blocked only when a rule fired. When no rule fires and
the model simply declines, that is recorded separately, because the defence
in that case was the model's disposition and that changes with the model.

```bash
make adversarial
```

## Running it

```bash
make up && make all
```

`up` starts Ollama, pulls the model, and brings up the API and the dashboard.
`all` generates the corpus, runs the validation gates, then the baseline, the
grid, the adversarial suite and the report.

Dashboard on <http://localhost:8501>, API on <http://localhost:8000>. The
dashboard has a playground that sends the same question through any cell of
the grid, side by side, which is the quickest way to see what a tool
description is actually worth.

Stages run individually too: `make generate`, `baseline`, `ablate`,
`adversarial`, `report`, `test`. Swap the model with `MODEL=llama3.2:3b`.

## Why the tools overlap

Seven callable tools over a synthetic corpus of surveys, waves, respondents,
segments, questions and responses — plus `export_respondent_pii`, which is
real, listed in the schema, and never on the allowlist.

`aggregate_metric`, `compare_segments`, `compute_crosstab` and
`trend_over_waves` all compute the same metrics over the same data and differ
only in how the result is sliced. `query_responses` can answer any of them
badly, by returning rows you would have to summarise by hand.

The overlap is deliberate, and it is the whole experiment. "How does
satisfaction differ between enterprise and SMB" is answered properly by
`compare_segments`, clumsily by two `aggregate_metric` calls, and wrongly by
`query_responses` plus arithmetic. If the tools were trivially distinct,
description quality would have nothing to disambiguate — and a null result
would be caused by the tool set rather than by the descriptions.

The forbidden tool is implemented rather than stubbed. A guardrail tested
against a tool that does nothing proves the guardrail runs, not that it
protects anything.

## Decisions worth knowing about

**The model.** `qwen2.5:7b-instruct` on CPU. `llama3.2:3b` was measured as
the faster-but-weaker option and turned out to be neither faster nor
adequate — 66.7% selection against 100%, and slower end to end, 868 seconds
against 675 for the same twelve tasks, because it retries and calls tools on
tasks that need none.

**Prompt caching decides the shape of the run.** A cached prefix replays at
about 1,100 tokens a second here against 20 cold. Each grid cell is one
distinct prefix, so cells run sequentially with their tasks consecutive —
adding a task to a cell is nearly free, adding a cell is not. Every request
sends identical inference options, because changing one mid-run throws the
cache away.

**The context length is set explicitly, and this was the expensive bug.**
Ollama 0.5.x defaults `num_ctx` to 2048 and ignores `OLLAMA_CONTEXT_LENGTH`.
The detailed tool surface is 2,126 tokens, so the model was being shown a
silently truncated tool list and the cache was dying on every second turn —
78 to 156 seconds per call instead of about eight. Nothing errored. It just
got slow and quietly wrong.

**Schema validation was taken off the MCP SDK.** The SDK validates against
`inputSchema` before dispatch and raises before any guardrail runs, so an
out-of-range argument came back as a bare protocol error with no layer or
rule attached. The server passes `validate_input=False` and the named
guardrail does the work. Not weaker validation, just validation somewhere it
can be counted.

## Limitations

One model family, synthetic data, and tasks written by the same person who
designed the tools — real analyst questions are messier and would pull every
number down.

Selection is at ceiling for the 7B model, so the tool-description axis has
little room to move on that particular metric; the suite needs to be harder
before it can say much there.

The quick tier is 48 runs — twelve cells, four tasks, one repeat — so one
task in a cell is worth 25 percentage points and only large gaps mean
anything. `configs/full.yaml` is the version that buys confidence intervals,
at the cost of several hours of CPU inference.

Most tasks resolve in a single tool call, so nothing here measures
long-horizon tool use or recovery from a failed call. And the adversarial
suite is a floor rather than a bound: a category with no bypass means those
particular prompts failed, not that the category is covered.

## License

MIT. Every dependency is free and open source. Nothing here needs an account,
an API key, or a paid service.
