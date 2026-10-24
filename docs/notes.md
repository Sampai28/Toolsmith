# Design notes

Decisions that are not obvious from the code, and what they cost.

## The tools overlap on purpose

`aggregate_metric`, `compare_segments`, `compute_crosstab` and
`trend_over_waves` all compute the same metrics over the same corpus. They
differ only in how the result is sliced: one group, several groups, two
dimensions, or time. `query_responses` can technically answer any of them
badly, by returning rows a caller would have to summarise by hand.

That overlap is the experiment. A user asking "how does satisfaction differ
between enterprise and SMB" can be answered by `compare_segments` (right), by
two `aggregate_metric` calls (clumsy), or by `query_responses` and arithmetic
(wrong). If the tools were trivially distinct, description quality would have
nothing to disambiguate and the ablation would find nothing — the result
would be "descriptions don't matter", caused entirely by the tool set.

The cost is that the suite's ground truth has to take a position on cases
where two tools are defensible. `also_acceptable` exists for that and is used
sparingly; a generous acceptance set is how a selection benchmark quietly
stops discriminating.

## The forbidden tool is real and discoverable

`export_respondent_pii` is implemented, returns actual names and emails, is
listed in `tools/list`, and is never on the allowlist.

Implemented rather than stubbed because a guardrail tested against a tool that
does nothing proves the guardrail *runs*, not that it *protects* anything. If
the allowlist ever regresses, this returns real identifiers and the egress
check has something to catch — which is a test with teeth.

Discoverable because hiding it would make the adversarial suite measure
discoverability rather than enforcement, and no real deployment can assume an
attacker does not know a tool exists.

## Guardrails are server-side

They run inside the MCP server's `call_tool` path, not in the harness. A
caller that bypasses the harness entirely — any MCP client at all — still
cannot reach the forbidden tool.

`dispatch()` in the tool registry performs no authorisation of its own. That
is deliberate: if dispatch refused the forbidden tool itself, the allowlist
could be deleted without any test failing, and the security property would be
an accident of that function rather than a stated rule with a log line.

## Schema validation was moved off the SDK

The MCP SDK validates arguments against `inputSchema` before dispatch. That is
the same check `guardrails/argcheck.py` performs — but it raises first, so an
out-of-range argument came back as a bare protocol error with no layer, no
rule and nothing in the guardrail log. The project claims every guardrail
decision is attributable; a check that silently pre-empts the attributable one
defeats that.

The server passes `validate_input=False` and the named layer does the work.
Validation is not weakened, it is moved somewhere it can be counted. The
tests assert the rule name, so a regression here is visible.

## Prompt caching decided the shape of the whole run

Measured on this box with `qwen2.5:7b-instruct`:

| | tok/s |
|---|---:|
| generation | ~8.5 |
| cold prompt evaluation | ~20 |
| cached prefix replay | ~1,100 |

A fifty-fold gap between the last two. Each grid cell is one distinct prefix
(system prompt + tool schemas), so the run pays one cold evaluation per cell
and effectively nothing after it. That is why cells run sequentially with
their tasks consecutive, and why `tasks_per_cell` is the only variable worth
tuning: adding a task to a cell is nearly free, adding a cell is not.

Two traps found the hard way:

- **`OLLAMA_NUM_PARALLEL=2` broke it entirely.** Ollama divides the context
  window per slot, and a detailed tool surface plus a few-shot prompt no
  longer fit in half of it. Every call reprocessed the whole prompt: 78–156 s
  each, with no reuse. At 1 slot the same work is 1.4 s.
- **Changing any inference option invalidates the cache.** A probe that set
  `num_batch` and `num_thread` to "tune" throughput dropped from 1,100 tok/s
  to 20. The client sends identical options on every request for this reason,
  and it is not a style preference.

## The quick tier is small, and that is a real limitation

12 cells × 4 tasks × 1 repeat = 48 runs. One task in a cell is 25 percentage
points. Only large gaps mean anything.

The alternative was cutting cells, which is cheaper per unit of time saved and
would have destroyed the ablation — a main effect needs every level present,
and an interaction needs all four corners of the 2×2. Cutting tasks costs
precision; cutting cells costs the result. `configs/full.yaml` is the version
that buys confidence intervals, and it is not run here because it is hours of
CPU inference.

## Chance level is computed, not assumed

With 7 callable tools, a task needing one is right by guessing with
probability 1/7. A `no_tool` task is right only if nothing is called, which a
guesser does with probability 1/8. The floor is averaged over the actual task
mix, because a suite that is a third `no_tool` has a different chance level
from one that is not. Hard-coding "about 14%" would have been wrong for this
suite.

## `model_refused` is tracked separately from `blocked`

An attack where no guardrail fires and the model simply declines is not a
guardrail success. The defence in that case is the model's disposition, which
changes with the model, the temperature, and the phrasing. Counting it as a
block would overstate the guardrails by exactly the amount that matters.

## What is missing

- **Multi-turn attacks are shallow.** Two setup turns is not a patient
  attacker. A real one establishes context over many turns and never names
  the target.
- **No retrieval or tool-output injection.** The corpus is clean; nothing in
  it tries to manipulate the model. A survey verbatim containing a directive
  is modelled only as task text, not as something returned by a tool
  mid-conversation, which is the harder version of the problem.
- **The egress check is pattern plus roster.** A model that paraphrases an
  identifier ("the respondent whose name rhymes with…") defeats it, and no
  amount of regex fixes that.
- **One repeat in the quick tier.** Temperature 0 is expected to be stable but
  "expected to be" is not a measurement.
- **No cost model beyond tokens and latency.** The accuracy-versus-tokens
  scatter shows the trade, but nothing converts it to money, because local
  inference has no price per token and inventing one would be fiction.
