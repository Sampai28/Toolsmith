<!--
Shared few-shot block, appended to either base prompt to form the
`*_fewshot` variants.

Four examples, chosen to cover the decisions the task suite actually tests:
one aggregate, one comparison that could be mistaken for an aggregate, one
trend that could be mistaken for two aggregates, and one question that needs
no tool at all. The no-tool example is the one that matters most — without it
a model calls something on every turn, and unnecessary invocation stops being
measurable.

Deliberately uses surveys and questions that exist in the corpus, so the
examples do not teach identifiers that would then be wrong.
-->

Examples of correct behaviour:

User: What is the average satisfaction score in the SaaS tracker?
Assistant: calls aggregate_metric(survey_id="srv_saas_nps", question="satisfaction", metric="mean")

User: How does support quality differ across regions in the SaaS tracker?
Assistant: calls compare_segments(survey_id="srv_saas_nps", question="support_quality", metric="mean", segment_field="region")
(Not aggregate_metric: the question contrasts several groups, so one call returns all of them.)

User: Has satisfaction improved over the course of the SaaS tracker?
Assistant: calls trend_over_waves(survey_id="srv_saas_nps", question="satisfaction", metric="mean")
(Not two aggregate_metric calls: the question is about change over time, and this returns the whole series in wave order.)

User: What is the difference between a mean and a top box share?
Assistant: answers directly in plain language and calls no tool.
(No survey data is required to answer this.)
