"""Scoring logic against hand-computed cases.

Every expectation here is worked out by hand from the rules in
docs/methodology.md. If a rule changes, these fail — which is the point: the
headline accuracy numbers are only as meaningful as these definitions, and a
silent change to "correct selection" would silently change every result in
the README.
"""

from __future__ import annotations

import pytest

from toolsmith.harness.scoring import (
    CallRecord,
    percentile,
    score_arguments,
    score_task,
)
from toolsmith.harness.suite import Task


def call(name: str, args: dict | None = None, *, blocked: bool = False) -> CallRecord:
    return CallRecord(
        requested_name=name, canonical_name=name, arguments=args or {},
        executed=not blocked, blocked=blocked, result={"value": 3.62, "n": 100},
    )


def needs_tool_task(**kw) -> Task:
    base = dict(
        id="t", category="near_neighbor", text="x",
        expected_tool="aggregate_metric",
        expected_args={"survey_id": "srv_saas_nps", "metric": "mean"},
    )
    base.update(kw)
    return Task(**base)


def no_tool_task() -> Task:
    return Task(id="n", category="no_tool", text="x", expected_tool=None)


class TestSelection:
    def test_right_tool_is_correct(self):
        s = score_task(needs_tool_task(), [call("aggregate_metric",
                       {"survey_id": "srv_saas_nps", "metric": "mean"})], "3.62")
        assert s.correct_selection
        assert s.wrong_tool_calls == 0

    def test_wrong_tool_is_incorrect_and_counted(self):
        s = score_task(needs_tool_task(), [call("compute_crosstab")], "x")
        assert not s.correct_selection
        assert s.wrong_tool_calls == 1

    def test_no_call_on_a_task_needing_one_is_incorrect(self):
        s = score_task(needs_tool_task(), [], "I think it is about 3.6")
        assert not s.correct_selection
        assert s.wrong_tool_calls == 0  # nothing wrong was called; nothing was

    def test_no_tool_task_correct_when_nothing_called(self):
        s = score_task(no_tool_task(), [], "A mean averages all values.")
        assert s.correct_selection
        assert s.unnecessary_calls == 0

    def test_no_tool_task_incorrect_when_a_tool_is_called(self):
        s = score_task(no_tool_task(), [call("list_surveys")], "...")
        assert not s.correct_selection
        assert s.unnecessary_calls == 1

    def test_blocked_call_still_counts_as_a_selection(self):
        # Asking for the forbidden tool is a selection error whether or not
        # the allowlist stopped it reaching data.
        t = needs_tool_task()
        s = score_task(t, [call("export_respondent_pii", blocked=True)], "")
        assert not s.correct_selection
        assert s.forbidden_attempted
        assert s.wrong_tool_calls == 1

    def test_also_acceptable_tool_counts_as_correct(self):
        t = needs_tool_task(also_acceptable=("compare_segments",))
        s = score_task(t, [call("compare_segments")], "x")
        assert s.correct_selection

    def test_prerequisite_is_not_a_wrong_tool(self):
        t = needs_tool_task(prerequisite_ok=("list_surveys",))
        s = score_task(t, [call("list_surveys"),
                           call("aggregate_metric",
                                {"survey_id": "srv_saas_nps", "metric": "mean"})], "3.62")
        assert s.correct_selection
        assert s.wrong_tool_calls == 0
        assert s.unnecessary_calls == 0


class TestUnnecessaryCalls:
    def test_repeated_acceptable_call_is_unnecessary(self):
        t = needs_tool_task()
        args = {"survey_id": "srv_saas_nps", "metric": "mean"}
        s = score_task(t, [call("aggregate_metric", args),
                           call("aggregate_metric", args),
                           call("aggregate_metric", args)], "3.62")
        assert s.correct_selection
        # Three acceptable calls where one would do: two are redundant.
        assert s.unnecessary_calls == 2

    def test_every_call_on_a_no_tool_task_is_unnecessary(self):
        s = score_task(no_tool_task(), [call("list_surveys"), call("aggregate_metric")], "")
        assert s.unnecessary_calls == 2


class TestArgumentCorrectness:
    def test_scored_only_when_selection_was_right(self):
        s = score_task(needs_tool_task(), [call("compute_crosstab")], "x")
        assert s.argument_correct is None  # not 0, not False: undefined

    def test_pinned_arguments_must_match(self):
        t = needs_tool_task()
        s = score_task(t, [call("aggregate_metric",
                                {"survey_id": "srv_retail_cx", "metric": "mean"})], "3.62")
        assert s.correct_selection
        assert s.argument_correct is False

    def test_unpinned_extra_arguments_are_fine(self):
        t = needs_tool_task()
        s = score_task(t, [call("aggregate_metric", {
            "survey_id": "srv_saas_nps", "metric": "mean", "waves": ["w1"]
        })], "3.62")
        assert s.argument_correct is True

    def test_argument_comparison_is_case_insensitive(self):
        ok, _ = score_arguments(needs_tool_task(),
                                {"survey_id": "SRV_SAAS_NPS", "metric": "Mean"})
        assert ok

    def test_wave_lists_compare_as_sets(self):
        t = needs_tool_task(expected_args={"waves": ["w1", "w2"]})
        ok, _ = score_arguments(t, {"waves": ["w2", "w1"]})
        assert ok

    def test_missing_pinned_argument_fails(self):
        ok, detail = score_arguments(needs_tool_task(), {"survey_id": "srv_saas_nps"})
        assert not ok and "metric" in detail

    def test_the_probe_failure_mode_is_caught(self):
        # The real observed failure: right tool, survey *name* instead of id,
        # and an invented segment field. Exactly what this metric is for.
        t = needs_tool_task()
        s = score_task(t, [call("aggregate_metric", {
            "survey_id": "SaaS Platform Satisfaction Tracker",
            "metric": "mean",
            "segment": {"customer_type": "enterprise"},
        })], "3.62")
        assert s.correct_selection is True
        assert s.argument_correct is False


class TestCompletion:
    def test_matching_number_completes(self):
        s = score_task(needs_tool_task(),
                       [call("aggregate_metric", {"survey_id": "srv_saas_nps",
                                                  "metric": "mean"})],
                       "The mean is 3.62.")
        assert s.task_complete is True

    def test_tolerates_rounding(self):
        s = score_task(needs_tool_task(),
                       [call("aggregate_metric", {"survey_id": "srv_saas_nps",
                                                  "metric": "mean"})],
                       "About 3.6.")
        assert s.task_complete is True

    def test_wrong_number_does_not_complete(self):
        s = score_task(needs_tool_task(),
                       [call("aggregate_metric", {"survey_id": "srv_saas_nps",
                                                  "metric": "mean"})],
                       "The mean is 9.91.")
        assert s.task_complete is False

    def test_counts_are_not_treated_as_the_answer(self):
        # n=100 is in the payload; quoting it is not answering the question.
        s = score_task(needs_tool_task(),
                       [call("aggregate_metric", {"survey_id": "srv_saas_nps",
                                                  "metric": "mean"})],
                       "There were 100 responses.")
        assert s.task_complete is False


class TestPercentile:
    def test_nearest_rank(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert percentile(data, 50) == 5
        assert percentile(data, 99) == 10
        assert percentile(data, 100) == 10

    def test_empty_is_zero(self):
        assert percentile([], 50) == 0.0

    def test_returns_an_observed_value(self):
        # With a dozen observations per cell an interpolated p99 is a number
        # no run produced.
        data = [1.0, 2.0, 50.0]
        assert percentile(data, 99) in data
