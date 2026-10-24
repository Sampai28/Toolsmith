"""Task-suite validation rules, dataset gates, and determinism."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from toolsmith.config import DatasetConfig, load_config
from toolsmith.data.generate import generate
from toolsmith.data.validate import validate as validate_dataset
from toolsmith.harness.suite import (
    RULE_DUPLICATE_ID,
    RULE_FORBIDDEN_GROUND_TRUTH,
    RULE_MALFORMED_ARGS,
    RULE_NO_ANSWER,
    RULE_UNKNOWN_TOOL,
    Task,
    load_tasks,
    select,
    validate_suite,
)
from toolsmith.tools.registry import FORBIDDEN_TOOL


class TestShippedSuite:
    def test_the_real_suite_passes_every_gate(self):
        rep = validate_suite(load_tasks())
        assert rep.ok, rep.to_json()

    def test_suite_is_large_enough(self):
        assert len(load_tasks()) >= 30

    def test_every_category_is_represented(self):
        cats = {t.category for t in load_tasks()}
        assert cats == {"single_tool", "near_neighbor", "multi_step", "no_tool"}

    def test_no_tool_tasks_exist_so_unnecessary_calls_are_measurable(self):
        assert sum(1 for t in load_tasks() if not t.needs_tool) >= 3


class TestSuiteValidationRules:
    def test_unknown_ground_truth_tool_is_caught(self):
        rep = validate_suite([Task(id="x", category="single_tool", text="q",
                                   expected_tool="no_such_tool")])
        assert RULE_UNKNOWN_TOOL in rep.counts

    def test_forbidden_tool_can_never_be_ground_truth(self):
        rep = validate_suite([Task(id="x", category="single_tool", text="q",
                                   expected_tool=FORBIDDEN_TOOL)])
        assert RULE_FORBIDDEN_GROUND_TRUTH in rep.counts

    def test_duplicate_ids_caught(self):
        t = Task(id="dup", category="no_tool", text="q", expected_tool=None)
        rep = validate_suite([t, t])
        assert RULE_DUPLICATE_ID in rep.counts

    def test_malformed_expected_args_caught(self):
        rep = validate_suite([Task(id="x", category="single_tool", text="q",
                                   expected_tool="aggregate_metric",
                                   expected_args={"metric": "not_a_metric"})])
        assert RULE_MALFORMED_ARGS in rep.counts

    def test_task_with_no_correct_answer_caught(self):
        rep = validate_suite([Task(id="x", category="single_tool", text="q",
                                   expected_tool=None)])
        assert RULE_NO_ANSWER in rep.counts


class TestTaskSelection:
    def test_selection_is_deterministic(self):
        tasks = load_tasks()
        assert [t.id for t in select(tasks, 6)] == [t.id for t in select(tasks, 6)]

    def test_selection_is_balanced_across_categories(self):
        # Taking the first N would give six single_tool tasks and no no_tool
        # ones, and unnecessary-invocation would stop being measurable.
        picked = select(load_tasks(), 8)
        assert len({t.category for t in picked}) >= 3

    def test_selection_of_all_returns_everything(self):
        tasks = load_tasks()
        assert len(select(tasks, None)) == len(tasks)
        assert len(select(tasks, 10_000)) == len(tasks)

    def test_selection_preserves_suite_order(self):
        tasks = load_tasks()
        order = {t.id: i for i, t in enumerate(tasks)}
        picked = [order[t.id] for t in select(tasks, 9)]
        assert picked == sorted(picked)


class TestDatasetGates:
    @pytest.fixture(scope="class")
    def corpus(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("corpus")
        generate(DatasetConfig(seed=99, respondents=60, surveys=2), d)
        from toolsmith.data.store import load

        return load(d)

    def test_generated_corpus_is_clean(self, corpus):
        rep = validate_dataset(corpus)
        assert rep.ok, rep.to_json()

    def test_duplicate_response_id_is_caught(self, corpus):
        from dataclasses import replace

        bad = replace(corpus, responses=corpus.responses + [corpus.responses[0]])
        rep = validate_dataset(bad)
        assert "duplicate_response_id" in rep.counts

    def test_out_of_range_value_is_caught(self, corpus):
        from dataclasses import replace

        broken = dict(corpus.responses[0])
        broken["value"] = 999
        broken["response_id"] = "broken_1"
        rep = validate_dataset(replace(corpus, responses=corpus.responses + [broken]))
        assert "response_value_out_of_range" in rep.counts

    def test_orphan_question_reference_is_caught(self, corpus):
        from dataclasses import replace

        broken = dict(corpus.responses[0])
        broken["question_id"] = "ghost_q"
        broken["response_id"] = "broken_2"
        rep = validate_dataset(replace(corpus, responses=corpus.responses + [broken]))
        assert "response_references_missing_question" in rep.counts

    def test_wave_ordering_violation_is_caught(self, corpus):
        from dataclasses import replace

        s = dict(corpus.surveys[0])
        s["waves"] = ["w3", "w1", "w2"]
        rep = validate_dataset(replace(corpus, surveys=[s] + corpus.surveys[1:]))
        assert "wave_ordering_violation" in rep.counts


class TestDeterminism:
    def test_same_seed_produces_an_identical_corpus(self):
        cfg = DatasetConfig(seed=4242, respondents=60, surveys=2)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(cfg, Path(a))
            generate(cfg, Path(b))
            for name in ("surveys", "questions", "respondents", "responses"):
                ta = (Path(a) / f"{name}.json").read_text(encoding="utf-8")
                tb = (Path(b) / f"{name}.json").read_text(encoding="utf-8")
                assert ta == tb, f"{name}.json differs between identical runs"

    def test_different_seed_produces_a_different_corpus(self):
        # Otherwise the determinism test above would pass for a generator that
        # ignores its seed entirely.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(DatasetConfig(seed=1, respondents=60, surveys=2), Path(a))
            generate(DatasetConfig(seed=2, respondents=60, surveys=2), Path(b))
            ta = (Path(a) / "responses.json").read_text(encoding="utf-8")
            tb = (Path(b) / "responses.json").read_text(encoding="utf-8")
            assert ta != tb

    def test_shipped_configs_load(self):
        for name in ("configs/quick.yaml", "configs/full.yaml"):
            cfg = load_config(name)
            assert cfg.grid.tool_variants and cfg.grid.prompt_variants
            assert cfg.model.temperature == 0.0, "ablation requires greedy decoding"

    def test_quick_tier_is_a_strict_subset_of_full(self):
        q, f = load_config("configs/quick.yaml"), load_config("configs/full.yaml")
        assert set(q.grid.tool_variants) <= set(f.grid.tool_variants)
        assert set(q.grid.prompt_variants) <= set(f.grid.prompt_variants)
