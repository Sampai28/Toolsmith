"""Power calculations, checked against values worked out independently.

These numbers end up in the README as design claims, so they are pinned
against textbook results rather than against whatever the code currently
returns. Where a published figure exists it is cited in the test.
"""

from __future__ import annotations

import math

import pytest

from toolsmith.harness import power as P


class TestTwoProportionPower:
    def test_no_difference_gives_alpha(self):
        # With p1 == p2 the rejection rate is the false positive rate.
        assert P.two_proportion_power(0.5, 0.5, 100) == pytest.approx(0.05)

    def test_power_rises_with_n(self):
        small = P.two_proportion_power(0.5, 0.7, 20)
        large = P.two_proportion_power(0.5, 0.7, 200)
        assert large > small

    def test_power_rises_with_effect_size(self):
        assert (P.two_proportion_power(0.5, 0.8, 50)
                > P.two_proportion_power(0.5, 0.6, 50))

    def test_matches_a_textbook_case(self):
        # Classic two-proportion sizing: p1=0.5, p2=0.7, alpha .05, power .80
        # needs ~93 per group (Fleiss, without continuity correction).
        n = P.required_n(0.5, 0.7)
        assert 80 <= n <= 110, n

    def test_small_n_is_badly_underpowered(self):
        # The design claim this project rests on: 4 runs cannot see 20 points.
        assert P.two_proportion_power(0.5, 0.7, 4) < 0.20

    def test_identical_proportions_have_no_required_n(self):
        assert P.required_n(0.5, 0.5) is None


class TestMDE:
    def test_mde_shrinks_as_n_grows(self):
        assert P.mde(400) < P.mde(40)

    def test_quick_tier_cell_mde_is_enormous(self):
        # 4 runs per cell. Anything this reports is the honest ceiling on what
        # a single cell can claim.
        m = P.mde(4)
        assert m is None or m > 0.5, m

    def test_mde_is_reachable_at_realistic_n(self):
        m = P.mde(200)
        assert m is not None and 0.05 < m < 0.25


class TestAttainableOutcomes:
    def test_four_tasks_give_five_possible_rates(self):
        assert P.attainable_outcomes(4) == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_granularity_is_the_real_floor(self):
        # A 10-point difference is not merely underpowered at n=4, it cannot
        # be represented: the nearest representable gap is 25 points.
        rates = P.attainable_outcomes(4)
        gaps = [round(b - a, 4) for a, b in zip(rates, rates[1:])]
        assert min(gaps) == 0.25


class TestMcNemar:
    def test_no_discordance_is_not_significant(self):
        assert P.mcnemar_exact_p(0, 0) == 1.0

    def test_symmetric_discordance_is_not_significant(self):
        assert P.mcnemar_exact_p(5, 5) == pytest.approx(1.0)

    def test_all_flips_one_way_is_significant(self):
        # 0 vs 8 discordant pairs: p = 2 * (1/2)^8 = 0.0078
        assert P.mcnemar_exact_p(0, 8) == pytest.approx(2 * 0.5 ** 8)

    def test_known_small_case(self):
        # b=1, c=9: two-sided exact = 2 * sum(C(10,0..1)) / 2^10 = 2*11/1024
        assert P.mcnemar_exact_p(1, 9) == pytest.approx(2 * 11 / 1024)

    def test_p_is_bounded_at_one(self):
        for b in range(0, 6):
            for c in range(0, 6):
                assert 0.0 <= P.mcnemar_exact_p(b, c) <= 1.0


class TestPairing:
    def test_pairing_never_loses_power_at_these_sizes(self):
        g = P.pairing_gain(0.5, 0.7, 48, correlation=0.7)
        assert g["paired_power_mcnemar"] >= g["unpaired_power"]

    def test_higher_correlation_concentrates_the_signal(self):
        low = P.pairing_gain(0.5, 0.7, 48, correlation=0.3)
        high = P.pairing_gain(0.5, 0.7, 48, correlation=0.9)
        assert high["implied_discordant_rate"] <= low["implied_discordant_rate"]

    def test_discordant_rate_is_at_least_the_difference(self):
        # Two cells differing by 30 points must disagree on at least 30% of
        # tasks; a model implying less than that would be incoherent.
        g = P.pairing_gain(0.4, 0.7, 48, correlation=0.99)
        assert g["implied_discordant_rate"] >= 0.30 - 1e-9


class TestInteraction:
    def test_interaction_needs_four_times_the_main_effect_sample(self):
        # The comparator has to be the main effect estimated from the same
        # 2x2 (variance = var_cell), not a plain two-group test
        # (variance = 2 var_cell). Against the right comparator the ratio is
        # exactly 4, because the interaction's SE is exactly twice as large.
        main = P.required_n_main_effect(0.20, 0.5)
        inter = P.required_n_interaction(0.20, 0.5)
        ratio = inter / main
        assert 3.7 <= ratio <= 4.3, (main, inter, ratio)

    def test_two_group_test_is_not_the_right_comparator(self):
        # Kept explicit so nobody "simplifies" the comparator back: a
        # two-group test needs more n than the averaged main-effect contrast,
        # which would make the interaction penalty look like 2x.
        assert P.required_n(0.5, 0.7) > P.required_n_main_effect(0.20, 0.5)

    def test_interaction_power_is_below_main_effect_power(self):
        n = 48
        assert (P.interaction_power(0.20, n, 0.5)
                < P.main_effect_power(0.20, n, 0.5))

    def test_zero_effect_gives_alpha(self):
        assert P.interaction_power(0.0, 100) == pytest.approx(0.05)


class TestMultiplicity:
    def test_pairwise_count_for_twelve_cells(self):
        assert P.pairwise_count(12) == 66

    def test_family_wise_error_is_near_certain_at_66_tests(self):
        fwer = P.family_wise_error(66)
        assert fwer > 0.95, fwer

    def test_three_prespecified_tests_are_manageable(self):
        assert P.family_wise_error(3) == pytest.approx(1 - 0.95 ** 3)

    def test_holm_threshold_tightens_with_more_tests(self):
        assert P.holm_threshold(66) < P.holm_threshold(3)


class TestVarianceSummary:
    def _rows(self, selections):
        return [
            {"task_id": tid, "correct_selection": v, "argument_correct": v}
            for tid, vals in selections.items()
            for v in vals
        ]

    def test_stable_outcomes_are_deterministic(self):
        rep = P.summarise_variance(
            self._rows({"t1": [True] * 10, "t2": [False] * 10}), "cell")
        assert rep.deterministic
        assert rep.selection_flip_rate == 0.0
        assert rep.repeats == 10

    def test_one_unstable_task_is_reported(self):
        rep = P.summarise_variance(
            self._rows({"t1": [True] * 10,
                        "t2": [True] * 5 + [False] * 5}), "cell")
        assert not rep.deterministic
        assert rep.selection_flip_rate == pytest.approx(0.5)

    def test_none_arguments_are_ignored_not_counted_as_flips(self):
        # argument_correct is None on wrong-tool runs by design; treating
        # None as a distinct value would report spurious instability.
        rows = [{"task_id": "t1", "correct_selection": True,
                 "argument_correct": None} for _ in range(5)]
        rows += [{"task_id": "t1", "correct_selection": True,
                  "argument_correct": True} for _ in range(5)]
        rep = P.summarise_variance(rows, "cell")
        assert rep.argument_flip_rate == 0.0


class TestDesignAnalysis:
    def test_quick_tier_shape(self):
        d = P.analyse_design(cells=12, tasks_per_cell=4, repeats=1)
        assert d["total_runs"] == 48
        assert d["runs_per_cell"] == 4
        assert d["granularity"] == 0.25
        assert d["multiplicity"]["pairwise_comparisons"] == 66

    def test_more_repeats_improve_the_mde(self):
        one = P.analyse_design(cells=12, tasks_per_cell=4, repeats=1)
        three = P.analyse_design(cells=12, tasks_per_cell=4, repeats=3)
        assert three["runs_per_cell"] > one["runs_per_cell"]
        assert (three["mde_per_cell"] or 1.0) <= (one["mde_per_cell"] or 1.0)

    def test_main_effect_is_better_powered_than_a_single_cell(self):
        # A main effect averages over the other axis, so it is backed by more
        # runs than any one cell. If this inverts, the aggregation is wrong.
        d = P.analyse_design(cells=12, tasks_per_cell=4, repeats=1)
        assert d["n_backing_a_main_effect_level"] > d["runs_per_cell"]
        assert (d["mde_main_effect"] or 1.0) <= (d["mde_per_cell"] or 1.0)

    def test_headline_mentions_the_three_required_quantities(self):
        d = P.analyse_design(cells=12, tasks_per_cell=4, repeats=1)
        f = P.analyse_design(cells=16, tasks_per_cell=36, repeats=3)
        text = P.headline(d, f)
        assert "quick tier" in text.lower()
        assert "full.yaml" in text
        assert "pairing" in text.lower()
