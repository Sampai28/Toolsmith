"""MCP protocol conformance: tools list, call, and error responses.

These drive the real server over a real stdio session, the same path the
harness uses. Importing the handlers directly would be faster and would test
a thing nobody runs.
"""

from __future__ import annotations

import pytest

from toolsmith.data import store
from toolsmith.data.store import DatasetError
from toolsmith.harness.mcp_client import MCPSession
from toolsmith.tools.registry import FORBIDDEN_TOOL, TOOL_NAMES
from toolsmith.tools.variants import VARIANTS, alias_map


def corpus_available() -> bool:
    try:
        store.default_dataset()
        return True
    except DatasetError:
        return False


requires_corpus = pytest.mark.skipif(
    not corpus_available(), reason="corpus not generated; run `make generate`"
)


@pytest.fixture(scope="module")
def session():
    s = MCPSession(tool_variant="detailed")
    s.start()
    yield s
    s.close()


@requires_corpus
class TestProtocolConformance:
    def test_lists_every_tool(self, session):
        names = {t["name"] for t in session.list_tools()}
        assert names == set(TOOL_NAMES)

    def test_forbidden_tool_is_discoverable(self, session):
        # Hiding it would make the adversarial suite test discoverability
        # rather than enforcement.
        names = {t["name"] for t in session.list_tools()}
        assert FORBIDDEN_TOOL in names

    def test_every_tool_has_a_description_and_schema(self, session):
        for t in session.list_tools():
            assert t["description"], f"{t['name']} has no description"
            schema = t["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_successful_call_returns_a_result(self, session):
        r = session.call_tool("aggregate_metric", {
            "survey_id": "srv_saas_nps", "question": "satisfaction", "metric": "mean"
        })
        assert r.ok
        assert r.payload["metric"] == "mean"
        assert isinstance(r.payload["value"], (int, float))
        assert r.payload["n"] > 0

    def test_list_surveys_takes_no_arguments(self, session):
        r = session.call_tool("list_surveys", {})
        assert r.ok
        assert len(r.payload["surveys"]) >= 1

    def test_unknown_tool_is_an_error_not_a_crash(self, session):
        r = session.call_tool("no_such_tool", {})
        assert not r.ok

    def test_unknown_survey_returns_a_tool_error(self, session):
        r = session.call_tool("aggregate_metric", {
            "survey_id": "does_not_exist", "question": "satisfaction", "metric": "mean"
        })
        assert not r.ok


@requires_corpus
class TestGuardrailsAreServerSide:
    """A caller that skips the harness must still not reach the forbidden tool."""

    def test_forbidden_tool_blocked_by_the_server(self, session):
        r = session.call_tool(FORBIDDEN_TOOL, {"survey_id": "srv_saas_nps"})
        assert r.blocked
        assert r.layer == "allowlist"
        assert r.rule == "tool_not_allowlisted"

    def test_out_of_range_argument_blocked_with_an_attributable_rule(self, session):
        r = session.call_tool("query_responses", {"survey_id": "srv_saas_nps",
                                                  "limit": 999999})
        assert r.blocked
        assert r.layer == "argument_validation"
        assert r.rule == "arg_value_out_of_range"

    def test_invented_segment_field_blocked(self, session):
        r = session.call_tool("compare_segments", {
            "survey_id": "srv_saas_nps", "question": "satisfaction",
            "metric": "mean", "segment_field": "email_address",
        })
        assert r.blocked
        assert r.rule == "arg_value_not_in_enum"

    def test_legitimate_row_query_is_not_blocked(self, session):
        r = session.call_tool("query_responses", {
            "survey_id": "srv_saas_nps", "question": "nps", "limit": 5
        })
        assert r.ok, "a permitted row-level query must survive the egress check"


@requires_corpus
class TestAmbiguousNamingVariant:
    def test_aliased_forbidden_tool_is_still_blocked(self):
        # The case an allowlist keyed on displayed names would miss entirely.
        with MCPSession(tool_variant="ambiguous") as s:
            names = {t["name"] for t in s.list_tools()}
            assert "get_contacts" in names
            assert FORBIDDEN_TOOL not in names

            r = s.call_tool("get_contacts", {"survey_id": "srv_saas_nps"})
            assert r.blocked
            assert r.rule == "tool_not_allowlisted"

    def test_aliased_permitted_tool_works(self):
        with MCPSession(tool_variant="ambiguous") as s:
            r = s.call_tool("get_results", {
                "survey_id": "srv_saas_nps", "question": "satisfaction", "metric": "mean"
            })
            assert r.ok


class TestVariantsExposeTheSameCapabilities:
    """Only the prose changes. A variant that changed the schema would be a
    different capability, and the ablation would not be measuring description
    quality."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_variant_renders_every_tool(self, variant):
        from toolsmith.tools.variants import render

        assert len(render(variant)) == len(TOOL_NAMES)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_aliases_round_trip_to_canonical_names(self, variant):
        assert set(alias_map(variant).values()) == set(TOOL_NAMES)

    def test_only_ambiguous_renames_tools(self):
        for variant in VARIANTS:
            m = alias_map(variant)
            renamed = any(k != v for k, v in m.items())
            assert renamed == (variant == "ambiguous")

    def test_terse_renames_parameters_and_others_do_not(self):
        from toolsmith.tools.variants import param_map

        assert param_map("terse")
        for variant in ("detailed", "example", "ambiguous"):
            assert param_map(variant) == {}
