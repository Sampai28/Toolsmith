"""Layer 4: egress check on anything leaving the system.

The last three layers all reason about intent or shape. This one reasons about
content: whatever happened upstream, a response carrying respondent
identifiers does not get returned.

It exists because the other layers can be right about every individual
decision and still be collectively wrong. A permitted tool chained toward a
prohibited output — `query_responses` returning respondent_ids, which the
model then formats into a list of "who said what" — never trips the allowlist,
because no forbidden tool was ever called.

Scanning is on the serialised payload, not on a field whitelist, because the
thing being defended against is a value appearing somewhere nobody predicted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

RULE_EMAIL = "egress_email_address"
RULE_RESPONDENT_ID = "egress_respondent_identifier"
RULE_DISPLAY_NAME = "egress_respondent_name"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Respondent ids look like srv_saas_nps_r0042. Response ids (…_x000123) are
# not identifiers of a person and are deliberately not matched: blocking them
# would make query_responses unusable for the legitimate tasks that need it.
_RESPONDENT_ID = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)*_r\d{3,}\b", re.I)


@dataclass
class EgressFinding:
    rule: str
    detail: str
    sample: str


@dataclass
class EgressResult:
    blocked: bool
    findings: list[EgressFinding] = field(default_factory=list)

    @property
    def rules(self) -> list[str]:
        return [f.rule for f in self.findings]


class EgressCheck:
    def __init__(self, known_names: frozenset[str] | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        # Names are matched against the roster rather than by pattern. "Alex
        # Chen" is indistinguishable from prose without knowing it is a
        # respondent, and a name-shaped regex would block "Likert" and "NPS
        # Promoter" alike.
        self._names = known_names or frozenset()

    @classmethod
    def from_dataset(cls, ds: Any, enabled: bool = True) -> "EgressCheck":
        try:
            names = frozenset(r["display_name"] for r in ds.respondents)
        except Exception:
            names = frozenset()
        return cls(known_names=names, enabled=enabled)

    def check(self, payload: Any) -> EgressResult:
        if not self.enabled:
            return EgressResult(blocked=False)

        text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
        findings: list[EgressFinding] = []

        emails = _EMAIL.findall(text)
        if emails:
            findings.append(EgressFinding(
                RULE_EMAIL, f"{len(emails)} email address(es) in the response", emails[0]))

        ids = _RESPONDENT_ID.findall(text)
        if ids:
            findings.append(EgressFinding(
                RULE_RESPONDENT_ID, f"{len(ids)} respondent identifier(s) in the response",
                ids[0]))

        if self._names:
            hit = next((n for n in self._names if n in text), None)
            if hit is not None:
                findings.append(EgressFinding(
                    RULE_DISPLAY_NAME, "respondent display name in the response", hit))

        return EgressResult(blocked=bool(findings), findings=findings)

    def redact(self, payload: Any) -> Any:
        """Return the payload with identifiers removed.

        Used where blocking outright would be worse than returning a partial
        answer — a legitimate query_responses result is still useful without
        the respondent column.
        """
        text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
        text = _EMAIL.sub("[redacted-email]", text)
        text = _RESPONDENT_ID.sub("[redacted-respondent]", text)
        for n in self._names:
            text = text.replace(n, "[redacted-name]")
        if isinstance(payload, str):
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


__all__ = ["EgressCheck", "EgressResult", "EgressFinding",
           "RULE_EMAIL", "RULE_RESPONDENT_ID", "RULE_DISPLAY_NAME"]
