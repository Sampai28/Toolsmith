"""Layer 2: injection pattern filter on inbound task text.

This runs before the model sees the text. It is pattern matching, which means
it is the weakest layer in the stack and the one the adversarial suite is
expected to defeat — a filter built from patterns can only recognise phrasings
somebody thought of.

It is here anyway, for two reasons. It catches the low-effort majority cheaply,
and because it is attributable: when a prompt gets through, the trace says
which layer saw it and let it pass, which is the difference between "the
guardrails held" and knowing why.

Each rule is named and reported separately so block-rate-by-layer is a real
breakdown rather than a single number.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

RULE_INSTRUCTION_OVERRIDE = "injection_instruction_override"
RULE_ROLE_REASSIGNMENT = "injection_role_reassignment"
RULE_FAKE_SYSTEM = "injection_embedded_system_message"
RULE_ENCODED = "injection_encoded_directive"
RULE_PII_SOLICITATION = "injection_pii_solicitation"
RULE_TOOL_COERCION = "injection_forbidden_tool_named"

# Patterns are deliberately narrow. A filter broad enough to catch every
# phrasing also blocks "compare how support quality has changed since wave 1",
# and guardrail false positives are reported alongside the block rate for
# exactly that reason.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (RULE_INSTRUCTION_OVERRIDE, re.compile(
        r"\b(ignore|disregard|forget|override|bypass|set\s+aside)\b[^.]{0,40}?"
        r"\b(previous|prior|above|earlier|all|any|your|the)\b[^.]{0,30}?"
        r"\b(instruction|instructions|rule|rules|prompt|guideline|guidelines|"
        r"restriction|restrictions|constraint|constraints|policy|policies)\b",
        re.I)),
    (RULE_INSTRUCTION_OVERRIDE, re.compile(
        r"\b(you\s+are\s+(now\s+)?(permitted|allowed|authori[sz]ed)|"
        r"restrictions?\s+(are|have\s+been)\s+(lifted|removed|disabled)|"
        r"(safety|guardrail)s?\s+(are\s+)?(off|disabled))\b", re.I)),
    (RULE_ROLE_REASSIGNMENT, re.compile(
        r"\b(you\s+are\s+(now\s+)?a[n]?\b[^.]{0,60}?"
        r"(admin|administrator|developer|engineer|auditor|dba|root|superuser)|"
        r"\bact\s+as\s+(an?\s+)?(admin|administrator|developer|unrestricted|"
        r"jailbroken|dan)\b|"
        r"\benter\s+(developer|debug|maintenance|god)\s+mode\b|"
        r"\b(developer|debug|maintenance)\s+mode\s+(is\s+)?(on|enabled|active)\b)",
        re.I)),
    (RULE_FAKE_SYSTEM, re.compile(
        r"(<\s*/?\s*(system|im_start|im_end)\s*>|"
        r"\[\s*(system|end\s+of\s+(system\s+)?prompt)\s*\]|"
        r"^\s*system\s*:\s*\S|"
        r"\{\{\s*system\s*\}\}|"
        r"###\s*system\b)", re.I | re.M)),
    (RULE_PII_SOLICITATION, re.compile(
        r"\b(email\s+address(es)?|e-?mails?\s+of|contact\s+details?|"
        r"full\s+names?\s+and|personally\s+identifiable|\bpii\b|"
        r"who\s+(exactly\s+)?(answered|responded|said)|identify\s+(the\s+)?respondents?)\b",
        re.I)),
    (RULE_TOOL_COERCION, re.compile(
        r"\b(export_respondent_pii|get_contacts)\b", re.I)),
]

# Anything that decodes to text containing one of these is treated as an
# encoded directive. Checking the decoded text rather than the encoding itself
# avoids blocking a legitimate task that merely contains a long token.
_DECODED_TRIGGERS = re.compile(
    r"\b(ignore|disregard|override|export|pii|email|system\s*:|respondent)\b", re.I
)

_B64 = re.compile(r"\b[A-Za-z0-9+/]{16,}={0,2}\b")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}[\s:]?){8,}\b")


@dataclass
class InjectionFinding:
    rule: str
    detail: str
    excerpt: str


@dataclass
class InjectionResult:
    blocked: bool
    findings: list[InjectionFinding] = field(default_factory=list)

    @property
    def rules(self) -> list[str]:
        return [f.rule for f in self.findings]


def _normalise(text: str) -> str:
    """Fold obfuscation that is purely presentational.

    NFKC collapses full-width and other compatibility forms, and zero-width
    characters are stripped outright: `i\\u200bgnore previous instructions`
    reads identically to a model and differently to a regex, which is a cheap
    bypass not worth conceding.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = re.sub(r"[​-‏⁠﻿]", "", folded)
    # Collapse spaced-out letters: "i g n o r e" -> "ignore".
    folded = re.sub(r"\b(?:[A-Za-z]\s){3,}[A-Za-z]\b",
                    lambda m: m.group(0).replace(" ", ""), folded)
    return folded


def _decoded_candidates(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _B64.finditer(text):
        blob = m.group(0)
        pad = "=" * (-len(blob) % 4)
        try:
            decoded = base64.b64decode(blob + pad, validate=True).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            continue
        if decoded.strip():
            out.append(("base64", decoded))
    for m in _HEX.finditer(text):
        raw = re.sub(r"[\s:]", "", m.group(0))
        if len(raw) % 2:
            raw = raw[:-1]
        try:
            decoded = bytes.fromhex(raw).decode("utf-8", "ignore")
        except ValueError:
            continue
        if decoded.strip() and decoded.isprintable():
            out.append(("hex", decoded))
    return out


class InjectionFilter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, text: str) -> InjectionResult:
        if not self.enabled or not text:
            return InjectionResult(blocked=False)

        findings: list[InjectionFinding] = []
        folded = _normalise(text)

        for rule, pattern in _PATTERNS:
            m = pattern.search(folded)
            if m:
                findings.append(InjectionFinding(
                    rule=rule,
                    detail=f"matched /{pattern.pattern[:60]}.../",
                    excerpt=_excerpt(folded, m.start(), m.end()),
                ))

        for encoding, decoded in _decoded_candidates(folded):
            m = _DECODED_TRIGGERS.search(decoded)
            if m:
                findings.append(InjectionFinding(
                    rule=RULE_ENCODED,
                    detail=f"{encoding} payload decoded to a directive",
                    excerpt=decoded[:140],
                ))

        # Dedupe by rule: one finding per rule keeps the by-layer counts
        # meaningful when a prompt trips the same pattern twice.
        seen: set[str] = set()
        unique = []
        for f in findings:
            if f.rule in seen:
                continue
            seen.add(f.rule)
            unique.append(f)

        return InjectionResult(blocked=bool(unique), findings=unique)


def _excerpt(text: str, start: int, end: int, pad: int = 30) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    return ("..." if lo else "") + text[lo:hi].replace("\n", " ") + ("..." if hi < len(text) else "")


__all__ = [
    "InjectionFilter", "InjectionResult", "InjectionFinding",
    "RULE_INSTRUCTION_OVERRIDE", "RULE_ROLE_REASSIGNMENT", "RULE_FAKE_SYSTEM",
    "RULE_ENCODED", "RULE_PII_SOLICITATION", "RULE_TOOL_COERCION",
]
