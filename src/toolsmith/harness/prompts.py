"""Axis 2 of the ablation: the system prompt.

Four variants from two independent choices — base prose (terse or detailed)
crossed with few-shot examples (absent or present). Keeping few-shot as a
suffix on a shared block rather than four hand-written files means
`terse_fewshot` and `detailed_fewshot` differ by exactly their base, which is
what makes the "+few-shot" main effect a clean contrast.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

from ..config import PROMPTS_DIR

VARIANTS = ("terse", "terse_fewshot", "detailed", "detailed_fewshot")

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


@functools.lru_cache(maxsize=8)
def _read(name: str) -> str:
    path = Path(PROMPTS_DIR) / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt variant file missing: {path}")
    text = path.read_text(encoding="utf-8")
    # Authoring notes live in HTML comments so the files stay readable without
    # shipping commentary to the model as if it were instruction.
    return _HTML_COMMENT.sub("", text).strip()


def system_prompt(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {VARIANTS}")
    base = "detailed" if variant.startswith("detailed") else "terse"
    text = _read(base)
    if variant.endswith("_fewshot"):
        text = f"{text}\n\n{_read('fewshot')}"
    return text


def describe_variant(variant: str) -> dict[str, object]:
    text = system_prompt(variant)
    return {
        "variant": variant,
        "base": "detailed" if variant.startswith("detailed") else "terse",
        "fewshot": variant.endswith("_fewshot"),
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }


def axis_levels(variant: str) -> tuple[str, bool]:
    """(base, fewshot) — the two factors this axis actually crosses."""
    return ("detailed" if variant.startswith("detailed") else "terse",
            variant.endswith("_fewshot"))
