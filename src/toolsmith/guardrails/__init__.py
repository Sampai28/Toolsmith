"""Guardrails: allowlist, injection filter, argument validation, egress check.

Each layer is a separate module with its own rule names so a block can be
attributed to one of them. The engine composes them and logs every decision.
"""

from .engine import (  # noqa: F401
    Decision,
    GuardrailEngine,
    GuardrailLog,
    Layer,
)
