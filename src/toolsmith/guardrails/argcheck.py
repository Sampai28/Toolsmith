"""Layer 3: argument validation against the tool's JSON Schema.

An allowlist decides *whether* a tool may run. This decides whether it may run
with the arguments it was handed. The gap between the two is a real attack
surface: `query_responses` is permitted, and `query_responses(limit=999999)`
or a segment filter naming a field that does not exist are both ways to drive
a permitted tool somewhere it should not go.

Only the subset of JSON Schema the tool registry actually uses is implemented
— types, enums, required, numeric bounds, array item types, and
additionalProperties. A general validator would be a dependency and a larger
surface to be wrong in; this one is small enough to read and is covered by
boundary tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RULE_UNKNOWN_PARAM = "arg_unknown_parameter"
RULE_MISSING_REQUIRED = "arg_missing_required"
RULE_WRONG_TYPE = "arg_wrong_type"
RULE_NOT_IN_ENUM = "arg_value_not_in_enum"
RULE_OUT_OF_RANGE = "arg_value_out_of_range"
RULE_NOT_AN_OBJECT = "arg_payload_not_an_object"

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass
class ArgViolation:
    rule: str
    parameter: str
    detail: str


@dataclass
class ArgCheckResult:
    valid: bool
    violations: list[ArgViolation] = field(default_factory=list)

    @property
    def rules(self) -> list[str]:
        return [v.rule for v in self.violations]

    def summary(self) -> str:
        return "; ".join(f"{v.parameter}: {v.detail}" for v in self.violations)


def _type_ok(value: Any, expected: str) -> bool:
    types = _TYPE_CHECKS.get(expected)
    if types is None:
        return True
    # bool is a subclass of int in Python, and accepting True where an integer
    # limit is expected is how `limit=True` becomes `limit=1` silently.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, types)


def validate_arguments(schema: dict[str, Any], args: Any) -> ArgCheckResult:
    violations: list[ArgViolation] = []

    if not isinstance(args, dict):
        return ArgCheckResult(
            valid=False,
            violations=[ArgViolation(RULE_NOT_AN_OBJECT, "<payload>",
                                     f"expected an object, got {type(args).__name__}")],
        )

    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    allow_extra = schema.get("additionalProperties", True) is not False

    for name in required:
        if name not in args or args[name] is None:
            violations.append(ArgViolation(RULE_MISSING_REQUIRED, name,
                                           "required parameter is absent"))

    for name, value in args.items():
        spec = properties.get(name)
        if spec is None:
            if not allow_extra:
                violations.append(ArgViolation(
                    RULE_UNKNOWN_PARAM, name,
                    f"not a parameter of this tool (known: {', '.join(sorted(properties)) or 'none'})",
                ))
            continue
        if value is None:
            continue
        violations.extend(_check_value(name, value, spec))

    return ArgCheckResult(valid=not violations, violations=violations)


def _check_value(name: str, value: Any, spec: dict[str, Any]) -> list[ArgViolation]:
    out: list[ArgViolation] = []
    expected = spec.get("type")

    if expected and not _type_ok(value, expected):
        out.append(ArgViolation(RULE_WRONG_TYPE, name,
                                f"expected {expected}, got {type(value).__name__}"))
        return out  # further checks would be noise once the type is wrong

    enum = spec.get("enum")
    if enum is not None and value not in enum:
        out.append(ArgViolation(RULE_NOT_IN_ENUM, name,
                                f"{value!r} is not one of {', '.join(map(str, enum))}"))

    if expected in ("integer", "number"):
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None and value < lo:
            out.append(ArgViolation(RULE_OUT_OF_RANGE, name, f"{value} below minimum {lo}"))
        if hi is not None and value > hi:
            out.append(ArgViolation(RULE_OUT_OF_RANGE, name, f"{value} above maximum {hi}"))

    if expected == "array":
        item_spec = spec.get("items") or {}
        item_type = item_spec.get("type")
        if item_type:
            for i, item in enumerate(value):
                if not _type_ok(item, item_type):
                    out.append(ArgViolation(
                        RULE_WRONG_TYPE, f"{name}[{i}]",
                        f"expected {item_type}, got {type(item).__name__}"))

    if expected == "object":
        extra = spec.get("additionalProperties")
        if isinstance(extra, dict) and extra.get("type"):
            for key, item in value.items():
                if not _type_ok(item, extra["type"]):
                    out.append(ArgViolation(
                        RULE_WRONG_TYPE, f"{name}.{key}",
                        f"expected {extra['type']}, got {type(item).__name__}"))

    return out


__all__ = [
    "validate_arguments", "ArgCheckResult", "ArgViolation",
    "RULE_UNKNOWN_PARAM", "RULE_MISSING_REQUIRED", "RULE_WRONG_TYPE",
    "RULE_NOT_IN_ENUM", "RULE_OUT_OF_RANGE", "RULE_NOT_AN_OBJECT",
]
