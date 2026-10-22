"""Schema-aware probability views for generated JSON leaf values.

This is presentation/trace analysis, not part of grammar enforcement. Grouped
probabilities use the recorded top token candidates and are therefore diagnostic
approximations rather than complete sequence probabilities over every token path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any

from .trace import GenerationTrace, TokenProbability


@dataclass(frozen=True)
class GroupedValueProbability:
    value: str | int
    probability: float
    log_probability: float


@dataclass(frozen=True)
class ValueTokenDetails:
    step: int
    selected_token_id: int
    selected_token: str
    candidates: list[TokenProbability]


@dataclass(frozen=True)
class ValueProbabilityAnalysis:
    value: str
    value_type: str
    step_start: int
    step_end: int
    heading: str
    grouped: list[GroupedValueProbability]
    selected_path_probability: float
    selected_path_log_probability: float
    token_details: list[ValueTokenDetails] = field(default_factory=list)


def json_leaf_value_spans(text: str) -> list[tuple[int, int]]:
    """Return scalar value spans, excluding field names and JSON punctuation."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] == '"':
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    break
                end += 1
            following = end + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following >= len(text) or text[following] != ":":
                # Keep empty spans so schema leaves remain positionally aligned.
                spans.append((index + 1, end))
            index = end + 1
            continue
        if text.startswith("true", index):
            spans.append((index, index + 4))
            index += 4
            continue
        if text.startswith("false", index):
            spans.append((index, index + 5))
            index += 5
            continue
        if text[index] == "-" or text[index].isdigit():
            end = index + 1
            while end < len(text) and text[end] in "0123456789.eE+-":
                end += 1
            spans.append((index, end))
            index = end
            continue
        index += 1
    return spans


def leaf_value_options(schema: dict[str, Any], value: Any) -> list[tuple[str, list[str] | None]]:
    """Finite legal values in the same order as JSON scalar leaves."""
    schema_type = schema["type"]
    if schema_type == "object":
        result: list[tuple[str, list[str] | None]] = []
        for name, child in schema["properties"].items():
            if name in value:
                result.extend(leaf_value_options(child, value[name]))
        return result
    if schema_type == "array":
        result = []
        for item in value:
            result.extend(leaf_value_options(schema["items"], item))
        return result
    if schema_type == "string" and "enum" in schema:
        values = [json.dumps(item, ensure_ascii=False)[1:-1] for item in schema["enum"]]
        return [("string", values)]
    if schema_type == "boolean":
        return [("boolean", ["true", "false"])]
    return [(schema_type, None)]


def candidate_value_fragment(text: str, string_value: bool) -> str:
    if not string_value:
        text = text.lstrip()
        end = 0
        while end < len(text) and text[end].isalpha():
            end += 1
        return text[:end]
    end = 0
    escaped = False
    while end < len(text):
        char = text[end]
        if char == '"' and not escaped:
            break
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
        end += 1
    return text[:end]


def distinct_string_token_options(
    candidates: list[TokenProbability], limit: int = 5
) -> list[tuple[str, float]]:
    """Highest-probability string starts without mutually included prefixes."""
    by_fragment: dict[str, float] = {}
    for candidate in candidates:
        fragment = candidate_value_fragment(candidate.text, string_value=True)
        if fragment:
            by_fragment[fragment] = by_fragment.get(fragment, 0.0) + candidate.probability
    ranked = sorted(by_fragment.items(), key=lambda item: item[1], reverse=True)
    selected: list[tuple[str, float]] = []
    for fragment, probability in ranked:
        normalized = fragment.strip().casefold()
        if not normalized:
            continue
        if any(
            normalized.startswith(existing.strip().casefold())
            or existing.strip().casefold().startswith(normalized)
            for existing, _ in selected
        ):
            continue
        selected.append((fragment, probability))
        if len(selected) == limit:
            break
    return selected


def grouped_integer_probabilities(
    candidates: list[TokenProbability],
) -> list[tuple[int, float]]:
    grouped: dict[int, float] = {}
    for candidate in candidates:
        match = re.match(r"-?(?:0|[1-9][0-9]*)", candidate.text.lstrip())
        if match:
            value = int(match.group(0))
            grouped[value] = grouped.get(value, 0.0) + candidate.probability
    return _normalize_rounding(grouped)


def grouped_value_probabilities(
    candidates: list[TokenProbability],
    prefix: str,
    legal_values: list[str],
    string_value: bool,
) -> list[tuple[str, float]]:
    """Split an ambiguous prefix's mass evenly over compatible final values."""
    compatible = [
        value for value in legal_values
        if value.startswith(prefix) or prefix.startswith(value)
    ]
    grouped = {value: 0.0 for value in compatible}
    for candidate in candidates:
        fragment = candidate_value_fragment(candidate.text, string_value)
        combined = prefix + fragment
        matches = [
            value for value in compatible
            if value.startswith(combined) or combined.startswith(value)
        ]
        if matches:
            share = candidate.probability / len(matches)
            for value in matches:
                grouped[value] += share
    return _normalize_rounding(grouped)


def _normalize_rounding(grouped: dict[Any, float]) -> list[tuple[Any, float]]:
    total = sum(grouped.values())
    if total > 1.0:  # float32 can exceed one by a few ULPs
        grouped = {value: probability / total for value, probability in grouped.items()}
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def analyze_value_probabilities(
    trace: GenerationTrace,
    schema: dict[str, Any],
    include_token_details: bool = False,
) -> list[ValueProbabilityAnalysis]:
    """Build the compact value-level probability view used by tests and UI."""
    if trace.status != "abgeschlossen":
        return []
    parsed = json.loads(trace.text)
    leaves = list(zip(json_leaf_value_spans(trace.text), leaf_value_options(schema, parsed)))
    token_ranges = []
    token_start = 0
    for step in trace.steps:
        token_end = len(step.generated_text)
        token_ranges.append((token_start, token_end, step))
        token_start = token_end

    analyses: list[ValueProbabilityAnalysis] = []
    for (start, end), (value_type, legal_values) in leaves:
        value_steps = [
            step for token_start, token_end, step in token_ranges
            if start < token_end and end > token_start
        ]
        if not value_steps:
            continue
        first_step = value_steps[0]
        selected_log = sum(
            math.log(step.selected_constrained_probability) for step in value_steps
        )
        candidates = first_step.constrained_top20[:10]
        heading = "Gruppierte legale Werte (Masse der Token-Top-10)"
        if legal_values:
            grouped = grouped_value_probabilities(
                candidates, "", legal_values, string_value=value_type == "string"
            )[:10]
        elif value_type == "integer":
            grouped = grouped_integer_probabilities(candidates)[:5]
        elif value_type == "string":
            grouped = distinct_string_token_options(first_step.constrained_top20, 5)
            heading = "Erste unterschiedliche String-Tokens"
        else:
            grouped = []

        details = []
        if include_token_details:
            details = [
                ValueTokenDetails(
                    step=step.step,
                    selected_token_id=step.selected_token_id,
                    selected_token=step.selected_token,
                    candidates=step.constrained_top20[:10],
                )
                for step in value_steps
            ]
        analyses.append(ValueProbabilityAnalysis(
            value=trace.text[start:end],
            value_type=value_type,
            step_start=first_step.step,
            step_end=value_steps[-1].step,
            heading=heading,
            grouped=[
                GroupedValueProbability(
                    value=value,
                    probability=probability,
                    log_probability=math.log(probability) if probability else -math.inf,
                )
                for value, probability in grouped
            ],
            selected_path_probability=math.exp(selected_log),
            selected_path_log_probability=selected_log,
            token_details=details,
        ))
    return analyses


def format_value_probabilities(analyses: list[ValueProbabilityAnalysis]) -> str:
    """Human-readable console rendering of the shared structured analysis."""
    lines: list[str] = []
    for analysis in analyses:
        label = str(analysis.step_start) if analysis.step_start == analysis.step_end else f"{analysis.step_start}–{analysis.step_end}"
        lines.append(f"Value {analysis.value!r} (Schritt {label})")
        for detail in analysis.token_details:
            lines.append(
                f"  Token-Schritt {detail.step}: gewählt {detail.selected_token} "
                f"(ID={detail.selected_token_id})"
            )
            for rank, candidate in enumerate(detail.candidates, 1):
                lines.append(
                    f"  {rank:2}. ID={candidate.token_id:<7} Token={candidate.token:<18} "
                    f"p={candidate.probability:.8f} log(p)={candidate.log_probability:.6f}"
                )
        if analysis.grouped:
            lines.append(f"  {analysis.heading}:")
            for rank, option in enumerate(analysis.grouped, 1):
                lines.append(
                    f"  {rank:2}. Value={option.value!r:<18} "
                    f"p={option.probability:.8f} log(p)={option.log_probability:.6f}"
                )
        else:
            lines.append(
                "  Wahrscheinlichkeit des gewählten Value-Tokenpfads: "
                f"p={analysis.selected_path_probability:.8g} "
                f"log(p)={analysis.selected_path_log_probability:.6f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
