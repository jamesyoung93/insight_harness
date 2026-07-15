"""Optional, structured, artifact-only phrasing for digest items.

The narrator cannot rank or compute. It must return a bounded JSON object, and
the response is accepted only when deterministic validation preserves metric,
scope, entities, direction, units, and every numeric claim while making no
causal assertion.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping

from .digest import DigestItemArtifact
from .llm_translator import DEFAULT_MODEL


_NUMBER = re.compile(
    r"(?<![\w])(?P<prefix>\$)?(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"(?P<suffix>%|σ)?")
_CAUSAL = re.compile(
    r"\b(caus(?:e|ed|es|ing|al)|because|due to|driven by|drove|"
    r"lead(?:s|ing)? to|led to|result(?:s|ed|ing)? (?:in|from)|as a result(?: of)?|"
    r"attribut(?:e|ed|es|ing|ion)|explain(?:s|ed|ing)?|"
    r"produc(?:e|ed|es|ing)|generat(?:e|ed|es|ing)|"
    r"contribut(?:e|ed|es|ing)(?: to)?|boost(?:s|ed|ing)?|"
    r"trigger(?:s|ed|ing)?|accelerat(?:e|ed|es|ing)|lift(?:s|ed|ing)?|"
    r"suppress(?:es|ed|ing)?|depress(?:es|ed|ing)?|influenc(?:e|ed|es|ing)|"
    r"spur(?:s|red|ring)?|spark(?:s|ed|ing)?|prompt(?:s|ed|ing)?|"
    r"induc(?:e|ed|es|ing)|yield(?:s|ed|ing)?|enabled?|responsible for|"
    r"impact of|effect of|in response to)\b", re.I)
_DIRECTION_WORDS = {
    "above": re.compile(r"\b(above|higher|up|increase(?:d)?|positive)\b", re.I),
    "below": re.compile(r"\b(below|lower|down|decrease(?:d)?|negative)\b", re.I),
    "different": re.compile(r"\b(differ(?:s|ent|ence)?|alternate|fork)\b", re.I),
}
_OPPOSITE = {"above": "below", "below": "above"}
_RESPONSE_FIELDS = {"text", "metric", "scope", "direction", "causal"}

SYSTEM = """Rewrite one computed digest headline into one concise sentence.
Return JSON only, with exactly these fields:
{"text": string, "metric": string, "scope": string,
 "direction": "above"|"below"|"different"|"neutral", "causal": false}
Copy metric, scope, and expected_direction exactly from the supplied JSON. The text
must explicitly name metric_label and scope, preserve every number, sign, and unit
from templated_headline, add no number, and use the supplied direction. Never claim
that an event caused, drove, explained, or produced a movement."""


def rewrite_policy(item: DigestItemArtifact) -> tuple[bool, str]:
    """Return whether model phrasing is allowed for this artifact class.

    Registered-event candidates intentionally retain their governed template.
    Even a numerically faithful paraphrase could turn temporal overlap into an
    attribution, so these items never leave the deterministic phrasing path.
    """

    if item.candidate.kind == "event" or item.candidate.event_id is not None:
        return False, "registered-event candidates use deterministic non-causal wording"
    return True, "model phrasing is permitted subject to deterministic validation"


def numeric_values(text: str) -> tuple[float, ...]:
    """Compatibility helper returning all parsed numeric values."""

    return tuple(value for value, _unit in numeric_claims(text))


def numeric_claims(text: str) -> tuple[tuple[float, str], ...]:
    claims = []
    for match in _NUMBER.finditer(text):
        try:
            value = float(match.group("number").replace(",", ""))
        except ValueError:
            continue
        unit = ({"$": "currency"}.get(match.group("prefix"))
                or {"%": "percent", "σ": "sigma"}.get(match.group("suffix"))
                or "number")
        claims.append((value, unit))
    return tuple(claims)


def _claim_key(claim: tuple[float, str]) -> tuple[str, str]:
    value, unit = claim
    return (format(value, ".12g"), unit)


def expected_direction(item: DigestItemArtifact) -> str:
    if item.candidate.kind == "divergence":
        return "different"
    facts = item.candidate.facts
    if facts is None:
        return "neutral"
    return "above" if facts.absolute_change >= 0 else "below"


def _decode_response(response: str | Mapping) -> tuple[dict | None, str | None]:
    if isinstance(response, Mapping):
        payload = dict(response)
    else:
        try:
            payload = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return None, "the narrator response was not a JSON object"
    if not isinstance(payload, dict):
        return None, "the narrator response was not a JSON object"
    if set(payload) != _RESPONSE_FIELDS:
        return None, "the narrator response did not match the bounded schema"
    return payload, None


def validate_rewrite(item: DigestItemArtifact,
                     response: str | Mapping) -> tuple[bool, str]:
    """Deterministically reject altered facts or semantically unsafe prose."""

    allowed, policy_reason = rewrite_policy(item)
    if not allowed:
        return False, policy_reason
    payload, error = _decode_response(response)
    if payload is None:
        return False, str(error)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or "\n" in text or len(text) > 800:
        return False, "the rewrite must be one bounded, non-empty line"
    cleaned = text.strip()
    expected_scope = item.fact_payload()["scope"]
    expected = expected_direction(item)
    if payload.get("metric") != item.candidate.metric:
        return False, "the rewrite changed the governed metric"
    if payload.get("scope") != expected_scope:
        return False, "the rewrite changed the governed scope"
    if payload.get("direction") != expected:
        return False, "the rewrite changed the computed direction"
    if payload.get("causal") is not False or _CAUSAL.search(cleaned):
        return False, "the rewrite introduced a causal claim"

    metric_label = str(item.fact_payload()["metric_label"])
    if metric_label.casefold() not in cleaned.casefold():
        return False, "the rewrite omitted the governed metric label"
    if expected_scope.casefold() not in cleaned.casefold():
        return False, "the rewrite omitted the governed scope"
    for value in item.candidate.filter_dict.values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        if any(str(entity).casefold() not in cleaned.casefold() for entity in values):
            return False, "the rewrite omitted a governed scope entity"

    required = Counter(_claim_key(claim) for claim in numeric_claims(item.template_headline))
    proposed = Counter(_claim_key(claim) for claim in numeric_claims(cleaned))
    if proposed != required:
        return False, "the rewrite changed a computed number, sign, unit, or multiplicity"

    if expected in _DIRECTION_WORDS:
        if not _DIRECTION_WORDS[expected].search(cleaned):
            return False, "the rewrite omitted the computed direction"
        opposite = _OPPOSITE.get(expected)
        if opposite and _DIRECTION_WORDS[opposite].search(cleaned):
            return False, "the rewrite contradicted the computed direction"
    return True, "validated against metric, scope, direction, units, and numeric facts"


def rewrite_item(item: DigestItemArtifact, api_key: str | None = None,
                 model: str = DEFAULT_MODEL, timeout_seconds: float = 12.0
                 ) -> DigestItemArtifact:
    """Return a validated structured rewrite or the deterministic template."""

    allowed, policy_reason = rewrite_policy(item)
    if not allowed:
        return item.with_narration(item.template_headline, {
            "narrator": "template", "validated": True,
            "fallback_kind": "policy", "fallback_reason": policy_reason,
        })
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return item.with_narration(item.template_headline, {
            "narrator": "template", "validated": True,
            "fallback_kind": "unavailable", "fallback_reason": "no API key provided",
        })
    started = time.perf_counter()
    raw = ""
    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=key, timeout=timeout_seconds, max_retries=0)
        facts = item.fact_payload() | {
            "expected_direction": expected_direction(item),
            "response_fields": sorted(_RESPONSE_FIELDS),
        }
        message = client.messages.create(
            model=model, max_tokens=220, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(
                facts, sort_keys=True, default=str)}],
        )
        raw = "".join(block.text for block in message.content
                      if getattr(block, "type", "") == "text").strip()
        valid, reason = validate_rewrite(item, raw)
        latency = int((time.perf_counter() - started) * 1000)
        if valid:
            payload = json.loads(raw)
            return item.with_narration(str(payload["text"]).strip(), {
                "narrator": "language_model", "validated": True, "model": model,
                "latency_ms": latency, "raw": raw, "validation_reason": reason,
            })
        return item.with_narration(item.template_headline, {
            "narrator": "template", "validated": True, "model": model,
            "latency_ms": latency, "raw": raw, "fallback_kind": "rejected",
            "fallback_reason": reason,
        })
    except Exception as exc:
        return item.with_narration(item.template_headline, {
            "narrator": "template", "validated": True, "model": model,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "raw": raw, "fallback_kind": "unavailable",
            "fallback_reason": f"{type(exc).__name__}: narrator request failed",
        })
