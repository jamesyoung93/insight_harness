"""Optional artifact-only language-model phrasing for digest items.

The narrator cannot rank or compute.  It receives a bounded fact payload and
its output is accepted only when every numeric claim matches the deterministic
template and no new number appears; otherwise the template is returned.
"""
from __future__ import annotations

import json
import os
import re
import time

from .digest import DigestItemArtifact
from .llm_translator import DEFAULT_MODEL


_NUMBER = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?")

SYSTEM = """Rewrite the supplied templated digest headline into one concise sentence.
Use only facts in the JSON. Preserve every number from templated_headline exactly in
meaning, add no number, do not make a causal claim, and return plain text only."""


def numeric_values(text: str) -> tuple[float, ...]:
    values = []
    for token in _NUMBER.findall(text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return tuple(values)


def _same_number(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-8, abs(left) * 1e-7, abs(right) * 1e-7)


def validate_rewrite(item: DigestItemArtifact, text: str) -> tuple[bool, str]:
    """Reject omitted or invented numeric claims and empty/multiline prose."""

    cleaned = text.strip()
    if not cleaned or "\n" in cleaned:
        return False, "the rewrite must be one non-empty line"
    required = numeric_values(item.template_headline)
    proposed = numeric_values(cleaned)
    for value in required:
        if not any(_same_number(value, candidate) for candidate in proposed):
            return False, f"the rewrite omitted the computed value {value}"
    for value in proposed:
        if not any(_same_number(value, candidate) for candidate in required):
            return False, f"the rewrite introduced the ungrounded value {value}"
    return True, "validated against the templated numeric facts"


def rewrite_item(item: DigestItemArtifact, api_key: str | None = None,
                 model: str = DEFAULT_MODEL) -> DigestItemArtifact:
    """Return a validated rewrite or the original deterministic template."""

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
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=model, max_tokens=180, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(
                item.fact_payload(), sort_keys=True, default=str)}],
        )
        raw = "".join(block.text for block in message.content
                      if getattr(block, "type", "") == "text").strip()
        valid, reason = validate_rewrite(item, raw)
        latency = int((time.perf_counter() - started) * 1000)
        if valid:
            return item.with_narration(raw, {
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
            "fallback_reason": str(exc),
        })

