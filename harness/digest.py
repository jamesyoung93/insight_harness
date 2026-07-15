"""Deterministic, personalized "things to look at" digest.

The scanner is registry-driven and every surfaced claim wraps a normal answer
artifact produced by :func:`pipeline.answer_intent`.  Ranking is deterministic,
cross-metric impact is normalized, history affects novelty explicitly, and a
metric-family constraint prevents three versions of the same story.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import pandas as pd

from . import pipeline, semantic_layer as sl, services, triage
from .digest_store import DigestHistoryStore, history_fingerprint
from .provenance import AnswerArtifact, TIER_ABSTAINED


DIGEST_VERSION = 2
_TYPE_WEIGHT = {"anomaly": 1.0, "watch": 1.25, "divergence": 1.15, "event": 1.1}
DIGEST_METRICS = ("trx", "nrx", "nbrx", "trx_share", "calls", "samples",
                  "speaker_attendance", "new_writers")
DIGEST_DIMENSIONS = ("region", "specialty", "payer_channel")


def _stable_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _normal_value(value):
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item) for item in value))
    return str(value)


def _normal_scope(scope: Mapping | None) -> dict[str, str | tuple[str, ...]]:
    return {str(key): _normal_value(value)
            for key, value in sorted((scope or {}).items())}


def _filter_items(filters: Mapping) -> tuple[tuple[str, str | tuple[str, ...]], ...]:
    return tuple(_normal_scope(filters).items())


def _filters_dict(items: tuple[tuple[str, str | tuple[str, ...]], ...]) -> dict:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in items}


def _metric_family(metric: str) -> str:
    definition = sl.METRICS[metric]
    return str(definition.get("family") or definition.get("metric_family") or metric)


def governance_fingerprint() -> str:
    """Fingerprint every registry/config input capable of changing a digest."""

    effective = {
        "materiality": sl.materiality(),
        "default_variants": {metric: sl.default_variant(metric)
                             for metric in sorted(sl.METRICS)},
        "metrics": sl.METRICS,
        "sources": sl.SOURCES,
        "dimensions": sl.DIMENSIONS,
        "events": sl.EVENTS,
    }
    return _stable_hash(effective)


@dataclass(frozen=True)
class MovementFacts:
    month: str
    latest: float
    trailing_mean: float
    trailing_std: float
    z: float
    absolute_change: float
    relative_change: float
    history_months: int

    def to_dict(self) -> dict:
        return {
            "month": self.month,
            "latest": self.latest,
            "trailing_mean": self.trailing_mean,
            "trailing_std": self.trailing_std,
            "z": self.z,
            "absolute_change": self.absolute_change,
            "relative_change": self.relative_change,
            "history_months": self.history_months,
        }


@dataclass(frozen=True)
class SignalCandidate:
    kind: str
    metric: str
    family: str
    filters: tuple[tuple[str, str | tuple[str, ...]], ...]
    source: str
    variant: str
    artifact: AnswerArtifact = field(compare=False, repr=False)
    facts: MovementFacts | None = None
    impact_score: float = 0.0
    scope_priority: float = 1.0
    event_id: str | None = None
    event_name: str | None = None
    fork_label: str | None = None
    fork_value: float | None = None
    fork_relative_difference: float | None = None

    @property
    def semantic_key(self) -> str:
        return _stable_hash({
            "kind": self.kind,
            "metric": self.metric,
            "family": self.family,
            "filters": self.filters,
            "source": self.source,
            "variant": self.variant,
            "event_id": self.event_id,
            "fork_label": self.fork_label,
        })

    @property
    def filter_dict(self) -> dict:
        return _filters_dict(self.filters)


@dataclass(frozen=True)
class ScanResult:
    candidates: tuple[SignalCandidate, ...]
    scanned_series: int
    metric_families: int


@dataclass(frozen=True)
class DigestItemArtifact:
    candidate: SignalCandidate = field(compare=False, repr=False)
    template_headline: str
    impact_text: str
    breakdown_question: str
    novelty_factor: float
    score: float
    narration: dict = field(default_factory=dict, compare=False)

    @property
    def result_hash(self) -> str:
        return _stable_hash({
            "signal_key": self.candidate.semantic_key,
            "answer_hash": self.candidate.artifact.result_hash,
            "facts": self.candidate.facts.to_dict() if self.candidate.facts else None,
            "fork": [self.candidate.fork_label, self.candidate.fork_value,
                     self.candidate.fork_relative_difference],
            "template_headline": self.template_headline,
            "impact_text": self.impact_text,
            "breakdown_question": self.breakdown_question,
            "novelty_factor": round(self.novelty_factor, 8),
            "score": round(self.score, 8),
        })

    @property
    def headline(self) -> str:
        return str(self.narration.get("text") or self.template_headline)

    def with_narration(self, text: str, metadata: dict) -> "DigestItemArtifact":
        return replace(self, narration={**metadata, "text": text})

    def fact_payload(self) -> dict:
        """The bounded JSON payload an optional narrator is allowed to see."""

        return {
            "signal_key": self.candidate.semantic_key,
            "kind": self.candidate.kind,
            "metric": self.candidate.metric,
            "metric_label": sl.METRICS[self.candidate.metric]["label"],
            "scope": sl.scope_string(self.candidate.filter_dict),
            "facts": self.candidate.facts.to_dict() if self.candidate.facts else None,
            "fork": {
                "label": self.candidate.fork_label,
                "value": self.candidate.fork_value,
                "relative_difference": self.candidate.fork_relative_difference,
            } if self.candidate.fork_label else None,
            "event": {"id": self.candidate.event_id, "name": self.candidate.event_name}
            if self.candidate.event_id else None,
            "templated_headline": self.template_headline,
            "impact_text": self.impact_text,
            "underlying_answer_hash": self.candidate.artifact.result_hash,
            "data_version": self.candidate.artifact.data_version,
        }

    def to_dict(self) -> dict:
        return {
            **self.fact_payload(),
            "headline": self.headline,
            "novelty_factor": self.novelty_factor,
            "normalized_impact": self.candidate.impact_score,
            "score": self.score,
            "result_hash": self.result_hash,
            "breakdown_question": self.breakdown_question,
            "narration": self.narration,
            "underlying_answer": json.loads(self.candidate.artifact.to_json()),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


@dataclass(frozen=True)
class DigestArtifact:
    digest_key: str
    persona: str
    scope: dict
    data_version: str
    governance_fingerprint: str
    input_fingerprint: str
    history_fingerprint: str
    scanned_series: int
    metric_families: int
    items: tuple[DigestItemArtifact, ...]

    @property
    def result_hash(self) -> str:
        return _stable_hash({
            "version": DIGEST_VERSION,
            "digest_key": self.digest_key,
            "persona": self.persona,
            "scope": self.scope,
            "data_version": self.data_version,
            "governance_fingerprint": self.governance_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "history_fingerprint": self.history_fingerprint,
            "scanned_series": self.scanned_series,
            "metric_families": self.metric_families,
            "items": [item.result_hash for item in self.items],
        })

    def to_dict(self) -> dict:
        return {
            "digest_key": self.digest_key,
            "persona": self.persona,
            "scope": self.scope,
            "data_version": self.data_version,
            "governance_fingerprint": self.governance_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "history_fingerprint": self.history_fingerprint,
            "scanned_series": self.scanned_series,
            "metric_families": self.metric_families,
            "items": [item.to_dict() for item in self.items],
            "result_hash": self.result_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def _scope_values(source: str, dimension: str) -> set[str]:
    df = sl.load_fact(source)
    return set(str(value) for value in df[dimension].dropna().unique()) \
        if dimension in df.columns else set()


def _validate_scope(scope: Mapping) -> dict:
    normalized = _normal_scope(scope)
    for dimension, value in normalized.items():
        if dimension not in sl.DIMENSIONS:
            raise ValueError(f"unregistered digest scope dimension: {dimension!r}")
        values = value if isinstance(value, tuple) else (value,)
        registered: set[str] = set()
        for source in sl.SOURCES:
            registered.update(_scope_values(source, dimension))
        if not values or any(item not in registered for item in values):
            raise ValueError(f"unregistered digest scope: {dimension}={value!r}")
    return normalized


def _scope_compatible(left: Mapping, right: Mapping) -> bool:
    for dimension in set(left).intersection(right):
        a = set(left[dimension] if isinstance(left[dimension], (list, tuple)) else [left[dimension]])
        b = set(right[dimension] if isinstance(right[dimension], (list, tuple)) else [right[dimension]])
        if not a.intersection(b):
            return False
    return True


def _merge_scopes(left: Mapping, right: Mapping) -> dict | None:
    if not _scope_compatible(left, right):
        return None
    out = _normal_scope(left)
    for dimension, value in _normal_scope(right).items():
        if dimension not in out:
            out[dimension] = value
            continue
        a = set(out[dimension] if isinstance(out[dimension], tuple) else [out[dimension]])
        b = set(value if isinstance(value, tuple) else [value])
        common = sorted(a.intersection(b))
        out[dimension] = common[0] if len(common) == 1 else tuple(common)
    return dict(sorted(out.items()))


def _canonical_keyword(metric: str) -> str:
    candidates = [keyword for keyword, registered in sl.METRIC_KEYWORDS.items()
                  if registered == metric]
    if candidates:
        return sorted(candidates, key=lambda value: (-len(value), value))[0]
    return sl.METRICS[metric]["label"].lower()


def _trend_question(metric: str, filters: Mapping) -> str:
    pieces = [f"Trend {_canonical_keyword(metric)} by month"]
    if filters:
        rendered = []
        for dimension, value in _normal_scope(filters).items():
            values = value if isinstance(value, tuple) else (value,)
            rendered.append(" and ".join(values) + " " + dimension.replace("_", " "))
        pieces.append("in " + " and ".join(rendered))
    return " ".join(pieces)


@lru_cache(maxsize=2048)
def _cached_answer_artifact(metric: str,
                            filter_items: tuple[tuple[str, str | tuple[str, ...]], ...],
                            data_version: str,
                            governance: str,
                            requested_source: str | None,
                            requested_variant: str | None) -> AnswerArtifact:
    del data_version, governance  # cache-key-only provenance inputs
    filters = _filters_dict(filter_items)
    intent = triage.Intent(
        question=_trend_question(metric, filters),
        question_class=triage.DESCRIPTIVE,
        metric=metric,
        filters={key: list(value) if isinstance(value, tuple) else value
                 for key, value in _normal_scope(filters).items()},
        trend=True,
    )
    return pipeline.answer_intent(intent, requested_source, requested_variant)


def _answer_artifact(metric: str, filters: Mapping,
                     requested_source: str | None = None,
                     requested_variant: str | None = None) -> AnswerArtifact:
    return _cached_answer_artifact(metric, _filter_items(filters),
                                   sl.data_version(), governance_fingerprint(),
                                   requested_source, requested_variant)


def _movement(artifact: AnswerArtifact) -> MovementFacts | None:
    chart = artifact.chart_df
    if artifact.tier == TIER_ABSTAINED or chart is None or chart.empty:
        return None
    columns = [column for column in chart.columns if column != "month"]
    if not columns:
        return None
    series = pd.to_numeric(chart[columns[0]], errors="coerce")
    valid = pd.DataFrame({"month": chart["month"].astype(str), "value": series}).dropna()
    if len(valid) < 5:
        return None
    history = valid["value"].iloc[-7:-1]
    latest = float(valid["value"].iloc[-1])
    mean = float(history.mean())
    std = float(history.std()) if len(history) > 1 else 0.0
    std = std if pd.notna(std) else 0.0
    absolute = latest - mean
    relative = absolute / abs(mean) if mean else 0.0
    z = absolute / std if std else 0.0
    return MovementFacts(
        month=str(valid["month"].iloc[-1]), latest=latest, trailing_mean=mean,
        trailing_std=std, z=float(z), absolute_change=absolute,
        relative_change=float(relative), history_months=len(history),
    )


def normalized_impact(facts: MovementFacts) -> float:
    """Comparable [0, 1] score blending standardized and relative movement."""

    z_component = min(abs(facts.z) / 4.0, 1.0)
    relative_component = min(abs(facts.relative_change) / 0.50, 1.0)
    return round(0.65 * z_component + 0.35 * relative_component, 8)


def _series_specs(metric: str, persona_scope: Mapping) -> list[tuple[dict, float]]:
    source = sl.METRICS[metric]["default_source"]
    df = sl.load_fact(source)
    scoped = sl.apply_filters(df, dict(persona_scope))
    specs: list[tuple[dict, float]] = []
    if persona_scope:
        specs.extend([(dict(persona_scope), 1.0), ({}, 0.60)])
    else:
        specs.append(({}, 1.0))
    for dimension in DIGEST_DIMENSIONS:
        if dimension in persona_scope or dimension not in scoped.columns:
            continue
        for value in sorted(str(item) for item in scoped[dimension].dropna().unique()):
            filters = dict(persona_scope)
            filters[dimension] = value
            specs.append((filters, 1.0))
    deduped: dict[tuple, tuple[dict, float]] = {}
    for filters, priority in specs:
        key = _filter_items(filters)
        if key not in deduped or priority > deduped[key][1]:
            deduped[key] = (filters, priority)
    return [deduped[key] for key in sorted(deduped, key=str)]


def scan_candidates(*, persona: str, scope: Mapping | None = None,
                    watches: Sequence[dict] = ()) -> ScanResult:
    """Scan movement, watch, divergence, and registered-event signals."""

    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("persona must be a non-empty string")
    persona_scope = _validate_scope(scope or {})
    candidates: list[SignalCandidate] = []
    artifacts: dict[tuple, tuple[AnswerArtifact, MovementFacts | None, float]] = {}
    active_metrics = tuple(metric for metric in DIGEST_METRICS if metric in sl.METRICS)
    families = {_metric_family(metric) for metric in active_metrics}

    def evaluate(metric: str, filters: Mapping, priority: float,
                 requested_source: str | None = None,
                 requested_variant: str | None = None) -> tuple[AnswerArtifact, MovementFacts | None]:
        key = (metric, _filter_items(filters), requested_source, requested_variant)
        if key not in artifacts:
            artifact = _answer_artifact(metric, filters, requested_source, requested_variant)
            artifacts[key] = (artifact, _movement(artifact), priority)
        else:
            artifact, facts, old_priority = artifacts[key]
            if priority > old_priority:
                artifacts[key] = (artifact, facts, priority)
        return artifacts[key][0], artifacts[key][1]

    def add_movement(kind: str, metric: str, filters: Mapping, priority: float,
                     *, event_id: str | None = None, event_name: str | None = None,
                     requested_source: str | None = None,
                     requested_variant: str | None = None) -> None:
        artifact, facts = evaluate(metric, filters, priority,
                                   requested_source, requested_variant)
        if facts is None or abs(facts.absolute_change) < 1e-12:
            return
        resolution = artifact.resolution
        candidates.append(SignalCandidate(
            kind=kind, metric=metric, family=_metric_family(metric),
            filters=_filter_items(filters), source=resolution.source,
            variant=resolution.variant, artifact=artifact, facts=facts,
            impact_score=normalized_impact(facts), scope_priority=priority,
            event_id=event_id, event_name=event_name,
        ))

    for metric in active_metrics:
        for filters, priority in _series_specs(metric, persona_scope):
            add_movement("anomaly", metric, filters, priority)

    for watch in watches:
        metric = watch.get("metric")
        if metric not in sl.METRICS:
            continue
        filters = _validate_scope(watch.get("filters") or {})
        priority = 1.0 if _scope_compatible(filters, persona_scope) else 0.45
        add_movement("watch", metric, filters, priority,
                     requested_source=watch.get("source"),
                     requested_variant=watch.get("variant"))

    for event_id, event in sorted(sl.EVENTS.items()):
        merged = _merge_scopes(persona_scope, event.get("scope", {}))
        if merged is None:
            continue
        for metric in sorted(event.get("metrics", [])):
            if metric not in active_metrics:
                continue
            latest = sl.months(sl.METRICS[metric]["default_source"])[-1]
            if str(event.get("start", "")) <= latest:
                add_movement("event", metric, merged, 1.05,
                             event_id=event_id, event_name=event.get("name", event_id))

    # Material forks become their own candidates.  Each unique evaluated
    # metric/scope is inspected exactly once, even if both anomaly and watch
    # signals point to it.
    for (metric, filter_items, _requested_source, _requested_variant), \
            (artifact, facts, priority) in sorted(
            artifacts.items(), key=lambda item: str(item[0])):
        for fork in artifact.divergence:
            if not fork.get("material"):
                continue
            relative = float(fork.get("rel_diff") or 0.0)
            impact = min(abs(relative) / max(sl.materiality() * 5.0, 0.05), 1.0)
            candidates.append(SignalCandidate(
                kind="divergence", metric=metric, family=_metric_family(metric),
                filters=filter_items, source=artifact.resolution.source,
                variant=artifact.resolution.variant, artifact=artifact, facts=facts,
                impact_score=round(impact, 8), scope_priority=priority,
                fork_label=str(fork.get("label") or fork.get("fork")),
                fork_value=float(fork["value"]),
                fork_relative_difference=relative,
            ))

    unique = {candidate.semantic_key: candidate for candidate in candidates}
    ordered = tuple(unique[key] for key in sorted(unique))
    return ScanResult(ordered, scanned_series=len(artifacts), metric_families=len(families))


def novelty_factor(candidate: SignalCandidate, recent_keys: Iterable[str]) -> float:
    count = sum(1 for key in recent_keys if key == candidate.semantic_key)
    if count == 0:
        return 1.0
    return max(0.25, round(0.55 ** count, 8))


def _candidate_score(candidate: SignalCandidate, novelty: float) -> float:
    scope_weight = 0.70 + 0.30 * max(0.0, min(candidate.scope_priority, 1.10))
    return round(candidate.impact_score * novelty
                 * _TYPE_WEIGHT.get(candidate.kind, 1.0) * scope_weight, 8)


def rank_candidates(candidates: Sequence[SignalCandidate], recent_keys: Iterable[str] = (),
                    limit: int = 3) -> list[tuple[SignalCandidate, float, float]]:
    """Rank deterministically and admit at most one item per metric family."""

    recent = tuple(recent_keys)
    scored = [
        (candidate, novelty_factor(candidate, recent),
         _candidate_score(candidate, novelty_factor(candidate, recent)))
        for candidate in candidates
    ]
    scored.sort(key=lambda row: (-row[2], row[0].semantic_key))
    selected: list[tuple[SignalCandidate, float, float]] = []
    families: set[str] = set()
    for row in scored:
        if row[0].family in families:
            continue
        selected.append(row)
        families.add(row[0].family)
        if len(selected) >= max(0, limit):
            break
    return selected


def _scope_label(filters: Mapping) -> str:
    label = sl.scope_string(dict(filters))
    return "All scopes" if label == "all scopes" else label


def _headline(candidate: SignalCandidate) -> tuple[str, str]:
    label = sl.METRICS[candidate.metric]["label"]
    scope = _scope_label(candidate.filter_dict)
    if candidate.kind == "divergence":
        rel = float(candidate.fork_relative_difference or 0.0) * 100
        governed = float(candidate.artifact.value or 0.0)
        headline = (f"{scope} {label} differs {rel:+.1f}% under {candidate.fork_label}; "
                    f"governed {governed:,.1f} vs alternate {candidate.fork_value:,.1f}.")
        impact = f"Material definition fork · {abs(rel):.1f}% relative difference"
        return headline, impact
    facts = candidate.facts
    if facts is None:
        return f"{scope} {label} needs review.", "Computed signal"
    direction = "above" if facts.absolute_change >= 0 else "below"
    movement = (f"{abs(facts.z):.1f}σ {direction} its six-month trailing mean"
                if facts.trailing_std else f"{facts.relative_change * 100:+.1f}% vs its trailing mean")
    prefix = "Watched: " if candidate.kind == "watch" else ""
    if candidate.kind == "event":
        prefix = f"A registered event ({candidate.event_name}) overlaps this series: "
    headline = (f"{prefix}{scope} {label} is {movement}; "
                f"{facts.relative_change * 100:+.1f}% vs that mean.")
    impact = (f"Latest {facts.latest:,.1f} · trailing mean {facts.trailing_mean:,.1f} · "
              f"absolute movement {facts.absolute_change:+,.1f}")
    return headline, impact


def _make_item(candidate: SignalCandidate, novelty: float, score: float) -> DigestItemArtifact:
    headline, impact = _headline(candidate)
    question = services.breakdown_question(candidate.metric, candidate.filter_dict)
    return DigestItemArtifact(
        candidate=candidate, template_headline=headline, impact_text=impact,
        breakdown_question=question, novelty_factor=novelty, score=score,
    )


def _watch_fingerprint(watches: Sequence[dict]) -> str:
    stable = [{
        "metric": watch.get("metric"),
        "filters": _normal_scope(watch.get("filters") or {}),
        "source": watch.get("source"),
        "variant": watch.get("variant"),
        "window": watch.get("window") or watch.get("window_spec"),
        "compare_basis": watch.get("compare_basis") or watch.get("basis"),
        "trend": watch.get("trend"),
    } for watch in watches]
    return _stable_hash(sorted(stable, key=lambda item: json.dumps(item, sort_keys=True)))


def build_digest(*, persona: str, scope: Mapping | None = None,
                 watches: Sequence[dict] | None = None,
                 store: DigestHistoryStore | None = None, limit: int = 3,
                 record: bool = True) -> DigestArtifact:
    """Build one stable snapshot and record it at most once.

    A previously recorded snapshot wins for the same input key, preventing a
    Streamlit rerun from rotating the digest merely because it was viewed.
    """

    if watches is None:
        watches = services.load_watchlist()
    watches = tuple(watches)
    normalized_scope = _validate_scope(scope or {})
    store = store or DigestHistoryStore()
    data_version = sl.data_version()
    governance = governance_fingerprint()
    input_fingerprint = _watch_fingerprint(watches)
    digest_key = _stable_hash({
        "version": DIGEST_VERSION,
        "data_version": data_version,
        "governance": governance,
        "persona": persona,
        "scope": normalized_scope,
        "inputs": input_fingerprint,
        "limit": int(limit),
    })

    scan = scan_candidates(persona=persona, scope=normalized_scope, watches=watches)
    by_key = {candidate.semantic_key: candidate for candidate in scan.candidates}
    existing = store.get(digest_key)

    if existing is not None:
        chosen = [by_key[key] for key in existing.get("item_keys", []) if key in by_key]
        history_hash = str(existing.get("history_fingerprint") or _stable_hash([]))
        recent_keys: tuple[str, ...] = tuple(existing.get("recent_item_keys", []))
        ranked_lookup = {candidate.semantic_key: (candidate, novelty, score)
                         for candidate, novelty, score in
                         rank_candidates(scan.candidates, recent_keys, limit=max(limit, len(chosen)))}
        selected = [ranked_lookup.get(candidate.semantic_key,
                                      (candidate, novelty_factor(candidate, recent_keys),
                                       _candidate_score(candidate, novelty_factor(candidate, recent_keys))))
                    for candidate in chosen]
    else:
        recent = store.recent(persona=persona, exclude_key=digest_key, limit=5)
        recent_keys = tuple(key for entry in recent for key in entry.get("item_keys", []))
        history_hash = history_fingerprint(recent)
        selected = rank_candidates(scan.candidates, recent_keys, limit=limit)

    items = tuple(_make_item(candidate, novelty, score)
                  for candidate, novelty, score in selected[:max(0, limit)])
    artifact = DigestArtifact(
        digest_key=digest_key, persona=persona, scope=normalized_scope,
        data_version=data_version, governance_fingerprint=governance,
        input_fingerprint=input_fingerprint, history_fingerprint=history_hash,
        scanned_series=scan.scanned_series, metric_families=scan.metric_families,
        items=items,
    )

    if record and existing is None:
        history_record = {
            "digest_key": digest_key,
            "persona": persona,
            "scope": normalized_scope,
            "data_version": data_version,
            "governance_fingerprint": governance,
            "input_fingerprint": input_fingerprint,
            "history_fingerprint": history_hash,
            "recent_item_keys": list(recent_keys),
            "item_keys": [item.candidate.semantic_key for item in items],
            "result_hash": artifact.result_hash,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        stored = store.record_once(history_record)
        if stored.get("item_keys") != history_record["item_keys"]:
            # A concurrent rerun recorded first.  Re-enter through the stable
            # snapshot path so both callers return the same selection.
            return build_digest(persona=persona, scope=normalized_scope,
                                watches=watches, store=store, limit=limit, record=False)
    return artifact
