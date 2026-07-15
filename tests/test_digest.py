"""Phase 4 contract: isolated, deterministic, diverse, grounded digests."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from harness import digest, digest_narrator, semantic_layer as sl
from harness.digest_store import (DigestHistoryStore, InMemoryDigestHistoryStore,
                                  history_fingerprint)
from harness import triage
from views import digest as digest_view


REPO = Path(__file__).parent.parent
APP = str(REPO / "app.py")


def _build(tmp_path=None, *, persona="Executive", scope=None, watches=(),
           record=False, store=None, owner_namespace=None):
    if store is None:
        store = (DigestHistoryStore(tmp_path / "digest_history.jsonl") if tmp_path
                 else InMemoryDigestHistoryStore())
    return digest.build_digest(
        persona=persona, scope=scope or {}, watches=list(watches), store=store,
        record=record, owner_namespace=owner_namespace)


def _movement_item():
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[])
    candidate = next(item for item in scan.candidates
                     if item.kind == "anomaly" and item.facts is not None
                     and item.facts.trailing_std)
    return digest._make_item(candidate, 1.0, 1.0)


def _event_item():
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[])
    candidate = next(item for item in scan.candidates if item.kind == "event")
    return digest._make_item(candidate, 1.0, 1.0)


def _response(item, *, text=None, metric=None, scope=None, direction=None,
              causal=False):
    return {
        "text": text or item.template_headline,
        "metric": metric or item.candidate.metric,
        "scope": scope or item.fact_payload()["scope"],
        "direction": direction or digest_narrator.expected_direction(item),
        "causal": causal,
    }


def test_digest_is_normalized_semantically_diverse_and_artifact_grounded():
    artifact = _build()
    assert 0 < len(artifact.items) <= 3
    families = [item.candidate.family for item in artifact.items]
    assert len(set(families)) == len(families)
    assert sum(item.candidate.metric in {"trx", "nrx", "nbrx"}
               for item in artifact.items) <= 1
    for item in artifact.items:
        assert 0.0 <= item.candidate.impact_score <= 1.0
        payload = json.loads(item.to_json())
        assert payload["underlying_answer_hash"] == item.candidate.artifact.result_hash
        assert payload["underlying_answer"]["result_hash"] == \
            item.candidate.artifact.result_hash
        assert payload["fact_hash"] == item.fact_hash == item.result_hash
        assert payload["presentation_hash"] == item.presentation_hash


def test_scanner_explicitly_covers_all_four_signal_paths():
    watch = {"metric": "trx", "filters": {}, "source": "source_b",
             "variant": "normalized", "window": "R6M", "basis": "YoY"}
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[watch])
    kinds = {candidate.kind for candidate in scan.candidates}
    assert {"anomaly", "watch", "divergence", "event"} <= kinds
    assert all(candidate.artifact.result_hash for candidate in scan.candidates)


def test_non_descriptive_watches_are_excluded_instead_of_reinterpreted():
    unsupported = [
        {"metric": "trx", "filters": {}, "source": "source_b",
         "variant": "normalized", "window": "R6M", "basis": "YoY",
         "question_class": triage.DIAGNOSTIC},
        {"metric": "trx", "filters": {}, "source": "source_b",
         "variant": "normalized", "window": "R6M", "basis": "YoY",
         "question_class": triage.RETRIEVAL},
    ]
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=unsupported)
    assert not [candidate for candidate in scan.candidates if candidate.kind == "watch"]
    assert _build(watches=unsupported).input_fingerprint == _build().input_fingerprint

    common = {"watched": True, "is_stale": False, "metric": "trx", "filters": {},
              "source": "source_b", "variant": "normalized", "window": "R6M",
              "basis": "YoY"}
    inputs, skipped = digest_view._descriptive_watch_inputs([
        SimpleNamespace(**common, question_class=triage.DESCRIPTIVE),
        SimpleNamespace(**common, question_class=triage.DIAGNOSTIC),
        SimpleNamespace(**common, question_class=triage.RETRIEVAL),
    ])
    assert len(inputs) == 1
    assert inputs[0]["question_class"] == triage.DESCRIPTIVE
    assert skipped == 2


def test_persona_scope_is_a_strict_lexicographic_tier():
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[])
    first = scan.candidates[0]
    second = next(item for item in scan.candidates if item.family != first.family)
    scoped = replace(first, scope_rank=2, impact_score=0.0001, scope_priority=0.0)
    national = replace(second, scope_rank=1, impact_score=1.0, scope_priority=1.0)
    ranked = digest.rank_candidates([national, scoped], limit=2)
    assert ranked[0][0] == scoped


def test_mapping_scope_is_preserved_and_ranked_ahead_of_national():
    fact = sl.load_fact(next(iter(sl.SOURCES)))
    dimension = "territory" if "territory" in sl.DIMENSIONS else sl.DIMENSIONS[0]
    value = str(sorted(fact[dimension].dropna().unique())[0])
    artifact = _build(persona="Sales Rep", scope={dimension: value})
    assert artifact.scope == {dimension: value}
    assert artifact.items
    assert all(item.candidate.scope_rank == 2 for item in artifact.items)


def test_digest_history_is_idempotent_and_does_not_rotate_on_rerun(tmp_path):
    store = DigestHistoryStore(tmp_path / "digest_history.jsonl")
    first = _build(tmp_path, store=store, record=True, owner_namespace="viewer-a")
    second = _build(tmp_path, store=store, record=True, owner_namespace="viewer-a")
    assert first.digest_key == second.digest_key
    assert first.result_hash == second.result_hash
    assert [item.candidate.semantic_key for item in first.items] == [
        item.candidate.semantic_key for item in second.items]
    assert len(store.load()) == 1


def test_history_isolated_by_owner_scope_and_watch_fingerprint(tmp_path):
    store = DigestHistoryStore(tmp_path / "digest_history.jsonl")
    base = {"persona": "Executive", "data_version": "v1",
            "governance_fingerprint": "g1", "item_keys": ["a"]}
    records = [
        base | {"digest_key": "match", "owner_namespace": "user-a",
                "scope": {"region": "East"}, "input_fingerprint": "watch-a"},
        base | {"digest_key": "other-owner", "owner_namespace": "user-b",
                "scope": {"region": "East"}, "input_fingerprint": "watch-a"},
        base | {"digest_key": "other-scope", "owner_namespace": "user-a",
                "scope": {"region": "West"}, "input_fingerprint": "watch-a"},
        base | {"digest_key": "other-watch", "owner_namespace": "user-a",
                "scope": {"region": "East"}, "input_fingerprint": "watch-b"},
    ]
    for record in records:
        store.record_once(record)
    recent = store.recent(
        persona="Executive", scope={"region": "East"},
        input_fingerprint="watch-a", owner_namespace="user-a")
    assert [record["digest_key"] for record in recent] == ["match"]

    first, second = InMemoryDigestHistoryStore(), InMemoryDigestHistoryStore()
    first.record_once(records[0])
    assert len(first.load()) == 1
    assert second.load() == []


def test_record_once_is_concurrency_safe(tmp_path):
    path = tmp_path / "digest_history.jsonl"
    record = {"digest_key": "one", "persona": "Executive", "scope": {},
              "input_fingerprint": "w", "owner_namespace": "u", "item_keys": ["a"]}

    def write_once(_):
        return DigestHistoryStore(path).record_once(record)

    with ThreadPoolExecutor(max_workers=8) as pool:
        stored = list(pool.map(write_once, range(24)))
    assert all(item == record for item in stored)
    assert DigestHistoryStore(path).load() == [record]


def test_store_skips_corrupt_lines_and_fingerprint_ignores_timestamp(tmp_path):
    path = tmp_path / "digest_history.jsonl"
    valid = {"digest_key": "one", "owner_namespace": "u", "persona": "Executive",
             "scope": {}, "input_fingerprint": "w", "data_version": "v1",
             "governance_fingerprint": "g1", "item_keys": ["a", "b"],
             "created_at": "first"}
    path.write_text(json.dumps(valid) + "\n" + '{"digest_key":' + "\n", encoding="utf-8")
    store = DigestHistoryStore(path)
    assert store.load() == [valid]
    assert history_fingerprint([valid]) == \
        history_fingerprint([valid | {"created_at": "later"}])


def test_watch_execution_context_survives_into_item_and_breakdown(monkeypatch):
    watch = {"metric": "trx", "filters": {"region": "West"},
             "source": "source_b", "variant": "normalized",
             "window": "R6M", "basis": "YoY"}
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[watch])
    candidate = next(item for item in scan.candidates if item.kind == "watch")
    item = digest._make_item(candidate, 1.0, 1.0)
    payload = item.fact_payload()
    context = payload["drillthrough_context"]
    assert context == {"source": "source_b", "variant": "normalized",
                       "window": "R6M", "basis": "yoy"}
    assert payload["ranking_method"] == {
        "id": digest.MOVEMENT_RANKING_METHOD,
        "description": "Latest observed month compared with the mean of the preceding six months.",
        "latest_periods": 1,
        "baseline_periods": 6,
        "uses_drillthrough_window": False,
        "uses_drillthrough_basis": False,
    }
    assert "last 6 months" in item.breakdown_question
    assert "same month last year" in item.breakdown_question

    queued = {}
    monkeypatch.setattr(digest_view, "st", SimpleNamespace(session_state={}))
    monkeypatch.setattr(
        digest_view.common, "queue_question_with_resolution",
        lambda question, source, variant, basis: queued.update(
            question=question, source=source, variant=variant, basis=basis))
    digest_view._queue_breakdown(item)
    assert digest_view.st.session_state["_digest_drillthrough_context"] == context
    assert queued == {"question": item.breakdown_question, "source": "source_b",
                      "variant": "normalized", "basis": "yoy"}

    alternate_context = replace(candidate, window="R12M", basis="prior_month")
    assert alternate_context.semantic_key == candidate.semantic_key
    alternate_item = digest._make_item(alternate_context, 1.0, 1.0)
    assert alternate_item.fact_hash == item.fact_hash
    assert alternate_item.presentation_hash != item.presentation_hash


def test_fact_and_presentation_hashes_have_distinct_contracts():
    artifact = _build()
    item = artifact.items[0]
    narrated = item.with_narration("Validated alternate wording", {
        "narrator": "language_model", "validated": True, "model": "model-a"})
    assert narrated.fact_hash == item.fact_hash
    assert narrated.presentation_hash != item.presentation_hash
    presented = replace(artifact, items=(narrated, *artifact.items[1:]))
    assert presented.fact_hash == artifact.fact_hash
    assert presented.presentation_hash != artifact.presentation_hash
    assert json.loads(presented.to_json())["presentation_hash"] == presented.presentation_hash

    copy_mutations = (
        replace(item, template_headline=item.template_headline + " Copy edit."),
        replace(item, impact_text=item.impact_text + " Copy edit."),
        replace(item, breakdown_question=item.breakdown_question + " Copy edit?"),
    )
    for mutated in copy_mutations:
        assert mutated.fact_hash == item.fact_hash
        assert mutated.presentation_hash != item.presentation_hash


def test_structured_narrator_accepts_only_a_fully_grounded_response():
    item = _movement_item()
    valid, reason = digest_narrator.validate_rewrite(item, _response(item))
    assert valid, reason


def test_narrator_rejects_polarity_unit_scope_metric_and_causal_attacks():
    item = _movement_item()
    direction = digest_narrator.expected_direction(item)
    opposite = "below" if direction == "above" else "above"
    polarity_text = item.template_headline.replace(direction, opposite, 1)
    attacks = [
        _response(item, text=polarity_text),
        _response(item, text=item.template_headline.replace("%", "$", 1)),
        _response(item, scope="region=Elsewhere"),
        _response(item, metric="not_the_metric"),
        _response(item, text=item.template_headline + " It was caused by the event."),
        _response(item, text=item.template_headline + " Forecast 999."),
    ]
    attacks.extend(
        _response(item, text=item.template_headline + phrase)
        for phrase in (
            " The program boosted the movement.",
            " The launch triggered the movement.",
            " The campaign generated the movement.",
            " The event contributed to the movement.",
            " This happened as a result of the program.",
        )
    )
    for attack in attacks:
        assert not digest_narrator.validate_rewrite(item, attack)[0]


def test_registered_event_rewrites_are_policy_rejected_without_using_ui_quota(monkeypatch):
    item = _event_item()
    valid, reason = digest_narrator.validate_rewrite(item, _response(item))
    assert not valid
    assert "registered-event" in reason

    invoked = []
    client = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **_: invoked.append(True)))
    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=lambda **_: client))
    direct = digest_narrator.rewrite_item(item, api_key="test-key", model="model-a")
    assert direct.narration["fallback_kind"] == "policy"
    assert not invoked

    state = {"_model_calls_used": 7}
    monkeypatch.setattr(digest_view, "st", SimpleNamespace(session_state=state))
    rendered = digest_view._narrated(item, "test-key", "model-a")
    assert rendered.narration["fallback_kind"] == "policy"
    assert state["_model_calls_used"] == 7
    assert not invoked


def test_narrator_success_and_rejection_fallback(tmp_path, monkeypatch):
    item = _movement_item()
    response = SimpleNamespace(content=[SimpleNamespace(
        type="text", text=json.dumps(_response(item)))])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=lambda **_: client))
    rendered = digest_narrator.rewrite_item(item, api_key="test-key", model="model-a")
    assert rendered.narration["narrator"] == "language_model"
    assert rendered.fact_hash == item.fact_hash
    assert rendered.presentation_hash != item.presentation_hash

    bad = SimpleNamespace(content=[SimpleNamespace(
        type="text", text=json.dumps(_response(
            item, text=item.template_headline + " caused by launch")))])
    client.messages.create = lambda **_: bad
    fallback = digest_narrator.rewrite_item(item, api_key="test-key", model="model-a")
    assert fallback.headline == item.template_headline
    assert fallback.narration["fallback_kind"] == "rejected"


def test_narrator_timeout_or_error_falls_back_without_exposing_exception(monkeypatch):
    item = _movement_item()

    def fail(**_):
        raise TimeoutError("secret provider diagnostic")

    client = SimpleNamespace(messages=SimpleNamespace(create=fail))
    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=lambda **_: client))
    fallback = digest_narrator.rewrite_item(item, api_key="test-key")
    assert fallback.headline == item.template_headline
    assert fallback.narration["fallback_kind"] == "unavailable"
    assert "secret provider diagnostic" not in fallback.narration["fallback_reason"]


def test_router_digest_view_uses_session_history_drills_through_and_downloads():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="nav").set_value("Digest").run()
    assert not at.exception
    assert isinstance(at.session_state["_digest_session_history_store"],
                      InMemoryDigestHistoryStore)
    labels = [button.label for button in at.get("download_button")]
    assert labels.count("Download artifact") == 3
    assert "Download complete digest" in labels
    next(button for button in at.button if button.label == "Break this down").click().run()
    assert at.session_state["nav"] == "Home"
    assert at.session_state["ask_src"] in sl.SOURCES
    assert at.session_state["ask_var"] != "governed default"
    assert at.session_state["ask_q"]
