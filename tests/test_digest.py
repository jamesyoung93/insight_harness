"""Phase 4 contract: deterministic, diverse, artifact-grounded digests."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from harness import digest
from harness import digest_narrator
from harness import semantic_layer as sl
from harness.digest_store import DigestHistoryStore, history_fingerprint


def _build(tmp_path, *, persona="Executive", scope=None, watches=(), record=False):
    return digest.build_digest(
        persona=persona, scope=scope or {}, watches=list(watches),
        store=DigestHistoryStore(tmp_path / "digest_history.jsonl"), record=record,
    )


def test_digest_is_normalized_diverse_and_artifact_grounded(tmp_path):
    artifact = _build(tmp_path)
    assert 0 < len(artifact.items) <= 3
    assert len({item.candidate.family for item in artifact.items}) == len(artifact.items)
    assert artifact.scanned_series >= len(artifact.items)
    assert artifact.metric_families >= len(artifact.items)
    for item in artifact.items:
        assert 0.0 <= item.candidate.impact_score <= 1.0
        assert item.candidate.artifact.data_version == artifact.data_version
        payload = json.loads(item.to_json())
        assert payload["underlying_answer_hash"] == item.candidate.artifact.result_hash
        assert payload["underlying_answer"]["result_hash"] == item.candidate.artifact.result_hash
        assert payload["templated_headline"] == item.template_headline


def test_digest_history_is_idempotent_and_does_not_rotate_on_rerun(tmp_path):
    store = DigestHistoryStore(tmp_path / "digest_history.jsonl")
    first = digest.build_digest(persona="Executive", scope={}, watches=[], store=store)
    second = digest.build_digest(persona="Executive", scope={}, watches=[], store=store)
    assert first.digest_key == second.digest_key
    assert first.result_hash == second.result_hash
    assert [item.candidate.semantic_key for item in first.items] == [
        item.candidate.semantic_key for item in second.items]
    assert len(store.load()) == 1


def test_persona_scope_is_ranked_ahead_of_national_equivalent(tmp_path):
    source = next(iter(sl.SOURCES))
    fact = sl.load_fact(source)
    dimension = "region" if "region" in sl.DIMENSIONS else sl.DIMENSIONS[0]
    value = str(sorted(fact[dimension].dropna().unique())[0])
    artifact = _build(tmp_path, persona="District Manager", scope={dimension: value})
    assert artifact.scope == {dimension: value}
    assert any(item.candidate.filter_dict.get(dimension) == value for item in artifact.items)


def test_novelty_can_promote_a_new_signal_within_the_same_family(tmp_path):
    scan = digest.scan_candidates(persona="Executive", scope={}, watches=[])
    grouped = {}
    for candidate in scan.candidates:
        grouped.setdefault(candidate.family, []).append(candidate)
    family_candidates = next(values for values in grouped.values() if len(values) >= 2)
    first, second = sorted(family_candidates, key=lambda value: value.semantic_key)[:2]
    first = replace(first, impact_score=1.0, scope_priority=1.0)
    second = replace(second, impact_score=1.0, scope_priority=1.0)
    initial = digest.rank_candidates([first, second], limit=1)[0][0]
    repeated = digest.rank_candidates(
        [first, second], recent_keys=[initial.semantic_key], limit=1)[0][0]
    assert repeated.semantic_key != initial.semantic_key


def test_store_skips_corrupt_lines_and_fingerprints_selection_only(tmp_path):
    path = tmp_path / "digest_history.jsonl"
    valid = {"digest_key": "one", "persona": "Executive", "scope": {},
             "data_version": "v1", "governance_fingerprint": "g1",
             "item_keys": ["a", "b"], "created_at": "first"}
    path.write_text(json.dumps(valid) + "\n" + '{"digest_key":' + "\n", encoding="utf-8")
    store = DigestHistoryStore(path)
    assert store.load() == [valid]
    changed_time = valid | {"created_at": "later"}
    assert history_fingerprint([valid]) == history_fingerprint([changed_time])
    assert store.record_once(valid | {"created_at": "ignored"}) == valid
    assert len(store.load()) == 1


def test_narrator_rejects_an_invented_number_and_falls_back(tmp_path, monkeypatch):
    item = _build(tmp_path).items[0]
    assert digest_narrator.validate_rewrite(item, item.template_headline)[0]
    assert not digest_narrator.validate_rewrite(
        item, item.template_headline + " Forecast 999.")[0]

    response = SimpleNamespace(content=[SimpleNamespace(
        type="text", text=item.template_headline + " Forecast 999.")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    fake_module = SimpleNamespace(Anthropic=lambda **_: client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    rendered = digest_narrator.rewrite_item(item, api_key="test-key")
    assert rendered.headline == item.template_headline
    assert rendered.narration["fallback_kind"] == "rejected"
    assert rendered.narration["narrator"] == "template"


def test_digest_view_renders_three_artifact_cards_without_a_model_key(tmp_path):
    history = (tmp_path / "digest_ui_history.jsonl").as_posix()
    script = f'''from harness.digest_store import DigestHistoryStore
from views import digest
digest.render(store=DigestHistoryStore(r"{history}"))'''
    at = AppTest.from_string(script, default_timeout=180).run()
    assert not at.exception
    assert [title.value for title in at.title] == ["Daily digest"]
    assert len(at.get("download_button")) == 3
    rendered = " ".join(caption.value for caption in at.caption)
    assert "Scanned" in rendered and "monthly data" in rendered
