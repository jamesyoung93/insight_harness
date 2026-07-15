"""Versioned migration, identity, session isolation, and saved evaluation."""
from dataclasses import replace

import pytest

from harness import pipeline, saved_insights, tile_runtime


def test_legacy_watch_migrates_with_declarative_defaults_and_round_trips():
    legacy = {
        "metric": "revenue",
        "filters": {"region": "West"},
        "label": "Revenue · region=West",
        "added": "2026-07-15T10:00:00+00:00",
    }
    saved = saved_insights.migrate_legacy_watch(legacy)
    assert saved.schema_version == saved_insights.SCHEMA_VERSION
    assert saved.spec.metric == "revenue"
    assert saved.spec.filters == (("region", "West"),)
    assert saved.spec.source is None
    assert saved.spec.variant is None
    assert saved.spec.window == "Latest"
    assert saved.spec.basis == "MoM"
    assert saved.spec.viz_kind == "sparkline"
    assert saved.watched is True
    assert saved.added_at == legacy["added"]
    assert not saved.is_stale

    restored = saved_insights.saved_insight_from_record(saved.to_record())
    assert restored == saved


def test_extended_legacy_watch_maps_codes_without_freezing_resolved_months():
    saved = saved_insights.migrate_legacy_watch({
        "metric": "revenue",
        "filters": {},
        "window": {"kind": "last_n", "n": 6},
        "compare_basis": "yoy",
        "source": "source_b",
        "variant": "gross",
        "viz": "line",
    })
    assert saved.spec.window == "R6M"
    assert saved.spec.basis == "YoY"
    assert saved.spec.source == "source_b"
    assert saved.spec.variant == "gross"
    assert saved.spec.viz_kind == "line"
    assert not saved.is_stale


def test_store_dedupes_complete_identity_not_just_metric_and_filters():
    store = saved_insights.InMemorySavedInsightStore()
    base = saved_insights.create_saved_insight(
        "revenue", {"region": "West"}, label="First")
    assert store.add(base).added

    relabeled = saved_insights.create_saved_insight(
        "revenue", {"region": "West"}, label="Different label")
    duplicate = store.add(relabeled)
    assert not duplicate.added
    assert duplicate.insight.id == base.id

    distinct = [
        replace(base, id="source", spec=replace(base.spec, source="source_b")),
        replace(base, id="variant", spec=replace(base.spec, variant="gross")),
        replace(base, id="window", spec=replace(base.spec, window="R3M")),
        replace(base, id="basis", spec=replace(base.spec, basis="YoY")),
        replace(base, id="viz", spec=replace(base.spec, viz_kind="line")),
        replace(base, id="persona",
                spec=replace(base.spec, default_personas=("executive",))),
        replace(base, id="watch-mode", watched=False),
    ]
    assert all(store.add(insight).added for insight in distinct)
    assert len(store) == 1 + len(distinct)


def test_stale_legacy_watch_remains_visible_removable_and_cannot_execute():
    stale = saved_insights.migrate_legacy_watch({
        "metric": "retired_metric",
        "filters": {},
        "label": "Old watch",
    })
    assert stale.is_stale
    assert "unregistered metric" in stale.stale_reason
    store = saved_insights.InMemorySavedInsightStore([stale])
    assert store.all() == (stale,)
    with pytest.raises(tile_runtime.StaleInsightError, match="unregistered metric"):
        tile_runtime.evaluate_saved(stale)
    assert store.remove(stale.id)
    assert len(store) == 0


def test_session_store_instances_do_not_leak_between_session_mappings():
    session_a, session_b = {}, {}
    store_a = saved_insights.session_store(session_a)
    store_b = saved_insights.session_store(session_b)
    store_a.save("revenue", {"region": "East"})
    assert len(store_a) == 1
    assert len(store_b) == 0
    assert saved_insights.session_store(session_a) is store_a
    assert saved_insights.session_store(session_b) is store_b

    legacy_session = {
        saved_insights.SESSION_STORE_KEY: [
            {"metric": "calls", "filters": {}, "label": "Calls"}
        ]
    }
    migrated = saved_insights.session_store(legacy_session)
    assert len(migrated) == 1
    assert migrated.all()[0].spec.metric == "calls"


def test_catalog_save_snapshots_controls_and_scope():
    saved = saved_insights.save_catalog_tile(
        "trx",
        window="R3M",
        basis="YoY",
        region="East",
        source="source_b",
        variant="normalized",
        viz_kind="line",
    )
    assert saved.catalog_tile_id == "trx"
    assert saved.spec.filters == (("region", "East"),)
    assert saved.spec.window == "R3M"
    assert saved.spec.basis == "YoY"
    assert saved.spec.source == "source_b"
    assert saved.spec.variant == "normalized"
    assert saved.spec.viz_kind == "line"


def test_saved_insight_evaluation_matches_direct_answer_intent():
    saved = saved_insights.create_saved_insight(
        "revenue",
        {"region": "West"},
        source="source_b",
        variant="gross",
        window="R3M",
        basis="YoY",
        viz_kind="line",
    )
    evaluated = tile_runtime.evaluate_saved(saved)
    direct = pipeline.answer_intent(
        evaluated.intent,
        source=saved.spec.source,
        variant=saved.spec.variant,
    )
    assert evaluated.saved_insight_id == saved.id
    assert evaluated.canonical_question == evaluated.intent.question
    assert evaluated.result_hash == direct.result_hash
    assert evaluated.artifact.resolution == direct.resolution
    assert evaluated.artifact.data_version == direct.data_version
