"""Phase 1 registry schema and pipeline-only tile execution."""
import pytest

from harness import pipeline, tile_runtime, tiles


def test_catalog_definitions_carry_the_saved_question_schema():
    assert tiles.TILE_DEFINITIONS
    for definition in tiles.TILE_DEFINITIONS:
        spec = tiles.question_spec(definition)
        assert spec.metric == definition.metric
        assert spec.filters == definition.filters
        assert spec.source == definition.source
        assert spec.variant == definition.variant
        assert spec.window == definition.window
        assert spec.basis == definition.basis
        assert spec.viz_kind == definition.viz_kind
        assert spec.default_personas == definition.default_personas
        assert spec.default_personas
        assert tiles.spec_errors(spec) == ()


@pytest.mark.parametrize("spec,error", [
    (tiles.SavedQuestionSpec("retired"), "unregistered metric"),
    (tiles.SavedQuestionSpec("calls", source="source_b"), "not registered"),
    (tiles.SavedQuestionSpec("units", variant="gross"), "not registered"),
    (tiles.SavedQuestionSpec("revenue", window="R9M"), "unknown window"),
    (tiles.SavedQuestionSpec("revenue", basis="WoW"), "unknown comparison"),
    (tiles.SavedQuestionSpec("revenue", viz_kind="pie"), "unknown visualization"),
    (tiles.SavedQuestionSpec("revenue", default_personas=("wizard",)), "unknown default"),
    (tiles.SavedQuestionSpec("revenue", filters=(("region", "Atlantis"),)),
     "unregistered region"),
])
def test_invalid_saved_question_specs_are_rejected(spec, error):
    with pytest.raises(ValueError, match=error):
        tiles.require_valid_spec(spec)


def test_saved_spec_question_round_trips_all_execution_fields():
    spec = tiles.SavedQuestionSpec(
        metric="revenue",
        filters=tiles.freeze_filters({"segment": "Enterprise", "region": "West"}),
        source="source_b",
        variant="gross",
        window="R3M",
        basis="YoY",
        viz_kind="line",
        default_personas=("executive",),
    )
    intent = tiles.intent_for_spec(spec)
    assert intent.question == (
        "Trend revenue by month last 3 months in West and Enterprise "
        "vs same month last year")
    assert intent.metric == "revenue"
    assert intent.filters == {"region": "West", "segment": "Enterprise"}
    assert intent.window.months
    assert intent.compare_basis == "yoy"


def test_runtime_executes_strictly_through_answer_intent(monkeypatch):
    real_answer_intent = pipeline.answer_intent
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("tile runtime must not call pipeline.answer")

    def spy(intent, source=None, variant=None, translation=None):
        calls.append((intent, source, variant, translation))
        return real_answer_intent(intent, source, variant, translation)

    monkeypatch.setattr(pipeline, "answer", forbidden)
    monkeypatch.setattr(pipeline, "answer_intent", spy)
    evaluation = tile_runtime.evaluate_tile(
        "trx",
        window="R3M",
        basis="YoY",
        region="West",
        source="source_a",
        variant="units",
    )

    assert len(calls) == 1
    intent, source, variant, translation = calls[0]
    assert intent is evaluation.intent
    assert source == "source_a"
    assert variant == "units"
    assert translation is None
    assert evaluation.artifact.resolution.source == "source_a"
    assert evaluation.artifact.resolution.variant == "units"
    assert evaluation.artifact.result_hash == evaluation.result_hash
    assert evaluation.canonical_question == intent.question


def test_registry_defaults_flow_into_runtime_resolution():
    definition = tiles.TileDefinition(
        "normalized_panel_trx",
        "Normalized panel TRx",
        "trx",
        source="source_b",
        variant="normalized",
        window="R6M",
        basis="QoQ",
        viz_kind="line",
        default_personas=("executive",),
    )
    evaluation = tile_runtime.evaluate_tile(definition)
    assert evaluation.spec.source == "source_b"
    assert evaluation.spec.variant == "normalized"
    assert evaluation.spec.window == "R6M"
    assert evaluation.spec.basis == "QoQ"
    assert evaluation.artifact.resolution.source == "source_b"
    assert evaluation.artifact.resolution.variant == "normalized"
