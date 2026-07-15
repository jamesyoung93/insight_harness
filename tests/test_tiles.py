"""Phase 0 tile contract: immutable specs, control mapping, and round trips."""
from dataclasses import FrozenInstanceError

import pytest

from harness import semantic_layer as sl
from harness import services, tiles, triage


def test_pharma_tile_catalog_is_immutable_and_registered():
    assert [tile.id for tile in tiles.TILE_DEFINITIONS] == [
        "trx",
        "nrx",
        "nbrx",
        "trx_share",
        "calls",
        "new_writers",
        "samples",
        "commercial_trx",
    ]
    assert all(tile.metric in sl.METRICS for tile in tiles.TILE_DEFINITIONS)
    assert all(hash(tile) for tile in tiles.TILE_DEFINITIONS)
    with pytest.raises(FrozenInstanceError):
        tiles.TILE_DEFINITIONS[0].label = "Changed"


def test_control_mappings_are_complete_and_immutable():
    assert dict(tiles.WINDOW_CONTROLS) == {
        "Latest": None,
        "R3M": 3,
        "R6M": 6,
        "R12M": 12,
    }
    assert dict(tiles.BASIS_CONTROLS) == {
        "MoM": "prior_month",
        "QoQ": "prior_quarter",
        "YoY": "yoy",
    }
    with pytest.raises(TypeError):
        tiles.WINDOW_CONTROLS["R9M"] = 9
    with pytest.raises(TypeError):
        tiles.BASIS_CONTROLS["WoW"] = "prior_week"


@pytest.mark.parametrize("definition", tiles.TILE_DEFINITIONS, ids=lambda tile: tile.id)
def test_every_tile_question_round_trips_to_the_same_intent(definition):
    question = tiles.canonical_question(definition)
    parsed = triage.parse(question)
    intent = tiles.intent_for(definition)

    assert parsed == intent
    assert intent.question == question
    assert intent.question_class == triage.DESCRIPTIVE
    assert intent.metric == definition.metric
    assert intent.filters == tiles.effective_filters(definition)
    assert intent.trend is True
    assert intent.window is None
    assert intent.compare_basis == "prior_month"


@pytest.mark.parametrize("control,n", [("R3M", 3), ("R6M", 6), ("R12M", 12)])
def test_window_controls_round_trip_to_last_n_intents(control, n):
    intent = tiles.intent_for("trx", window=control)
    assert intent.window is not None
    assert intent.window.kind == "last_n"
    assert intent.window.months == sl.months()[-n:]
    assert f"last {n} months" in intent.question


@pytest.mark.parametrize("control,basis", list(tiles.BASIS_CONTROLS.items()))
def test_basis_controls_round_trip(control, basis):
    intent = tiles.intent_for("nrx", basis=control)
    assert intent.compare_basis == basis


def test_global_region_scopes_tiles_without_dropping_catalog_filters():
    assert tiles.region_options()[0] == tiles.ALL_REGIONS
    assert tiles.effective_filters("trx", "East") == {"region": "East"}
    assert tiles.intent_for("trx", region="East").filters == {"region": "East"}

    expected = {"payer_channel": "Commercial", "region": "East"}
    assert tiles.effective_filters("commercial_trx", "East") == expected
    commercial = tiles.intent_for("commercial_trx", region="East")
    assert commercial.filters == expected
    assert "East and Commercial" in commercial.question


def test_combined_controls_have_one_parser_round_trip():
    question = tiles.canonical_question(
        "trx", window="R3M", basis="YoY", region="South")
    assert question == (
        "Trend TRx by month last 3 months in South vs same month last year")
    assert tiles.intent_for(
        "trx", window="R3M", basis="YoY", region="South") == triage.parse(question)


def test_tile_breakdown_keeps_scope_window_and_basis_in_the_question():
    question = services.breakdown_question(
        "trx", {"region": "West"}, window_n=3, basis="yoy")
    intent = triage.parse(question)
    assert intent.question_class == triage.DIAGNOSTIC
    assert intent.metric == "trx"
    assert intent.filters == {"region": "West"}
    assert intent.window.months == sl.months()[-3:]
    assert intent.compare_basis == "yoy"


def test_cache_key_is_hashable_and_covers_every_answer_input():
    governance = (0.02, (("trx", "units"),))
    base = tiles.cache_key(
        "trx", window="R3M", basis="YoY", region="East",
        source="source_a", variant="units", governance=governance,
        data_version="data-v1")

    assert hash(base)
    assert base.tile is tiles.TILES_BY_ID["trx"]
    assert base.window_control == "R3M"
    assert base.basis_control == "YoY"
    assert base.effective_region == "East"
    assert base.source == "source_a"
    assert base.variant == "units"
    assert base.governance == governance
    assert base.data_version == "data-v1"

    changed = [
        tiles.cache_key("nrx", window="R3M", basis="YoY", region="East",
                        source="source_a", variant="units", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R6M", basis="YoY", region="East",
                        source="source_a", variant="units", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="MoM", region="East",
                        source="source_a", variant="units", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="YoY", region="South",
                        source="source_a", variant="units", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="YoY", region="East",
                        source="source_b", variant="units", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="YoY", region="East",
                        source="source_a", variant="dollars", governance=governance,
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="YoY", region="East",
                        source="source_a", variant="units", governance=(0.10, ()),
                        data_version="data-v1"),
        tiles.cache_key("trx", window="R3M", basis="YoY", region="East",
                        source="source_a", variant="units", governance=governance,
                        data_version="data-v2"),
    ]
    assert all(key != base for key in changed)


@pytest.mark.parametrize("kwargs", [
    {"window": "R9M"},
    {"basis": "WoW"},
    {"region": "Atlantis"},
])
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        tiles.intent_for("trx", **kwargs)


def test_unknown_tile_id_is_rejected():
    with pytest.raises(ValueError, match="unknown tile id"):
        tiles.canonical_question("not-a-tile")
