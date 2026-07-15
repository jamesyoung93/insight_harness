"""Pipeline-only execution for registry tiles and saved insights."""
from __future__ import annotations

from dataclasses import dataclass

from . import pipeline, tiles
from .saved_insights import SavedInsight


_DEFAULT = object()


class StaleInsightError(ValueError):
    """Raised when a retained legacy record can no longer be evaluated."""


@dataclass(frozen=True)
class TileEvaluation:
    spec: tiles.SavedQuestionSpec
    canonical_question: str
    intent: object
    artifact: object
    catalog_tile_id: str | None = None
    saved_insight_id: str | None = None

    @property
    def result_hash(self) -> str:
        return self.artifact.result_hash


def evaluate_spec(spec: tiles.SavedQuestionSpec, *,
                  region: str | None = tiles.ALL_REGIONS,
                  translation: dict | None = None,
                  catalog_tile_id: str | None = None,
                  saved_insight_id: str | None = None) -> TileEvaluation:
    """Evaluate through the same governed entry point used by other surfaces."""

    tiles.require_valid_spec(spec)
    intent = tiles.intent_for_spec(spec, region)
    artifact = pipeline.answer_intent(
        intent,
        source=spec.source,
        variant=spec.variant,
        translation=translation,
    )
    return TileEvaluation(
        spec=spec,
        canonical_question=intent.question,
        intent=intent,
        artifact=artifact,
        catalog_tile_id=catalog_tile_id,
        saved_insight_id=saved_insight_id,
    )


def evaluate_tile(tile: tiles.TileDefinition | str, *, window: str | None = None,
                  basis: str | None = None,
                  region: str | None = tiles.ALL_REGIONS,
                  source=_DEFAULT, variant=_DEFAULT,
                  viz_kind: str | None = None,
                  translation: dict | None = None) -> TileEvaluation:
    definition = tiles.tile_definition(tile)
    kwargs = {}
    if source is not _DEFAULT:
        kwargs["source"] = source
    if variant is not _DEFAULT:
        kwargs["variant"] = variant
    spec = tiles.question_spec(
        definition,
        window=window,
        basis=basis,
        viz_kind=viz_kind,
        **kwargs,
    )
    return evaluate_spec(
        spec,
        region=region,
        translation=translation,
        catalog_tile_id=definition.id,
    )


def evaluate_saved(insight: SavedInsight, *,
                   region: str | None = tiles.ALL_REGIONS,
                   translation: dict | None = None) -> TileEvaluation:
    if insight.is_stale:
        raise StaleInsightError(insight.stale_reason)
    return evaluate_spec(
        insight.spec,
        region=region,
        translation=translation,
        catalog_tile_id=insight.catalog_tile_id,
        saved_insight_id=insight.id,
    )


def evaluate_many(insights, *, region: str | None = tiles.ALL_REGIONS) -> tuple[TileEvaluation, ...]:
    return tuple(evaluate_saved(insight, region=region) for insight in insights)
