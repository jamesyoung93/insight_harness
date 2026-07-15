"""Registry-driven tile and saved-question contract.

Tiles are saved, parser-round-trippable descriptive questions.  This module
contains no Streamlit code and performs no answer computation; the UI can use
the immutable cache key below with ``st.cache_data`` and continue to route the
result through ``pipeline.answer_intent``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

from . import semantic_layer as sl
from . import services, triage


ALL_REGIONS = "All regions"

# UI labels map to the governed Intent vocabulary.  The values are immutable,
# as are the mapping views, so they are safe to share across sessions.
WINDOW_CONTROLS = MappingProxyType({
    "Latest": None,
    "R3M": 3,
    "R6M": 6,
    "R12M": 12,
})
BASIS_CONTROLS = MappingProxyType({
    "MoM": "prior_month",
    "QoQ": "prior_quarter",
    "YoY": "yoy",
})
VIZ_KINDS = ("sparkline", "metric", "line", "table")
PERSONA_IDS = (
    "sales_rep",
    "district_manager",
    "brand_marketing",
    "market_access",
    "executive",
)


FilterValue = str | tuple[str, ...]
FilterItems = tuple[tuple[str, FilterValue], ...]
GovernanceKey = tuple[float, tuple[tuple[str, str], ...]]
_INHERIT = object()


def freeze_filters(filters: Mapping | FilterItems | None) -> FilterItems:
    """Canonicalize filters so identity does not depend on dict/list ordering."""

    if not filters:
        return ()
    items = filters.items() if isinstance(filters, Mapping) else filters
    frozen = []
    for dimension, value in items:
        if isinstance(value, (list, tuple)):
            normalized: FilterValue = tuple(sorted(str(item) for item in value))
        else:
            normalized = str(value)
        frozen.append((str(dimension), normalized))
    order = {dimension: index for index, dimension in enumerate(sl.DIMENSIONS)}
    return tuple(sorted(frozen, key=lambda item: (order.get(item[0], len(order)), item[0])))


@dataclass(frozen=True)
class SavedQuestionSpec:
    """Portable, immutable saved-question contract.

    Window and basis remain declarative control values. Resolved month lists
    are intentionally not persisted, so R3M advances when data advances.
    """

    metric: str
    filters: FilterItems = ()
    source: str | None = None
    variant: str | None = None
    window: str = "Latest"
    basis: str = "MoM"
    viz_kind: str = "sparkline"
    default_personas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", freeze_filters(self.filters))
        object.__setattr__(self, "default_personas", tuple(self.default_personas))


def spec_errors(spec: SavedQuestionSpec) -> tuple[str, ...]:
    """Return registry violations without rejecting stale persisted records."""

    errors: list[str] = []
    metric = sl.METRICS.get(spec.metric)
    if metric is None:
        errors.append(f"unregistered metric: {spec.metric!r}")
    else:
        if spec.source is not None and spec.source not in metric["sources"]:
            errors.append(f"source {spec.source!r} is not registered for {spec.metric!r}")
        if spec.variant is not None and spec.variant not in metric["variants"]:
            errors.append(f"variant {spec.variant!r} is not registered for {spec.metric!r}")

    dimensions = [dimension for dimension, _ in spec.filters]
    duplicates = sorted({dimension for dimension in dimensions
                         if dimensions.count(dimension) > 1})
    if duplicates:
        errors.append(f"duplicate filter dimension(s): {duplicates!r}")
    fact = sl.load_fact("source_a")
    for dimension, value in spec.filters:
        if dimension not in sl.DIMENSIONS:
            errors.append(f"unregistered dimension: {dimension!r}")
            continue
        values = value if isinstance(value, tuple) else (value,)
        if not values:
            errors.append(f"empty filter values for {dimension!r}")
            continue
        registered = set(str(item) for item in fact[dimension].unique())
        unknown = [item for item in values if item not in registered]
        if unknown:
            errors.append(f"unregistered {dimension} value(s): {unknown!r}")

    if spec.window not in WINDOW_CONTROLS:
        errors.append(f"unknown window control: {spec.window!r}")
    if spec.basis not in BASIS_CONTROLS:
        errors.append(f"unknown comparison control: {spec.basis!r}")
    if spec.viz_kind not in VIZ_KINDS:
        errors.append(f"unknown visualization kind: {spec.viz_kind!r}")
    unknown_personas = [p for p in spec.default_personas if p not in PERSONA_IDS]
    if unknown_personas:
        errors.append(f"unknown default persona(s): {unknown_personas!r}")
    return tuple(errors)


def require_valid_spec(spec: SavedQuestionSpec) -> SavedQuestionSpec:
    errors = spec_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    return spec


@dataclass(frozen=True)
class TileDefinition:
    """A hashable saved-question definition, independent of display state."""

    id: str
    label: str
    metric: str
    filters: FilterItems = ()
    source: str | None = None
    variant: str | None = None
    window: str = "Latest"
    basis: str = "MoM"
    viz_kind: str = "sparkline"
    default_personas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", freeze_filters(self.filters))
        object.__setattr__(self, "default_personas", tuple(self.default_personas))
        dimensions = [dimension for dimension, _ in self.filters]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(f"duplicate tile filter dimension: {self.filters!r}")
        require_valid_spec(SavedQuestionSpec(
            metric=self.metric,
            filters=self.filters,
            source=self.source,
            variant=self.variant,
            window=self.window,
            basis=self.basis,
            viz_kind=self.viz_kind,
            default_personas=self.default_personas,
        ))


TILE_DEFINITIONS = (
    TileDefinition("trx", "TRx", "trx", default_personas=PERSONA_IDS),
    TileDefinition("nrx", "NRx", "nrx",
                   default_personas=("district_manager", "brand_marketing", "executive")),
    TileDefinition("nbrx", "NBRx", "nbrx",
                   default_personas=("district_manager", "brand_marketing", "executive")),
    TileDefinition("trx_share", "TRx market share", "trx_share",
                   default_personas=("brand_marketing", "market_access", "executive")),
    TileDefinition("calls", "Details", "calls",
                   default_personas=("sales_rep", "district_manager")),
    TileDefinition("new_writers", "New writers", "new_writers",
                   default_personas=("sales_rep", "district_manager", "brand_marketing")),
    TileDefinition("samples", "Samples dropped", "samples",
                   default_personas=("sales_rep", "district_manager")),
    TileDefinition("commercial_trx", "Commercial TRx", "trx",
                   (("payer_channel", "Commercial"),),
                   default_personas=("market_access", "executive")),
)
TILES_BY_ID = MappingProxyType({tile.id: tile for tile in TILE_DEFINITIONS})
if len(TILES_BY_ID) != len(TILE_DEFINITIONS):
    raise RuntimeError("tile registry ids must be unique")


class TileCacheKey(NamedTuple):
    """Every answer-affecting input used to cache one tile artifact."""

    tile: TileDefinition
    window_control: str
    basis_control: str
    effective_region: str | tuple[str, ...] | None
    source: str | None
    variant: str | None
    governance: GovernanceKey
    data_version: str


def tile_definition(tile: TileDefinition | str) -> TileDefinition:
    """Resolve a definition or public tile id to one immutable definition."""

    if isinstance(tile, TileDefinition):
        return tile
    try:
        return TILES_BY_ID[tile]
    except KeyError as exc:
        raise ValueError(f"unknown tile id: {tile!r}") from exc


def question_spec(tile: TileDefinition | str, *, window: str | None = None,
                  basis: str | None = None, source=_INHERIT, variant=_INHERIT,
                  viz_kind: str | None = None) -> SavedQuestionSpec:
    """Materialize a registry tile as the portable saved-question schema."""

    definition = tile_definition(tile)
    return require_valid_spec(SavedQuestionSpec(
        metric=definition.metric,
        filters=definition.filters,
        source=definition.source if source is _INHERIT else source,
        variant=definition.variant if variant is _INHERIT else variant,
        window=window if window is not None else definition.window,
        basis=basis if basis is not None else definition.basis,
        viz_kind=viz_kind if viz_kind is not None else definition.viz_kind,
        default_personas=definition.default_personas,
    ))


def region_options() -> tuple[str, ...]:
    """Regions available to the global scope control, with all-scope first."""

    regions = tuple(sorted(str(value) for value in sl.load_fact("source_a")["region"].unique()))
    return (ALL_REGIONS, *regions)


def _validated_controls(window: str, basis: str, region: str | None) -> None:
    if window not in WINDOW_CONTROLS:
        raise ValueError(f"unknown window control: {window!r}")
    if basis not in BASIS_CONTROLS:
        raise ValueError(f"unknown comparison control: {basis!r}")
    if region is not None and region not in region_options():
        raise ValueError(f"unknown global region: {region!r}")


def effective_filters(tile: TileDefinition | str,
                      region: str | None = ALL_REGIONS) -> dict[str, FilterValue]:
    """Return a fresh filter mapping after applying the global region scope.

    A tile's definition-level region is authoritative.  This keeps the fixed
    West / Enterprise tile honest when a different global region is selected;
    the global selector scopes every tile that does not already pin a region.
    """

    definition = tile_definition(tile)
    _validated_controls("Latest", "MoM", region)
    filters = dict(definition.filters)
    if "region" not in filters and region not in (None, ALL_REGIONS):
        filters["region"] = region
    return filters


def effective_spec_filters(spec: SavedQuestionSpec,
                           region: str | None = ALL_REGIONS) -> dict[str, FilterValue]:
    """Apply the global region scope to an arbitrary saved-question spec."""

    _validated_controls(spec.window, spec.basis, region)
    filters = dict(spec.filters)
    if "region" not in filters and region not in (None, ALL_REGIONS):
        filters["region"] = region
    return filters


def _scope_values(filters: dict[str, FilterValue]) -> list[str]:
    values: list[str] = []
    for dimension in sl.DIMENSIONS:
        if dimension not in filters:
            continue
        value = filters[dimension]
        values.extend(value if isinstance(value, tuple) else (value,))
    return values


def canonical_question_for_spec(spec: SavedQuestionSpec,
                                region: str | None = ALL_REGIONS) -> str:
    """Build the canonical parser-round-trippable text for a saved spec."""

    require_valid_spec(spec)
    _validated_controls(spec.window, spec.basis, region)
    keyword = services.PRIMARY_KEYWORD.get(spec.metric, spec.metric.replace("_", " "))
    parts = [f"Trend {keyword} by month"]

    months = WINDOW_CONTROLS[spec.window]
    if months is not None:
        parts.append(f"last {months} months")

    scope = _scope_values(effective_spec_filters(spec, region))
    if scope:
        parts.append("in " + " and ".join(scope))

    parts.append(triage.BASIS_LABELS[BASIS_CONTROLS[spec.basis]])
    return " ".join(parts)


def canonical_question(tile: TileDefinition | str, *, window: str | None = None,
                       basis: str | None = None,
                       region: str | None = ALL_REGIONS) -> str:
    """Build the sole question text used by both a tile and its Ask drill-through."""

    return canonical_question_for_spec(
        question_spec(tile, window=window, basis=basis), region)


def intent_for_spec(spec: SavedQuestionSpec,
                    region: str | None = ALL_REGIONS) -> triage.Intent:
    """Parse and verify an arbitrary saved-question spec against the registry."""

    question = canonical_question_for_spec(spec, region)
    intent = triage.parse(question)
    expected_filters = effective_spec_filters(spec, region)
    expected_intent_filters = {
        dimension: (list(value) if len(value) > 1 else value[0])
        if isinstance(value, tuple) else value
        for dimension, value in expected_filters.items()
    }
    if intent.question_class != triage.DESCRIPTIVE or not intent.trend \
            or intent.metric != spec.metric or intent.filters != expected_intent_filters:
        raise RuntimeError(f"saved question no longer round-trips: {question!r}")
    return intent


def intent_for(tile: TileDefinition | str, *, window: str | None = None,
               basis: str | None = None,
               region: str | None = ALL_REGIONS) -> triage.Intent:
    """Return the governed Intent obtained by parsing the canonical tile question."""

    return intent_for_spec(question_spec(tile, window=window, basis=basis), region)


def governance_cache_key() -> GovernanceKey:
    """Snapshot effective governance inputs that can change a tile artifact."""

    variants = tuple((metric, sl.default_variant(metric)) for metric in sorted(sl.METRICS))
    return sl.materiality(), variants


def cache_key(tile: TileDefinition | str, *, window: str | None = None,
              basis: str | None = None,
              region: str | None = ALL_REGIONS, source: str | None = None,
              variant: str | None = None, governance: GovernanceKey | None = None,
              data_version: str | None = None) -> TileCacheKey:
    """Return the immutable key for a cached ``answer_intent`` tile call."""

    definition = tile_definition(tile)
    window_control = window if window is not None else definition.window
    basis_control = basis if basis is not None else definition.basis
    _validated_controls(window_control, basis_control, region)
    filters = effective_filters(definition, region)
    effective_region = filters.get("region")
    return TileCacheKey(
        tile=definition,
        window_control=window_control,
        basis_control=basis_control,
        effective_region=effective_region,
        source=source if source is not None else definition.source,
        variant=variant if variant is not None else definition.variant,
        governance=governance if governance is not None else governance_cache_key(),
        data_version=data_version if data_version is not None else sl.data_version(),
    )
