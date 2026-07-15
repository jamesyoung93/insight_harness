"""Glanceable governed KPIs composed with the question workspace."""
from __future__ import annotations

import logging

import altair as alt
import pandas as pd
import streamlit as st

from harness import profiles, saved_insights, services, tile_runtime, tiles
from harness import semantic_layer as sl
from harness.provenance import TIER_ABSTAINED
from views import ask, common


_CONTROL_KEYS = ("home_window", "home_basis", "home_scope", "home_source", "home_variant")
_PERSONA_KEY = "home_persona"
_PERSONA_LAST_KEY = "_home_persona_last"
_PERSONA_APPLIED_KEY = "_home_persona_applied"
_SESSION_DEFAULTS_KEY = "_home_persona_session_defaults"
_DEFAULT_PERSONA = "executive"


@st.cache_data(show_spinner=False)
def _cached_tile(cache_identity: tiles.TileCacheKey):
    """Execute the exact immutable spec represented by the cache identity."""

    return tile_runtime.evaluate_spec(
        cache_identity.spec, scope=dict(cache_identity.effective_scope)).artifact


def _format_value(art) -> str:
    return common.format_artifact_value(art, art.value, places=0)


def _format_delta(art) -> str | None:
    comparison = art.extras.get("comparison", {})
    delta = common.format_comparison_delta(art, comparison)
    if delta is None:
        return None
    label = comparison.get("basis_label", "comparison")
    return f"{delta} {label}"


def _sparkline(chart_df: pd.DataFrame | None) -> None:
    if chart_df is None or chart_df.empty:
        st.caption("No trend is available for this scope.")
        return
    series = [column for column in chart_df.columns if column != "month"]
    long = chart_df.melt("month", var_name="series", value_name="value").dropna()
    if long.empty:
        return
    palette = [common.C_PRIMARY, common.C_REFERENCE][:len(series)]
    dash = [[1, 0], [6, 4]][:len(series)]
    chart = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("month:O", axis=None),
        y=alt.Y("value:Q", axis=None, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", legend=alt.Legend(title=None, orient="bottom"),
                        scale=alt.Scale(domain=series, range=palette)),
        strokeDash=alt.StrokeDash("series:N", legend=None,
                                  scale=alt.Scale(domain=series, range=dash)),
        tooltip=[alt.Tooltip("month:O"), alt.Tooltip("series:N"),
                 alt.Tooltip("value:Q", format=",.1f")],
    ).properties(height=82, description="KPI trend; reference series uses a dashed line.")
    st.altair_chart(chart, width="stretch")


def _compact_stamp(art) -> None:
    resolution = art.resolution
    source = sl.SOURCES[resolution.source]["name"] if resolution else "No source"
    st.markdown(common.chip(art.tier), unsafe_allow_html=True)
    st.caption(f"{source} · result `{art.result_hash}` · data `{art.data_version}`")
    if resolution:
        st.caption(resolution.reason)


def _tile_label(definition: tiles.TileDefinition, scope) -> str:
    external = dict(tiles.require_valid_scope(scope))
    filters = tiles.effective_filters(definition, scope=scope)
    if external and all(filters.get(key) == value for key, value in external.items()):
        value = next(iter(external.values()))
        return f"{definition.label} · {value}"
    return definition.label


def _queue_tile_question(question: str, source: str | None, variant: str | None,
                         basis: str | None = None) -> None:
    helper = getattr(common, "queue_question_with_resolution", None)
    if helper is not None:
        helper(question, source, variant, basis=basis)
        return
    st.session_state["ask_src"] = source or "governed default"
    st.session_state["_ask_src_last"] = st.session_state["ask_src"]
    st.session_state["ask_var"] = variant or "governed default"
    st.session_state["_ask_var_last"] = st.session_state["ask_var"]
    common.queue_question(question)


def _watch_tile(spec: tiles.SavedQuestionSpec, scope, definition: tiles.TileDefinition) -> None:
    insight = saved_insights.save_spec(
        spec, label=_tile_label(definition, scope), catalog_tile_id=definition.id,
        watched=True, scope=scope)
    result = saved_insights.session_store(st.session_state).add(insight)
    st.toast("Watching — see Monitoring." if result.added else "Already on the watchlist.")


def _format_fork_value(art, value: float) -> str:
    if art.resolution and sl.metric_kind(art.resolution.metric) == "ratio":
        return f"{float(value):.1%}"
    if art.resolution and art.resolution.variant in ("dollars", "net", "gross"):
        return f"${float(value):,.1f}"
    return f"{float(value):,.1f}"


def _render_tile_body(definition: tiles.TileDefinition, spec: tiles.SavedQuestionSpec,
                      scope, art) -> None:
    label = _tile_label(definition, scope)
    if spec.viz_kind == "count":
        count = len(art.table) if art.table is not None else 0
        st.metric(label, f"{count:,}")
        st.caption(art.headline)
        if art.table is not None and not art.table.empty:
            st.dataframe(art.table.head(5), width="stretch", hide_index=True,
                         height=min(220, 38 + 35 * min(len(art.table), 5)))
    elif spec.viz_kind == "table":
        st.markdown(f"**{label}**")
        st.caption(art.headline)
        if art.table is not None:
            st.dataframe(art.table.head(8), width="stretch", hide_index=True,
                         height=min(300, 38 + 35 * min(len(art.table), 8)))
    else:
        st.metric(label, _format_value(art), _format_delta(art))
        if spec.viz_kind in ("sparkline", "line"):
            _sparkline(art.chart_df)


def _tile_actions(definition: tiles.TileDefinition, spec: tiles.SavedQuestionSpec,
                  scope, art) -> None:
    filters = tiles.effective_spec_filters(spec, scope)
    window_n = tiles.WINDOW_CONTROLS[spec.window] \
        if spec.question_class != "Retrieval" else None
    basis_code = tiles.BASIS_CONTROLS[spec.basis] \
        if spec.question_class != "Retrieval" else None
    breakdown = services.breakdown_question(spec.metric, filters, window_n, basis_code)
    question = tiles.canonical_question_for_spec(spec, scope)
    with st.popover(f"Actions for {definition.label}", width="stretch"):
        st.button(f"Watch {definition.label}", key=f"tile_{definition.id}_watch",
                  on_click=_watch_tile, args=(spec, scope, definition), width="stretch")
        st.download_button(
            f"Download {definition.label} JSON", data=art.to_json(),
            file_name=f"tile_{definition.id}_{art.result_hash}.json",
            mime="application/json", key=f"tile_{definition.id}_download", width="stretch")
        st.button(f"Open {definition.label}", key=f"tile_{definition.id}_open",
                  on_click=_queue_tile_question,
                  args=(question, spec.source, spec.variant, basis_code), width="stretch")
        st.button(f"Break down {definition.label}", key=f"tile_{definition.id}_breakdown",
                  on_click=_queue_tile_question,
                  args=(breakdown, spec.source, spec.variant, basis_code), width="stretch")


def _render_tile(definition: tiles.TileDefinition, window: str, basis: str,
                 scope, source: str | None, variant: str | None) -> None:
    try:
        materialized = tiles.materialize_spec(
            definition, window=window, basis=basis, source=source, variant=variant)
        scope_disclosures = tiles.fixed_scope_disclosures(materialized.spec, scope)
        identity = tiles.cache_key_for_spec(materialized.spec, scope=scope)
        art = _cached_tile(identity)
    except Exception:
        logging.getLogger(__name__).exception("tile evaluation failed: %s", definition.id)
        st.error(f"{definition.label} could not be evaluated; no unverified value is shown.")
        return

    if art.tier == TIER_ABSTAINED:
        st.error(art.headline)
        _compact_stamp(art)
        return

    _render_tile_body(definition, materialized.spec, scope, art)
    _compact_stamp(art)
    for disclosure in (*materialized.disclosures, *scope_disclosures):
        st.caption(f"Override note: {disclosure}")

    material = [fork for fork in art.divergence if fork.get("material")]
    if material:
        with st.expander(f"⚠ {len(material)} material definition fork(s)"):
            for fork in material:
                note = f" · {fork['note']}" if fork.get("note") else ""
                st.markdown(f"- **{fork['label']}**: {_format_fork_value(art, fork['value'])} "
                            f"({fork['rel_diff'] * 100:+.1f}%){note}")
    _tile_actions(definition, materialized.spec, scope, art)


def _restore_controls() -> None:
    if "home_scope" not in st.session_state and "home_region" in st.session_state:
        try:
            st.session_state["home_scope"] = tiles.scope_key(st.session_state["home_region"])
        except ValueError:
            pass
    for key in _CONTROL_KEYS:
        previous = f"_{key}_last"
        if key not in st.session_state and previous in st.session_state:
            st.session_state[key] = st.session_state[previous]


def _remember_controls() -> None:
    for key in _CONTROL_KEYS:
        if key in st.session_state:
            st.session_state[f"_{key}_last"] = st.session_state[key]


def _set_controls(window: str, basis: str, scope) -> None:
    values = {"home_window": window, "home_basis": basis,
              "home_scope": tiles.scope_key(scope)}
    for key, value in values.items():
        st.session_state[key] = value
        st.session_state[f"_{key}_last"] = value


def _session_defaults() -> dict[str, dict]:
    raw = st.session_state.get(_SESSION_DEFAULTS_KEY, {})
    return dict(raw) if isinstance(raw, dict) else {}


def _valid_snapshot(persona: profiles.PersonaDefinition) -> dict | None:
    snapshot = _session_defaults().get(persona.id)
    if not isinstance(snapshot, dict):
        return None
    tile_ids = snapshot.get("tile_ids")
    if not isinstance(tile_ids, (list, tuple)) or any(
            tile_id not in tiles.TILES_BY_ID for tile_id in tile_ids):
        return None
    if snapshot.get("window") not in tiles.WINDOW_CONTROLS:
        return None
    if snapshot.get("basis") not in tiles.BASIS_CONTROLS:
        return None
    raw_scope = snapshot.get("scope", snapshot.get("region", tiles.ALL_REGIONS))
    if tiles.scope_errors(raw_scope):
        return None
    snapshot = dict(snapshot)
    snapshot["scope"] = dict(tiles.require_valid_scope(raw_scope))
    return snapshot


def _replace_layout(persona: profiles.PersonaDefinition, tile_ids: tuple[str, ...]) -> None:
    profiles.reset_layout(st.session_state, persona)
    current = profiles.layout_for(st.session_state, persona)
    for tile_id in tuple(current.tile_ids):
        if tile_id not in tile_ids:
            profiles.remove_tile(st.session_state, persona, tile_id)
    current = profiles.layout_for(st.session_state, persona)
    for tile_id in tile_ids:
        if tile_id not in current.tile_ids:
            profiles.add_tile(st.session_state, persona, tile_id)
    profiles.reorder_tiles(st.session_state, persona, tile_ids)


def _apply_persona(persona: profiles.PersonaDefinition) -> None:
    snapshot = _valid_snapshot(persona)
    if snapshot is None:
        _set_controls(persona.default_window, persona.default_basis, persona.default_scope)
        return
    _set_controls(snapshot["window"], snapshot["basis"], snapshot["scope"])
    _replace_layout(persona, tuple(snapshot["tile_ids"]))


def _persona_selector() -> profiles.PersonaDefinition:
    if _PERSONA_KEY not in st.session_state:
        remembered = st.session_state.get(_PERSONA_LAST_KEY, _DEFAULT_PERSONA)
        st.session_state[_PERSONA_KEY] = (
            remembered if remembered in profiles.PERSONAS_BY_ID else _DEFAULT_PERSONA)
    persona_id = st.selectbox(
        "Persona view",
        tuple(profiles.PERSONAS_BY_ID),
        key=_PERSONA_KEY,
        format_func=lambda value: profiles.PERSONAS_BY_ID[value].label,
        help="Each preset supplies a starting tile set, scope, window, and comparison.",
    )
    st.session_state[_PERSONA_LAST_KEY] = persona_id
    persona = profiles.persona_definition(persona_id)
    if st.session_state.get(_PERSONA_APPLIED_KEY) != persona_id:
        _apply_persona(persona)
        st.session_state[_PERSONA_APPLIED_KEY] = persona_id
    st.caption(f"Preset: {persona.default_window} · {persona.default_basis} · "
               f"{tiles.scope_label(persona.default_scope)}")
    return persona


def _move_tile(persona_id: str, tile_id: str, offset: int) -> None:
    state = profiles.layout_for(st.session_state, persona_id)
    ordered = list(state.tile_ids)
    index = ordered.index(tile_id)
    target = index + offset
    if 0 <= target < len(ordered):
        ordered[index], ordered[target] = ordered[target], ordered[index]
        profiles.reorder_tiles(st.session_state, persona_id, ordered)


def _save_session_default(persona_id: str) -> None:
    state = profiles.layout_for(st.session_state, persona_id)
    defaults = _session_defaults()
    defaults[persona_id] = {
        "tile_ids": list(state.tile_ids),
        "window": st.session_state.get("home_window"),
        "basis": st.session_state.get("home_basis"),
        "scope": dict(tiles.scope_from_key(st.session_state.get("home_scope", tiles.ALL_SCOPES))),
    }
    st.session_state[_SESSION_DEFAULTS_KEY] = defaults
    st.toast("Saved for this session.")


def _reset_persona(persona_id: str) -> None:
    persona = profiles.persona_definition(persona_id)
    profiles.reset_layout(st.session_state, persona)
    defaults = _session_defaults()
    defaults.pop(persona_id, None)
    if defaults:
        st.session_state[_SESSION_DEFAULTS_KEY] = defaults
    else:
        st.session_state.pop(_SESSION_DEFAULTS_KEY, None)
    _set_controls(persona.default_window, persona.default_basis, persona.default_scope)


def _customize_tiles(persona: profiles.PersonaDefinition) -> profiles.LayoutResolution:
    state = profiles.layout_for(st.session_state, persona)
    with st.expander("Customize tiles"):
        st.caption("Changes stay in this app session and never alter another viewer's layout.")
        for index, tile_id in enumerate(state.tile_ids):
            definition = tiles.TILES_BY_ID[tile_id]
            label, up, down, remove = st.columns([3.4, 1, 1, 1.2])
            label.markdown(f"**{definition.label}**")
            up.button(
                "Move up", key=f"home_{persona.id}_{tile_id}_up",
                disabled=index == 0, on_click=_move_tile,
                args=(persona.id, tile_id, -1), width="stretch")
            down.button(
                "Move down", key=f"home_{persona.id}_{tile_id}_down",
                disabled=index == len(state.tile_ids) - 1, on_click=_move_tile,
                args=(persona.id, tile_id, 1), width="stretch")
            remove.button(
                "Remove", key=f"home_{persona.id}_{tile_id}_remove",
                on_click=profiles.remove_tile,
                args=(st.session_state, persona.id, tile_id), width="stretch")

        if state.retired_tile_ids:
            st.warning("Some saved tiles are no longer registered. They stay recoverable until "
                       "you remove or reset them.")
            for tile_id in state.retired_tile_ids:
                st.button(
                    f"Remove retired tile: {tile_id}",
                    key=f"home_{persona.id}_{tile_id}_retired_remove",
                    on_click=profiles.remove_tile,
                    args=(st.session_state, persona.id, tile_id))

        missing = tuple(tile_id for tile_id in tiles.TILES_BY_ID if tile_id not in state.tile_ids)
        if missing:
            add_choice = st.selectbox(
                "Tile to add", missing, key=f"home_{persona.id}_add_choice",
                format_func=lambda tile_id: tiles.TILES_BY_ID[tile_id].label)
            st.button(
                "Add tile", key=f"home_{persona.id}_add",
                on_click=profiles.add_tile,
                args=(st.session_state, persona.id, add_choice))
        else:
            st.caption("All registered tiles are already shown.")

        reset, save = st.columns(2)
        reset.button(
            "Reset persona defaults", key=f"home_{persona.id}_reset",
            on_click=_reset_persona, args=(persona.id,), width="stretch")
        save.button(
            "Save as current-session default", key=f"home_{persona.id}_save",
            on_click=_save_session_default, args=(persona.id,), width="stretch")
    return state


def _control_band(definitions: tuple[tiles.TileDefinition, ...]):
    _restore_controls()
    c1, c2, c3 = st.columns([2, 2, 2])
    window = c1.radio("Window", tuple(tiles.WINDOW_CONTROLS), horizontal=True,
                      key="home_window")
    basis = c2.radio("Compare", tuple(tiles.BASIS_CONTROLS), horizontal=True,
                     key="home_basis")
    options = tiles.scope_options()
    option_by_key = {option.key: option for option in options}
    if st.session_state.get("home_scope") not in option_by_key:
        st.session_state["home_scope"] = tiles.ALL_SCOPES
    scope_key = c3.selectbox(
        "Scope", tuple(option_by_key), key="home_scope",
        format_func=lambda key: option_by_key[key].label,
        help="Scope by geography, specialty, or payer channel using registered values.")
    c4, c5 = st.columns(2)
    metric_ids = {definition.metric for definition in definitions}
    sources = tuple(source for source in sl.SOURCES
                    if any(source in sl.METRICS[metric]["sources"] for metric in metric_ids))
    source_pick = c4.selectbox(
        "Prescription source", ("governed default", *sources), key="home_source",
        format_func=lambda value: value if value == "governed default" else sl.SOURCES[value]["name"])
    variants = tuple(sorted({variant for metric in metric_ids
                             for variant in sl.METRICS[metric]["variants"]}))
    variant_pick = c5.selectbox(
        "Sales type", ("governed default", *variants),
        key="home_variant")
    _remember_controls()
    return (window, basis, option_by_key[scope_key].filters,
            None if source_pick == "governed default" else source_pick,
            None if variant_pick == "governed default" else variant_pick)


def render_kpi_band(persona: profiles.PersonaDefinition) -> None:
    st.subheader("Business pulse")
    st.caption("Monthly governed metrics. Compatible tiles use the selected controls; a tile "
               "with a conflicting fixed scope keeps its governed definition and says so.")
    layout = _customize_tiles(persona)
    definitions = tuple(tiles.TILES_BY_ID[tile_id] for tile_id in layout.tile_ids)
    if not definitions:
        st.info("No tiles are selected for this persona. Add one under Customize tiles.")
        return
    window, basis, scope, source, variant = _control_band(definitions)
    for start in range(0, len(definitions), 3):
        row = definitions[start:start + 3]
        columns = st.columns(len(row), gap="medium")
        for column, definition in zip(columns, row):
            with column:
                _render_tile(definition, window, basis, scope, source, variant)


def render() -> None:
    st.title("Home")
    st.caption("Start with the governed state of the business, then interrogate any number. "
               "Every surface resolves through the same deterministic answer pipeline.")
    persona = _persona_selector()
    render_kpi_band(persona)
    st.divider()
    st.subheader("Explore")
    st.caption("Ask about governed metrics in plain language. Unreliable requests receive a "
               "scoped refusal with a concrete reframe.")
    ask.render_workspace()
