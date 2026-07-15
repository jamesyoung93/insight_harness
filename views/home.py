"""Glanceable governed KPIs composed with the question workspace."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from harness import profiles, services, tile_runtime, tiles
from harness import semantic_layer as sl
from harness.provenance import TIER_ABSTAINED
from views import ask, common


_CONTROL_KEYS = ("home_window", "home_basis", "home_region", "home_source", "home_variant")
_PERSONA_KEY = "home_persona"
_PERSONA_LAST_KEY = "_home_persona_last"
_PERSONA_APPLIED_KEY = "_home_persona_applied"
_SESSION_DEFAULTS_KEY = "_home_persona_session_defaults"
_DEFAULT_PERSONA = "executive"


@st.cache_data(show_spinner=False)
def _cached_tile(tile_id: str, window: str, basis: str, region: str,
                 source: str | None, variant: str | None, cache_identity: tuple):
    """Evaluate a tile through the governed pipeline.

    ``cache_identity`` deliberately participates in Streamlit's key even
    though execution reconstructs the Intent from the public controls. It
    carries effective governance and the data version, so neither can leave a
    stale number on screen.
    """
    del cache_identity
    return tile_runtime.evaluate_tile(
        tile_id, window=window, basis=basis, region=region,
        source=source, variant=variant).artifact


def _format_value(art) -> str:
    if art.value is None:
        return "—"
    value = float(art.value)
    if art.resolution and art.resolution.metric == "trx_share":
        return f"{value * 100:.1f}%"
    if art.resolution and art.resolution.variant in ("dollars", "net", "gross"):
        return f"${value:,.0f}"
    return f"{value:,.0f}"


def _format_delta(art) -> str | None:
    comparison = art.extras.get("comparison", {})
    pct = comparison.get("delta_pct")
    if not comparison.get("available") or pct is None:
        return None
    label = comparison.get("basis_label", "comparison")
    return f"{float(pct) * 100:+.1f}% {label}"


def _sparkline(chart_df: pd.DataFrame | None) -> None:
    if chart_df is None or chart_df.empty:
        st.caption("No trend is available for this scope.")
        return
    series = [column for column in chart_df.columns if column != "month"]
    long = chart_df.melt("month", var_name="series", value_name="value").dropna()
    if long.empty:
        return
    palette = [common.C_PRIMARY, common.C_REFERENCE][:len(series)]
    chart = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("month:O", axis=None),
        y=alt.Y("value:Q", axis=None, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", legend=None,
                        scale=alt.Scale(domain=series, range=palette)),
        tooltip=[alt.Tooltip("month:O"), alt.Tooltip("series:N"),
                 alt.Tooltip("value:Q", format=",.1f")],
    ).properties(height=82)
    st.altair_chart(chart, width="stretch")


def _compact_stamp(art) -> None:
    resolution = art.resolution
    source = sl.SOURCES[resolution.source]["name"] if resolution else "No source"
    st.markdown(common.chip(art.tier), unsafe_allow_html=True)
    st.caption(f"{source} · result `{art.result_hash}` · data `{art.data_version}`")
    if resolution:
        st.caption(resolution.reason)


def _tile_label(definition: tiles.TileDefinition, region: str) -> str:
    filters = tiles.effective_filters(definition, region)
    if definition.id == "west_enterprise_revenue":
        return definition.label
    if region != tiles.ALL_REGIONS and filters.get("region") == region:
        return f"{definition.label} · {region}"
    return definition.label


def _queue_tile_question(question: str, source: str | None, variant: str | None) -> None:
    st.session_state["ask_src"] = source or "governed default"
    st.session_state["_ask_src_last"] = st.session_state["ask_src"]
    st.session_state["ask_var"] = variant or "governed default"
    st.session_state["_ask_var_last"] = st.session_state["ask_var"]
    common.queue_question(question)


def _render_tile(definition: tiles.TileDefinition, window: str, basis: str,
                 region: str, source: str | None, variant: str | None) -> None:
    identity = tiles.cache_key(definition, window=window, basis=basis, region=region,
                               source=source, variant=variant)
    art = _cached_tile(definition.id, window, basis, region, source, variant, identity)

    if art.tier == TIER_ABSTAINED:
        st.error(art.headline)
        _compact_stamp(art)
        return

    st.metric(_tile_label(definition, region), _format_value(art), _format_delta(art))
    _sparkline(art.chart_df)
    _compact_stamp(art)

    material = [fork for fork in art.divergence if fork.get("material")]
    if material:
        with st.expander(f"⚠ {len(material)} material definition fork(s)"):
            for fork in material:
                note = f" · {fork['note']}" if fork.get("note") else ""
                st.markdown(f"- **{fork['label']}**: {fork['value']:,.1f} "
                            f"({fork['rel_diff'] * 100:+.1f}%){note}")

    filters = tiles.effective_filters(definition, region)
    window_n = tiles.WINDOW_CONTROLS[window]
    basis_code = tiles.BASIS_CONTROLS[basis]
    breakdown = services.breakdown_question(definition.metric, filters, window_n, basis_code)
    question = tiles.canonical_question(definition, window=window, basis=basis, region=region)
    action1, action2 = st.columns(2)
    action1.button("Break this down", key=f"tile_{definition.id}_breakdown",
                   on_click=_queue_tile_question, args=(breakdown, source, variant),
                   width="stretch")
    action2.button("Open question", key=f"tile_{definition.id}_open",
                   on_click=_queue_tile_question, args=(question, source, variant),
                   width="stretch")


def _restore_controls() -> None:
    for key in _CONTROL_KEYS:
        previous = f"_{key}_last"
        if key not in st.session_state and previous in st.session_state:
            st.session_state[key] = st.session_state[previous]


def _remember_controls() -> None:
    for key in _CONTROL_KEYS:
        if key in st.session_state:
            st.session_state[f"_{key}_last"] = st.session_state[key]


def _set_controls(window: str, basis: str, region: str) -> None:
    values = {"home_window": window, "home_basis": basis, "home_region": region}
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
    if snapshot.get("region") not in tiles.region_options():
        return None
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
    _set_controls(snapshot["window"], snapshot["basis"], snapshot["region"])
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
               f"{persona.default_scope}")
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
        "region": st.session_state.get("home_region"),
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


def _control_band() -> tuple[str, str, str, str | None, str | None]:
    _restore_controls()
    c1, c2, c3 = st.columns([2, 2, 2])
    window = c1.radio("Window", tuple(tiles.WINDOW_CONTROLS), horizontal=True,
                      key="home_window")
    basis = c2.radio("Compare", tuple(tiles.BASIS_CONTROLS), horizontal=True,
                     key="home_basis")
    region = c3.selectbox("Region", tiles.region_options(), key="home_region")
    c4, c5 = st.columns(2)
    source_pick = c4.selectbox(
        "Prescription source", ["governed default"] + list(sl.SOURCES), key="home_source",
        format_func=lambda value: value if value == "governed default" else sl.SOURCES[value]["name"])
    variant_pick = c5.selectbox(
        "Sales type", ["governed default", "units", "dollars", "normalized"],
        key="home_variant")
    _remember_controls()
    return (window, basis, region,
            None if source_pick == "governed default" else source_pick,
            None if variant_pick == "governed default" else variant_pick)


def render_kpi_band(persona: profiles.PersonaDefinition) -> None:
    st.subheader("Business pulse")
    st.caption("Monthly governed metrics. Use the controls once; every tile keeps the same "
               "scope when you open or break down its question.")
    window, basis, region, source, variant = _control_band()
    layout = _customize_tiles(persona)
    definitions = tuple(tiles.TILES_BY_ID[tile_id] for tile_id in layout.tile_ids)
    if not definitions:
        st.info("No tiles are selected for this persona. Add one under Customize tiles.")
        return
    for start in range(0, len(definitions), 3):
        row = definitions[start:start + 3]
        columns = st.columns(len(row), gap="medium")
        for column, definition in zip(columns, row):
            with column:
                _render_tile(definition, window, basis, region, source, variant)


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
