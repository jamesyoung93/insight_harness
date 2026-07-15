"""Personalized, deterministic digest of governed monthly signals."""
from __future__ import annotations

import logging
import os
from typing import Mapping

import streamlit as st

from harness import digest as digest_service
from harness import digest_narrator, profiles, saved_insights, semantic_layer as sl, tiles
from harness.digest_store import DigestHistoryStore
from views import common


def _resolved_context(persona: str | None, scope: Mapping | None) -> tuple[str, dict]:
    selected_persona = persona or st.session_state.get("home_persona") \
        or st.session_state.get("_home_persona_last") \
        or st.session_state.get("persona") or st.session_state.get("persona_id") \
        or "executive"
    try:
        definition = profiles.persona_definition(str(selected_persona))
    except ValueError:
        definition = next((item for item in profiles.PERSONAS
                           if item.label.lower() == str(selected_persona).lower()), None)
    label = definition.label if definition else str(selected_persona)
    if scope is not None:
        selected_scope = dict(scope)
    elif isinstance(st.session_state.get("digest_scope"), Mapping):
        selected_scope = dict(st.session_state["digest_scope"])
    elif definition and definition.digest_scope != tiles.ALL_REGIONS:
        selected_scope = {"region": definition.digest_scope}
    else:
        selected_scope = dict(st.session_state.get("persona_scope", {}) or {})
    return label, selected_scope


def _narrated(item, api_key: str, model: str):
    cache = st.session_state.setdefault("_digest_narration_cache", {})
    key = (item.result_hash, model)
    if key not in cache:
        cache[key] = digest_narrator.rewrite_item(item, api_key=api_key, model=model)
    return cache[key]


def _render_item(item, index: int, *, api_key: str | None, model: str,
                 use_model: bool) -> None:
    rendered = _narrated(item, api_key, model) if use_model and api_key else item
    candidate = rendered.candidate
    with st.container(border=True):
        st.markdown(common.chip(candidate.artifact.tier)
                    + f"&nbsp; **{candidate.kind.title()} signal**",
                    unsafe_allow_html=True)
        st.subheader(rendered.headline)
        st.caption(rendered.impact_text)
        narration = rendered.narration
        if narration.get("narrator") == "language_model":
            latency = f" · {narration['latency_ms']} ms" \
                if narration.get("latency_ms") is not None else ""
            st.caption(f"language model (validated){latency}")
        elif narration.get("fallback_kind"):
            st.caption("Templated phrasing retained because the optional rewrite was not validated.")

        action1, action2, action3 = st.columns([1.3, 1.2, 1.4])
        action1.button("Break this down", key=f"digest_break_{index}_{rendered.result_hash}",
                       on_click=common.queue_question,
                       args=(rendered.breakdown_question,), width="stretch")
        action2.download_button(
            "Download artifact", data=rendered.to_json(), mime="application/json",
            file_name=f"digest_item_{rendered.result_hash}.json",
            key=f"digest_download_{index}_{rendered.result_hash}",
        )
        action3.code(rendered.result_hash, language=None)

        with st.expander("Why this surfaced"):
            st.markdown(
                f"- **Normalized impact:** {candidate.impact_score:.3f}\n"
                f"- **Novelty factor:** {rendered.novelty_factor:.3f}\n"
                f"- **Ranking score:** {rendered.score:.3f}\n"
                f"- **Metric family:** {candidate.family}\n"
                f"- **Underlying answer:** `{candidate.artifact.result_hash}`\n"
                f"- **Resolution:** {candidate.artifact.resolution.reason}"
            )
            if candidate.event_name:
                st.caption(f"Registered event flag: {candidate.event_name}. This is context, "
                           "not a causal attribution.")
            if candidate.fork_label:
                st.caption(f"Material alternate: {candidate.fork_label}.")


def render(persona: str | None = None, scope: Mapping | None = None,
           store: DigestHistoryStore | None = None) -> None:
    st.title("Daily digest")
    st.caption("The three highest-ranked governed signals in the current monthly data. "
               "The selection changes when data, governance, scope, or watched items change.")
    persona, scope = _resolved_context(persona, scope)
    st.markdown(f"**View:** {persona} · **Scope:** {sl.scope_string(scope)}")

    insight_store = saved_insights.session_store(st.session_state)
    watches = [{"metric": insight.metric, "filters": insight.filters,
                "source": insight.source, "variant": insight.variant,
                "window": insight.window, "basis": insight.basis}
               for insight in insight_store.all()
               if insight.watched and not insight.is_stale]
    try:
        with st.spinner("Scanning governed series…"):
            artifact = digest_service.build_digest(
                persona=persona, scope=scope, watches=watches,
                store=store or DigestHistoryStore(),
            )
    except Exception:
        logging.getLogger(__name__).exception("digest computation failed")
        st.error("The digest could not be computed, so no unverified ranking is shown.")
        return

    st.caption(f"Scanned {artifact.scanned_series} series across "
               f"{artifact.metric_families} metric families · data "
               f"`{artifact.data_version}` · governance "
               f"`{artifact.governance_fingerprint}` · digest `{artifact.result_hash}`")

    api_key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    use_model = st.checkbox(
        "Use optional language-model phrasing",
        value=False, disabled=not bool(api_key), key="digest_use_model",
        help="Ranking and facts stay deterministic; only the validated wording can change.",
    )
    if not api_key:
        st.caption("Templated phrasing is active. Add a session API credential to enable "
                   "optional validated rewriting.")
    model = st.session_state.get("llm_model") or digest_narrator.DEFAULT_MODEL

    if not artifact.items:
        st.info("No governed signal has enough monthly history to rank yet.")
        return
    for index, item in enumerate(artifact.items, start=1):
        _render_item(item, index, api_key=api_key, model=model, use_model=use_model)
