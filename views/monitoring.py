"""Monitoring: movements ranked by business impact, each one click away from
its breakdown."""
from __future__ import annotations

import logging

import streamlit as st

from harness import saved_insights, services, tile_runtime, tiles
from harness import semantic_layer as sl
from views import common


def _display_value(art, value: float | None) -> str:
    if value is None:
        return "—"
    metric = sl.METRICS[art.resolution.metric]
    if metric.get("kind") == "ratio":
        return f"{float(value):.1%}"
    if art.resolution.variant == "dollars" or art.resolution.metric == "revenue":
        return "$" + f"{float(value):,.1f}"
    return f"{float(value):,.1f}"


def _remove_saved(store: saved_insights.InMemorySavedInsightStore,
                  insight_id: str) -> None:
    store.remove(insight_id)


def _watched_section() -> None:
    st.subheader("Watched")
    store = common.saved_insight_store()
    watched = store.all()
    if not watched:
        st.caption("Nothing watched yet. Pin a metric with the 👁 Watch action under "
                   "any measured answer.")
        return
    for i, insight in enumerate(watched):
        c1, c2, c3, c4, c5 = st.columns([2.6, 2.6, 1.1, 1.6, 1.0])
        c1.markdown(f"**{insight.label}**")
        if insight.is_stale:
            c1.caption("saved definition is no longer registered")
            c2.caption(insight.stale_reason)
            c3.caption("—")
        else:
            try:
                evaluation = tile_runtime.evaluate_saved(insight)
                art = evaluation.artifact
                comparison = art.extras.get("comparison", {})
                source = sl.SOURCES[art.resolution.source]["name"]
                variant = sl.METRICS[art.resolution.metric]["variants"][
                    art.resolution.variant]["label"]
                c1.caption(f"{source} · {variant}")
                latest = _display_value(art, art.value)
                if comparison.get("available"):
                    reference = _display_value(art, comparison.get("reference_value"))
                    pct = comparison.get("delta_pct")
                    delta = f" · {float(pct):+.1%}" if pct is not None else ""
                    c2.caption(f"{latest} latest vs {reference} "
                               f"{comparison['basis_label']}{delta}")
                else:
                    c2.caption(f"{latest} latest · comparison history unavailable")
                material = sum(1 for fork in art.divergence if fork.get("material"))
                c3.caption(f"⚠ {material} fork(s)" if material else art.tier.lower())
                filters = dict(insight.spec.filters)
                question = services.breakdown_question(
                    insight.spec.metric,
                    filters,
                    tiles.WINDOW_CONTROLS[insight.spec.window],
                    tiles.BASIS_CONTROLS[insight.spec.basis],
                )
                c4.button("Break this down", key=f"watch_{i}",
                          on_click=common.queue_question, args=(question,))
            except Exception:
                logging.getLogger(__name__).exception(
                    "saved insight evaluation failed: %s", insight.id)
                c2.caption("This saved insight could not be evaluated.")
                c3.caption("—")
        c5.button("Remove", key=f"unwatch_{i}", on_click=_remove_saved,
                  args=(store, insight.id))


def render() -> None:
    st.title("Monitoring")
    st.caption("Movements worth your attention, ranked by business impact rather than "
               "statistical significance.")
    z = st.slider("Sensitivity (z-score threshold)", 1.5, 3.5, 2.0, 0.25)
    st.caption("Sensitivity applies to system-surfaced movements; watched cards use "
               "their saved comparison basis.")

    _watched_section()
    st.divider()
    st.subheader("Surfaced by the system")
    try:
        with st.spinner("Scanning for movements…"):
            feed = services.anomaly_feed(z)
    except Exception:
        logging.getLogger(__name__).exception("anomaly feed failed")
        st.error("The monitoring feed couldn't be computed. Check the data feed, "
                 "then reload this page.")
        return

    if len(feed) == 0:
        st.info("Nothing exceeds the threshold this month. Lower the sensitivity to see "
                "smaller movements.")
        return

    st.caption("Each row is a movement the system surfaced on its own. "
               "Break it down to see where it sits.")
    for row in feed.itertuples():
        c1, c2, c3, c4 = st.columns([2.4, 2.4, 1.4, 1.8])
        c1.markdown(f"**{row.metric}** · {row.scope}")
        c2.caption(f"{row.latest:,.1f} latest vs {row.trailing_mean:,.1f} trailing · z={row.z}")
        c3.caption(f"impact {row.impact:,.1f}")
        c4.button("Break this down", key=f"mon_{row.Index}",
                  on_click=common.queue_question,
                  args=(services.breakdown_question(row.metric_id, {row.dim: row.value}),))
    with st.expander("All flagged movements, as a table"):
        st.dataframe(feed.drop(columns=["metric_id", "dim", "value"]),
                     width="stretch", hide_index=True)
