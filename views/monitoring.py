"""Monitoring: movements ranked by business impact, each one click away from
its breakdown."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from harness import services
from views import common


def _watched_section(z: float) -> None:
    st.subheader("Watched")
    watches = services.load_watchlist()
    if not watches:
        st.caption("Nothing watched yet. Pin a metric with the 👁 Watch action under "
                   "any measured answer.")
        return
    feed = services.watch_feed(watches, z)
    for i, row in enumerate(feed.itertuples()):
        c1, c2, c3, c4, c5 = st.columns([2.6, 2.6, 1.1, 1.6, 1.0])
        c1.markdown(f"**{row.label}**")
        registered = not pd.isna(row.latest) if row.latest is not None else False
        if not registered:
            c2.caption("this metric is no longer registered — remove the watch")
            c3.caption("—")
        elif row.trailing_mean is None or pd.isna(row.trailing_mean):
            c2.caption("not enough history to judge movement yet")
            c3.caption("—")
        else:
            c2.caption(f"{row.latest:,.1f} latest vs {row.trailing_mean:,.1f} trailing · z={row.z}")
            c3.caption("⚠ moved" if row.flagged else "steady")
        if registered:
            c4.button("Break this down", key=f"watch_{i}",
                      on_click=common.queue_question,
                      args=(services.breakdown_question(row.metric_id, row.filters),))
        c5.button("Remove", key=f"unwatch_{i}", on_click=services.remove_watch,
                  args=(row.metric_id, row.filters))


def render() -> None:
    st.title("Monitoring")
    st.caption("Movements worth your attention, ranked by business impact rather than "
               "statistical significance.")
    z = st.slider("Sensitivity (z-score threshold)", 1.5, 3.5, 2.0, 0.25)

    _watched_section(z)
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
