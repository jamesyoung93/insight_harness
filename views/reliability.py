"""Reliability: the system's public track record — accuracy, reproducibility,
correct refusals, and user corrections, with trend across recorded runs."""
from __future__ import annotations

import logging

import altair as alt
import pandas as pd
import streamlit as st

from harness import pipeline, services

_SERIES = ["pass", "reproducible", "correct refusals", "corrections"]
_SERIES_COLORS = ["#2a78d6", "#B07C0E", "#6B7280", "#e34948"]  # validated set


def _trend_chart(hist: pd.DataFrame) -> None:
    df = hist.rename(columns={"pass_rate": "pass", "reproducible_rate": "reproducible",
                              "correct_refusal_rate": "correct refusals",
                              "correction_rate": "corrections"})
    df = df.assign(run=range(1, len(df) + 1))
    long = df.melt(["run"], _SERIES, var_name="measure", value_name="rate").dropna()
    st.altair_chart(
        alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("run:O", title="run"),
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("measure:N",
                            scale=alt.Scale(domain=_SERIES, range=_SERIES_COLORS),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["run:O", "measure:N", alt.Tooltip("rate:Q", format=".0%")],
        ).properties(height=220), width="stretch")


def render() -> None:
    st.title("Reliability")
    st.caption("The system's track record, so you can decide how much to trust it: a "
               "standing question set with independently computed expected answers, run "
               "on demand. Declining correctly is scored, and every question runs twice "
               "and must produce identical result hashes.")

    if st.button("Run accuracy check", type="primary"):
        try:
            with st.spinner("Running every question twice…"):
                st.session_state["golden"] = pipeline.run_golden()
            recorded = pipeline.eval_history()
            st.session_state["golden_ts"] = recorded.iloc[-1]["ts"] if len(recorded) else None
        except Exception:
            logging.getLogger(__name__).exception("accuracy check failed")
            st.error("The accuracy check couldn't finish. Check the data feed, then "
                     "run it again.")

    hist = pipeline.eval_history()
    if "golden" in st.session_state:
        res = st.session_state["golden"]
        # anchor the delta on the run this session actually recorded, so a
        # newer run from another session can't shift the comparison
        matches = hist.index[hist["ts"] == st.session_state.get("golden_ts")].tolist()
        prev = hist.iloc[matches[0] - 1] if matches and matches[0] >= 1 else None
        refusals = res[res["tier"] == "Abstained"]

        def delta(cur, col):
            return f"{(cur - prev[col])*100:+.0f}pp" if prev is not None else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Pass rate", f"{res['pass'].mean()*100:.0f}%",
                  delta=delta(res["pass"].mean(), "pass_rate"))
        c2.metric("Reproducible", f"{res['reproducible'].mean()*100:.0f}%",
                  delta=delta(res["reproducible"].mean(), "reproducible_rate"))
        c3.metric("Correct refusals", f"{refusals['pass'].mean()*100:.0f}%",
                  delta=delta(refusals["pass"].mean(), "correct_refusal_rate"))
        st.dataframe(res, width="stretch", hide_index=True)
    elif len(hist):
        last = hist.iloc[-1]
        st.caption(f"Last recorded run ({last['ts']}): pass {last['pass_rate']*100:.0f}% · "
                   f"reproducible {last['reproducible_rate']*100:.0f}% · correct refusals "
                   f"{last['correct_refusal_rate']*100:.0f}%. Run the accuracy check to "
                   "refresh the record.")
    else:
        st.info("Run the accuracy check to see the current pass, reproducibility, and "
                "correct-refusal rates.")

    if len(hist) >= 2:
        st.markdown("**Trend across recorded runs**")
        _trend_chart(hist)

    st.divider()
    st.markdown("**Correction record** — answers users flagged, reviewed against the "
                "governed definitions")
    fb = services.feedback_history()
    votes = fb[fb["verdict"].isin(["correct", "wrong"])] if len(fb) else fb
    if len(votes):
        st.metric("Correction rate", f"{(votes['verdict']=='wrong').mean()*100:.1f}%",
                  help="Share of feedback that flagged an answer as wrong. Its trend "
                       "across runs is charted above.")
        st.dataframe(fb.tail(20), width="stretch", hide_index=True)
    else:
        st.info("No feedback yet. Use 👍 / 🚩 on any answer to build the record.")

    st.divider()
    stats = st.session_state.get("llm_stats", {"validated": 0, "rejected": 0})
    st.markdown("**Language-model translation, this session**")
    st.caption(f"{stats['validated']} translation(s) validated against the registry · "
               f"{stats['rejected']} rejected and answered by the built-in parser instead. "
               "Every translation is checked before anything runs; a rejected one never "
               "reaches an engine.")
