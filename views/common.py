"""Shared answer rendering: tier chips, the provenance stamp, artifact actions,
decomposition waterfalls, the causal design brief, and scoped-refusal
presentation. Every page renders answers through render_answer()."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from harness import semantic_layer as sl
from harness import saved_insights, services, tiles
from harness.provenance import TIER_ABSTAINED

TIER_COLORS = {"Verified": "#0E7C7B", "Directional": "#B07C0E", "Hypothesis": "#8A4FBE",
               TIER_ABSTAINED: "#6B7280"}

# chart palette (validated: in lightness band, >=3:1 on surface, CVD-safe pairs)
C_UP, C_DOWN, C_TOTAL = "#2a78d6", "#B07C0E", "#6B7280"
C_PRIMARY, C_REFERENCE = "#2a78d6", "#6B7280"
C_LABEL = "#52514e"
HOME_PAGE = "Home"
_BASIS_CONTROL = {code: label for label, code in tiles.BASIS_CONTROLS.items()}


# --------------------------------------------------------------------------- #
# Navigation and question queueing
# --------------------------------------------------------------------------- #
def goto(page: str, **state) -> None:
    """Callback: switch page and set session state in one click."""
    st.session_state.update(state)
    st.session_state["nav"] = page


def queue_question(question: str) -> None:
    """Callback: put a question in the Home explorer and go there."""
    st.session_state["ask_q"] = question
    st.session_state["_ask_q_last"] = question
    st.session_state.pop("replay_key", None)
    st.session_state["nav"] = HOME_PAGE


def clear_replay() -> None:
    st.session_state.pop("replay_key", None)


def saved_insight_store() -> saved_insights.InMemorySavedInsightStore:
    """Return this viewer's store, copying legacy watches once without writes."""

    if saved_insights.SESSION_STORE_KEY not in st.session_state:
        legacy = services.load_watchlist()
        st.session_state[saved_insights.SESSION_STORE_KEY] = \
            saved_insights.InMemorySavedInsightStore(legacy)
    return saved_insights.session_store(st.session_state)


def _watch_window(intent) -> str:
    window = getattr(intent, "window", None)
    if window is None:
        return "Latest"
    if window.kind == "last_n":
        match = next((label for label, months in tiles.WINDOW_CONTROLS.items()
                      if months == len(window.months)), None)
        if match:
            return match
    raise ValueError("Only Latest, R3M, R6M, and R12M windows can be watched.")


def _save_watch(art) -> saved_insights.SaveResult:
    intent = art.extras["intent"]
    resolution = art.resolution
    basis_code = intent.compare_basis
    if basis_code is None and art.engine == "decomposition":
        basis_code = "prior_quarter"
    basis = _BASIS_CONTROL.get(basis_code or "prior_month", "MoM")
    viz_kind = "line" if intent.trend or art.chart_df is not None else "sparkline"
    label = (f"{sl.METRICS[resolution.metric]['label']} · "
             f"{sl.scope_string(intent.filters)}")
    insight = saved_insights.create_saved_insight(
        resolution.metric,
        intent.filters,
        label=label,
        source=resolution.source,
        variant=resolution.variant,
        window=_watch_window(intent),
        basis=basis,
        viz_kind=viz_kind,
    )
    return saved_insight_store().add(insight)


# --------------------------------------------------------------------------- #
# Chips and the stamp
# --------------------------------------------------------------------------- #
def chip(tier: str) -> str:
    return f'<span class="tier-chip" style="background:{TIER_COLORS.get(tier, "#333")}">{tier}</span>'


def render_stamp(art) -> None:
    s = art.stamp()
    tr = art.extras.get("translation", {})
    if tr.get("translator") == "llm":
        tr_label = "language model (validated)"
        if tr.get("latency_ms") is not None:
            tr_label += f", {tr['latency_ms']} ms"
    elif tr.get("fallback_reason"):
        tr_label = "built-in parser (language-model fallback"
        tr_label += f", {tr['latency_ms']} ms)" if tr.get("latency_ms") is not None else ")"
    else:
        tr_label = "built-in parser"
    reviewed = art.extras.get("analyst_reviewed") or \
        st.session_state.get("reviewed", {}).get(art.result_hash)
    reviewed_str = f" · analyst-reviewed {reviewed[:10]}" if reviewed else ""
    st.markdown(f'<div class="stamp">translator: {tr_label} · engine: {s["engine"]} · '
                f'result hash: {s["result_hash"]} · '
                f'data version: {s["data_version"]} · {s["created_at"]}{reviewed_str}</div>',
                unsafe_allow_html=True)
    if tr.get("fallback_reason"):
        st.caption(f"ℹ️ {tr['fallback_reason']}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def line_chart(chart_df: pd.DataFrame) -> None:
    series = [c for c in chart_df.columns if c != "month"]
    long = chart_df.melt("month", var_name="series", value_name="value")
    color = alt.Color("series:N",
                      scale=alt.Scale(domain=series, range=[C_PRIMARY, C_REFERENCE][:len(series)]),
                      legend=None if len(series) == 1 else alt.Legend(orient="top", title=None))
    st.altair_chart(
        alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("month:O", title=None), y=alt.Y("value:Q", title=None), color=color,
            tooltip=[alt.Tooltip("month:O"), alt.Tooltip("series:N"),
                     alt.Tooltip("value:Q", format=",.1f")],
        ).properties(height=260), width="stretch")


def waterfall(t: pd.DataFrame, m0: str, m1: str) -> alt.Chart:
    """Start level → per-group deltas → end level, as a bridge: levels are
    ticks (not zero-anchored bars, so the zoomed axis misrepresents nothing),
    deltas are floating bars, and labels carry the sign so polarity never
    rides on color alone."""
    start, end = float(t["period_start"].sum()), float(t["period_end"].sum())
    rows = [{"label": m0, "y0": start, "y1": start, "kind": "total", "amount": start}]
    run = start
    for r in t.sort_values("delta", ascending=False).itertuples():
        rows.append({"label": str(r.value), "y0": run, "y1": run + float(r.delta),
                     "kind": "up" if r.delta >= 0 else "down", "amount": float(r.delta)})
        run += float(r.delta)
    rows.append({"label": m1, "y0": end, "y1": end, "kind": "total", "amount": end})
    df = pd.DataFrame(rows)
    df["ymax"] = df[["y0", "y1"]].max(axis=1)
    df["text"] = [f"{r.amount:+,.0f}" if r.kind != "total" else f"{r.amount:,.0f}"
                  for r in df.itertuples()]
    connectors = pd.DataFrame([
        {"x": rows[i]["label"], "x2": rows[i + 1]["label"], "y": rows[i]["y1"]}
        for i in range(len(rows) - 1)])

    # sort=None keeps the band domain in data order; a custom sort array is
    # dropped by Vega-Lite when layers share an x scale with an x2 channel
    yscale = alt.Scale(zero=False, nice=True)
    base = alt.Chart(df).encode(
        x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)))
    bars = base.transform_filter(alt.datum.kind != "total").mark_bar(
        size=38, cornerRadius=2).encode(
        y=alt.Y("y0:Q", title=None, scale=yscale), y2="y1:Q",
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["up", "down"], range=[C_UP, C_DOWN])),
        tooltip=[alt.Tooltip("label:N", title="group"),
                 alt.Tooltip("text:N", title="change")],
    )
    levels = base.transform_filter(alt.datum.kind == "total").mark_tick(
        size=46, thickness=3, color=C_TOTAL).encode(
        y=alt.Y("y1:Q", title=None, scale=yscale),
        tooltip=[alt.Tooltip("label:N", title="period"),
                 alt.Tooltip("text:N", title="level")],
    )
    links = alt.Chart(connectors).mark_rule(
        strokeDash=[3, 3], color=C_TOTAL, opacity=0.6).encode(
        x=alt.X("x:N", sort=None), x2="x2:N", y=alt.Y("y:Q", scale=yscale))
    labels = base.mark_text(dy=-10, fontSize=11, color=C_LABEL).encode(
        y=alt.Y("ymax:Q", scale=yscale), text="text:N")
    return (links + bars + levels + labels).properties(height=280)


# --------------------------------------------------------------------------- #
# Engine-specific answer bodies
# --------------------------------------------------------------------------- #
def _decomposition_body(art, key: str) -> None:
    st.caption(art.extras["note"])
    m0, m1 = art.extras["m0"], art.extras["m1"]
    tables = art.extras["tables"]
    if tables:  # empty when every dimension in scope is pinned to one value
        lead = art.extras["lead_dim"]
        dims = sorted(tables, key=lambda d: d != lead)  # lead dimension first
        tabs = st.tabs([f"by {d}" for d in dims])
        for tab, dim in zip(tabs, dims):
            with tab:
                st.altair_chart(waterfall(tables[dim], m0, m1), width="stretch")
                with st.expander("Contribution detail"):
                    st.dataframe(tables[dim], width="stretch", hide_index=True)
    for ev in art.extras.get("overlapping_events", []):
        c1, c2 = st.columns([3, 2])
        c1.info(f"A registered event overlaps this window: **{ev['name']}**. "
                "Attribution needs a designed test, not a read of this chart.")
        c2.button("Test attribution in Causal Studio", key=f"{key}_studio_{ev['id']}",
                  on_click=goto, args=("Causal Studio",),
                  kwargs={"studio_event": ev["id"], "studio_autorun": True,
                          "studio_metric": art.resolution.metric})


def _signoff(art) -> None:
    ts = services.log_review(art)
    st.session_state.setdefault("reviewed", {})[art.result_hash] = ts
    art.extras["analyst_reviewed"] = ts


def _causal_brief(art, key: str) -> None:
    est, ev, sens = art.extras["estimate"], art.extras["event"], art.extras["sensitivity"]
    scope = ", ".join(f"{k}={v}" for k, v in ev["scope"].items())
    controls = ", ".join(ev["candidate_controls"]["region"])

    st.caption(f"Question: {art.question}")
    st.markdown(
        f"**Design** — difference-in-differences around **{ev['name']}** (from {ev['start']}). "
        f"Treated scope: {scope}. Control group: {controls}. "
        f"Pre-period: {len(est['pre'])} months · post-period: {len(est['post'])} months.")

    st.markdown("**Assumption checks — computed from the data**")
    checks = pd.DataFrame(art.extras["checks"])
    checks["status"] = checks["status"].map(
        {"pass": "✅ pass", "flag": "⚠️ flag", "manual": "👤 needs review"})
    st.dataframe(checks, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Estimated effect (control-adjusted)", f"{est['did_pct']*100:+.1f}%")
    c2.metric("Naive pre/post read", f"{est['naive_pct']*100:+.1f}%",
              help="What the change looks like with no control group. The difference "
                   "between the two numbers is what the comparison corrects.")
    if art.chart_df is not None:
        line_chart(art.chart_df)

    sens_bits = [f"{sl.METRICS[art.resolution.metric]['variants'][v]['label']} "
                 f"{p*100:+.1f}%" for v, p in sens.items()]
    sens_bits += [f"{sl.SOURCES[s]['name']} {p*100:+.1f}%"
                  for s, p in art.extras.get("source_sensitivity", {}).items()
                  if s != art.resolution.source]
    st.caption("Sensitivity — the estimate under each registered alternate: "
               + " · ".join(sens_bits))
    if art.extras.get("note"):
        st.caption(art.extras["note"])

    reviewed = st.session_state.get("reviewed", {}).get(art.result_hash) or \
        art.extras.get("analyst_reviewed")
    if reviewed:
        # rehydrate recomputed artifacts so the JSON download matches the badge
        art.extras.setdefault("analyst_reviewed", reviewed)
        st.success(f"Analyst-reviewed on {reviewed[:10]}. The sign-off is recorded with "
                   "the answer's provenance.")
    else:
        st.button("Mark as analyst-reviewed", key=f"{key}_signoff",
                  on_click=_signoff, args=(art,),
                  help="Records your sign-off against this exact result hash and "
                       "data version.")


# --------------------------------------------------------------------------- #
# Cross-cutting blocks: divergence, caveats, provenance, actions
# --------------------------------------------------------------------------- #
def _divergence_block(art) -> None:
    material = [d for d in art.divergence if d["material"]]
    if not material:
        return
    with st.expander(f"⚠ Same question, different answer — {len(material)} material fork(s)"):
        st.caption("The answer above uses the governed default. These registered alternates "
                   "move it materially; escalate unresolved forks to metric governance.")
        for d in art.divergence:
            flag = "**material**" if d["material"] else "immaterial"
            note = f" · {d['note']}" if d["note"] else ""
            st.markdown(f"- {d['label']} → {d['value']:,.1f} ({d['rel_diff']*100:+.1f}%, {flag}){note}")


def _caveats_block(art) -> None:
    if not art.caveats:
        return
    with st.expander(f"⚠ Caveats ({len(art.caveats)})"):
        for c in art.caveats:
            st.markdown(f'<div class="caveat">{c}</div>', unsafe_allow_html=True)


def _provenance_block(art) -> None:
    with st.expander("Provenance — how this number was produced"):
        r = art.resolution
        if r:
            st.markdown(f"- **Metric:** {sl.METRICS[r.metric]['label']} · **variant:** "
                        f"{sl.METRICS[r.metric]['variants'][r.variant]['label']} · **source:** "
                        f"{sl.SOURCES[r.source]['name']}\n- **Resolution:** {r.reason}")
        tr = art.extras.get("translation", {})
        if tr.get("translator") == "llm":
            st.markdown("**Translation (language-model output, registry-validated "
                        "before execution):**")
            st.code(tr.get("raw", ""), language="json")
        st.code(art.code, language="python")


def _actions_row(art, key: str) -> None:
    slug = art.question_class.lower().replace(" ", "_")
    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.5, 0.9, 1.3, 0.9, 2.2])
    c1.download_button("Download answer (JSON)", data=art.to_json(),
                       file_name=f"answer_{slug}_{art.result_hash}.json",
                       mime="application/json", key=f"{key}_json")
    if art.table is not None:
        c2.download_button("Download table (CSV)",
                           data=art.table.to_csv(index=False),
                           file_name=f"answer_{slug}_{art.result_hash}.csv",
                           mime="text/csv", key=f"{key}_csv")
    if c3.button("👍 Correct", key=f"{key}_up"):
        services.log_feedback(art, "correct")
        st.toast("Logged.")
    if c4.button("🚩 Number is wrong", key=f"{key}_down"):
        services.log_feedback(art, "wrong")
        st.toast("Logged — flagged answers are reviewed.")
    watchable = art.resolution is not None and art.extras.get("intent") is not None \
        and art.engine in ("descriptive", "decomposition", "causal_advisor")
    if watchable:
        if c5.button("👁 Watch", key=f"{key}_watch",
                     help="Pin this metric and scope to the Watched list in Monitoring."):
            try:
                result = _save_watch(art)
                st.toast("Watching — see Monitoring." if result.added
                         else "Already on the watchlist.")
            except ValueError as exc:
                st.toast(str(exc))
    with c6:
        st.code(art.result_hash, language=None)  # copyable result hash


def _refusal(art, key: str) -> None:
    reason = art.headline.removeprefix("Declined: ")
    st.markdown(f'<div class="refusal"><b>Scoped refusal</b> — “{art.question}”<br>{reason}</div>',
                unsafe_allow_html=True)
    reframes = art.extras.get("reframes", [])
    if reframes:
        st.caption("Reliable ways to ask this:")
        cols = st.columns(len(reframes) + 1)
        for i, r in enumerate(reframes):
            cols[i].button(r, key=f"{key}_reframe_{i}", on_click=queue_question, args=(r,))
    render_stamp(art)


# --------------------------------------------------------------------------- #
# The single answer renderer
# --------------------------------------------------------------------------- #
def render_answer(art, key: str = "ans") -> None:
    key = f"{key}_{art.result_hash}"
    st.markdown(chip(art.tier) + f"&nbsp; **{art.question_class}** question",
                unsafe_allow_html=True)

    if art.tier == TIER_ABSTAINED:
        _refusal(art, key)
        _actions_row(art, key)
        return

    st.subheader(art.headline)

    if art.engine == "causal_advisor":
        _causal_brief(art, key)
    elif art.engine == "decomposition":
        _decomposition_body(art, key)
    else:
        if art.chart_df is not None:
            line_chart(art.chart_df)
        if art.table is not None and art.engine != "descriptive":
            st.dataframe(art.table, width="stretch",
                         height=min(320, 45 + 35 * len(art.table)))

    _divergence_block(art)
    _caveats_block(art)
    _provenance_block(art)
    render_stamp(art)
    _actions_row(art, key)
