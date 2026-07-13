"""Semantic layer: the governed foundation, browsable by everyone."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from harness import semantic_layer as sl


def render() -> None:
    st.title("Semantic layer")
    st.caption("The governed foundation every answer is built on: one definition per metric, "
               "with named variants that have owners — not spreadsheet folklore.")
    t1, t2 = st.tabs(["Metric registry", "Source registry"])
    with t1:
        for mid, m in sl.METRICS.items():
            default = sl.default_variant(mid)
            marker = "" if default == m["default_variant"] else " (set by governance)"
            src_name = sl.SOURCES[m["default_source"]]["name"]
            with st.expander(f"{m['label']}  ·  default: {default}{marker} on {src_name}",
                             expanded=(mid == "revenue")):
                rows = [{"variant": k, "label": v["label"], "owner": v["owner"], "notes": v["notes"],
                         "default": "✓" if k == default else ""}
                        for k, v in m["variants"].items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(f"Available on: {', '.join(sl.SOURCES[s]['name'] for s in m['sources'])} · "
                           f"materiality threshold for divergence flags: {sl.materiality()*100:.1f}%")
    with t2:
        for sid, s in sl.SOURCES.items():
            with st.expander(s["name"], expanded=True):
                st.markdown(f"kind: `{s['kind']}` · cadence: {s['cadence']} · lag: {s['lag_months']} month(s)")
                for n in s["notes"]:
                    st.markdown(f"- {n}")

    st.divider()
    with st.expander("Administration — governance settings"):
        st.caption("These settings change the governed defaults for every user. Every "
                   "change is logged: governance changes are provenance too.")
        mat = st.number_input("Materiality threshold for divergence flags (%)",
                              min_value=0.5, max_value=20.0,
                              value=min(20.0, max(0.5, float(sl.materiality() * 100))),
                              step=0.5)
        multi = {mid: m for mid, m in sl.METRICS.items() if len(m["variants"]) > 1}
        picks = {}
        cols = st.columns(max(len(multi), 1))
        for col, (mid, m) in zip(cols, multi.items()):
            opts = list(m["variants"])
            picks[mid] = col.selectbox(f"{m['label']} default variant", opts,
                                       index=opts.index(sl.default_variant(mid)),
                                       key=f"adm_{mid}")
        if st.button("Apply governance changes"):
            change = sl.set_governance(materiality_rel=mat / 100, default_variants=picks)
            if change:
                st.toast("Applied and logged.")
                st.rerun()
            else:
                st.toast("Nothing changed.")
        log = sl.governance_log()
        if log:
            st.markdown("**Change log**")
            for rec in reversed(log[-10:]):
                ch = rec.get("change", {})
                bits = []
                if "materiality_rel" in ch:
                    bits.append(f"materiality threshold set to {ch['materiality_rel']*100:.1f}%")
                for m, v in ch.get("default_variants", {}).items():
                    label = sl.METRICS[m]["label"] if m in sl.METRICS else m
                    bits.append(f"{label} default variant set to {v}")
                st.markdown(f"- {rec.get('ts', '')} — {'; '.join(bits)}")
        else:
            st.caption("No governance changes recorded.")
