"""Causal design advisor: a design proposal engine, not an answer engine.

Given a causal question that matches a registered event, it proposes the
appropriate quasi-experimental design, COMPUTES the assumption checks
(pre-trend gap), runs the estimate, and computes sensitivity under the
alternate metric variant. Output is tiered as Hypothesis — requires analyst
validation. The 'arguments against' are calculated, never narrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import semantic_layer as sl
from ..provenance import AnswerArtifact, TIER_HYPOTHESIS
from ..triage import Intent

PRE_WINDOW = 6  # months used for the pre-period


def _monthly(df: pd.DataFrame, col: str, scope: dict) -> pd.Series:
    for k, v in scope.items():
        df = df[df[k] == v]
    return df.groupby("month")[col].sum().sort_index()


def _did(treated: pd.Series, control: pd.Series, start: str):
    months = sorted(set(treated.index) & set(control.index))
    pre = [m for m in months if m < start][-PRE_WINDOW:]
    post = [m for m in months if m >= start]
    t_pre, t_post = treated[pre].mean(), treated[post].mean()
    c_pre, c_post = control[pre].mean(), control[post].mean()
    naive_pct = (t_post - t_pre) / t_pre if t_pre else float("nan")
    control_pct = (c_post - c_pre) / c_pre if c_pre else float("nan")
    did_pct = naive_pct - control_pct  # ratio-based DiD: growth gap vs control
    did_abs = did_pct * t_pre
    # pre-trend check: slope gap between treated and control over the pre window
    x = np.arange(len(pre))
    slope_t = np.polyfit(x, treated[pre].values, 1)[0] / t_pre
    slope_c = np.polyfit(x, control[pre].values, 1)[0] / c_pre
    return {"pre": pre, "post": post, "did_abs": did_abs, "did_pct": did_pct,
            "naive_pct": naive_pct, "pretrend_gap_pp_per_month": (slope_t - slope_c) * 100,
            "t_pre": t_pre, "t_post": t_post, "c_pre": c_pre, "c_post": c_post}


DESIGN_METRICS = ("revenue", "units")  # metrics with registered attribution designs


def propose(intent: Intent, res: sl.Resolution) -> AnswerArtifact:
    ev = sl.EVENTS[intent.event_id]
    asked_metric = res.metric
    metric = asked_metric if asked_metric in DESIGN_METRICS else "revenue"
    if metric != asked_metric:
        # the design runs on a design-registered metric; the resolution must
        # describe what actually ran, and the substitution is disclosed below
        res = sl.resolve(metric)
    df = sl.load_fact(res.source)
    col = sl.column_for(metric, res.variant if res.variant in sl.METRICS[metric]["variants"] else
                        sl.METRICS[metric]["default_variant"])

    treated = _monthly(df, col, ev["scope"])
    ctrl_scope = {"region": ev["candidate_controls"]["region"]}
    cdf = df[df["region"].isin(ctrl_scope["region"])]
    control = cdf.groupby("month")[col].sum().sort_index()

    est = _did(treated, control, ev["start"])

    # computed sensitivity: does the conclusion hold under the alternate
    # variant, and under the alternate registered source?
    sens = {}
    for v in sl.METRICS[metric]["variants"]:
        alt_col = sl.column_for(metric, v)
        alt = _did(_monthly(df, alt_col, ev["scope"]),
                   df[df["region"].isin(ctrl_scope["region"])].groupby("month")[alt_col].sum().sort_index(),
                   ev["start"])
        sens[v] = alt["did_pct"]

    src_sens = {}
    for s in sl.METRICS[metric]["sources"]:
        try:
            sdf = sl.load_fact(s)
            alt = _did(_monthly(sdf, col, ev["scope"]),
                       sdf[sdf["region"].isin(ctrl_scope["region"])]
                       .groupby("month")[col].sum().sort_index(),
                       ev["start"])
            src_sens[s] = alt["did_pct"]
        except Exception:
            continue  # a source that can't reproduce the design is omitted, not faked

    pretrend_ok = abs(est["pretrend_gap_pp_per_month"]) < 0.5
    short_post = len(est["post"]) < 4
    fork_threshold = sl.materiality()  # the governed divergence threshold, in pp of effect

    checks = [
        {"check": "Parallel pre-trends (computed)",
         "result": f"slope gap {est['pretrend_gap_pp_per_month']:+.2f} pp/month over {len(est['pre'])} pre months",
         "status": "pass" if pretrend_ok else "flag"},
        {"check": "Post-period length",
         "result": f"{len(est['post'])} months since event",
         "status": "flag" if short_post else "pass"},
        {"check": "Sensitivity to metric variant (computed)",
         "result": " | ".join(f"{v}: {p*100:+.1f}%" for v, p in sens.items()),
         "status": "pass" if (max(sens.values()) - min(sens.values())) < fork_threshold else "flag"},
        {"check": "No concurrent event contaminating controls",
         "result": ev["notes"], "status": "manual"},
    ]
    if len(src_sens) > 1:
        checks.insert(3, {
            "check": "Sensitivity to source (computed)",
            "result": " | ".join(f"{sl.SOURCES[s]['name']}: {p*100:+.1f}%"
                                 for s, p in src_sens.items()),
            "status": "pass" if (max(src_sens.values()) - min(src_sens.values()))
            < fork_threshold else "flag"})

    chart = pd.DataFrame({"month": treated.index, "treated": treated.values}) \
        .merge(pd.DataFrame({"month": control.index, "control (avg-scaled)":
                             control.values * (est["t_pre"] / est["c_pre"])}), on="month")

    headline = (f"Design proposal: difference-in-differences for '{ev['name']}' on "
                f"{sl.METRICS[metric]['label'].lower()} — Hypothesis, pending analyst review")

    code = (f"treated = monthly(df, '{col}', {ev['scope']})\n"
            f"control = monthly(df, '{col}', {ctrl_scope})\n"
            f"DiD = (post(treated) - pre(treated)) - (post(control) - pre(control))\n"
            f"pre window = {PRE_WINDOW}m; assumption checks computed, not asserted")

    art = AnswerArtifact(intent.question, intent.question_class, TIER_HYPOTHESIS, "causal_advisor",
                         headline=headline, value=float(est["did_pct"]), code=code, chart_df=chart)
    art.resolution = res
    art.extras = {"event": ev, "estimate": est, "checks": checks, "sensitivity": sens,
                  "source_sensitivity": src_sens,
                  "note": ("Causal estimates answer a narrow, designed version of 'why'. "
                           "This one is publishable only after an analyst signs off on "
                           "the checks above.")}
    if metric != asked_metric and asked_metric in sl.METRICS:
        art.caveats.append(f"The question referenced {sl.METRICS[asked_metric]['label'].lower()}; "
                           f"attribution designs are registered for revenue and units, so this "
                           f"design runs on {sl.METRICS[metric]['label'].lower()}.")
    if short_post:
        art.caveats.append("Short post-period: estimate will move as more months arrive.")
    return art
