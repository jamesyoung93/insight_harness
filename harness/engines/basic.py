"""Deterministic engines: descriptive aggregation/trend and retrieval templates."""
from __future__ import annotations

from .. import semantic_layer as sl
from .. import services
from ..provenance import AnswerArtifact, TIER_ABSTAINED, TIER_VERIFIED
from ..triage import BASIS_LABELS, Intent

_BASIS_STEPS = {"prior_month": 1, "prior_quarter": 3, "yoy": 12}


def _window_refusal(intent: Intent, res: sl.Resolution, src_months: list) -> AnswerArtifact:
    """The resolved source has no data for the asked window. Refuse rather
    than report a silently wrong number for a period the source can't cover."""
    src = sl.SOURCES[res.source]["name"]
    art = AnswerArtifact(intent.question, intent.question_class, TIER_ABSTAINED, "abstention",
                         headline=f"Declined: {src} has no data for {intent.window.label}; "
                                  f"it covers {src_months[0]} through {src_months[-1]}.")
    art.resolution = res
    kw = services.PRIMARY_KEYWORD.get(res.metric, "revenue")
    art.extras["reframes"] = [f"Trend {kw} by month", f"What is {kw} in the West region?"]
    return art


def descriptive(intent: Intent, res: sl.Resolution) -> AnswerArtifact:
    df = sl.load_fact(res.source)
    col = sl.column_for(res.metric, res.variant)
    fdf = sl.apply_filters(df, intent.filters)
    label = sl.METRICS[res.metric]["variants"][res.variant]["label"]
    scope = sl.scope_string(intent.filters)
    src_months = sorted(df["month"].unique())
    caveats: list[str] = []

    # a window is validated against the default calendar at parse time; the
    # resolved source may cover less, so re-clamp here and disclose the clamp
    w = intent.window
    wmonths, wlabel = None, None
    if w:
        wmonths = [m for m in w.months if m in src_months]
        if not wmonths:
            return _window_refusal(intent, res, src_months)
        wlabel = w.label
        if len(wmonths) < len(w.months):
            wlabel = f"{w.label} (partial: {wmonths[0]}–{wmonths[-1]} available here)"
            caveats.append(f"{sl.SOURCES[res.source]['name']} covers {src_months[0]} through "
                           f"{src_months[-1]}, so the window was clamped to the months it has.")

    if intent.trend:
        out = fdf.groupby("month", as_index=False)[col].sum().rename(columns={col: label})
        if wmonths:
            out = out[out["month"].isin(wmonths)].reset_index(drop=True)
        latest = out.iloc[-1]
        if intent.compare_basis:
            caveats.append("The trend already shows the measured history, so the requested "
                           "comparison basis is omitted.")
        code = (f"df = load('{res.source}')\n"
                f"df = filter(df, {intent.filters})\n"
                + (f"df = df[df.month.isin({wmonths})]\n" if wmonths else "")
                + f"trend = df.groupby('month')['{col}'].sum()")
        art = AnswerArtifact(intent.question, intent.question_class, TIER_VERIFIED, "descriptive",
                             headline=f"{label} ({scope}), {wlabel + ', ' if wlabel else ''}"
                                      f"latest month {latest['month']}: {latest[label]:,.1f}",
                             value=float(latest[label]), chart_df=out, table=out, code=code)
    elif w:
        val = float(fdf[fdf["month"].isin(wmonths)][col].sum())
        headline = f"{label} ({scope}), {wlabel}: {val:,.1f}"
        if intent.compare_basis:
            if len(wmonths) == 1:
                steps = _BASIS_STEPS[intent.compare_basis]
                i = src_months.index(wmonths[0])
                prior = float(fdf.loc[fdf["month"] == src_months[i - steps], col].sum()) \
                    if i - steps >= 0 else 0.0
                if i - steps >= 0 and prior:
                    headline += (f" ({(val - prior) / prior * 100:+.1f}% "
                                 f"{BASIS_LABELS[intent.compare_basis]}, "
                                 f"{src_months[i - steps]}: {prior:,.1f})")
                else:
                    caveats.append("The requested comparison month predates the available "
                                   "history, so the comparison is omitted.")
            else:
                caveats.append("A comparison basis applies to a single anchor month; this "
                               "answer aggregates a multi-month window, so the requested "
                               "comparison is omitted.")
        code = (f"df = load('{res.source}')\n"
                f"df = filter(df, {intent.filters})\n"
                f"value = df[df.month.isin({wmonths})]['{col}'].sum()")
        art = AnswerArtifact(intent.question, intent.question_class, TIER_VERIFIED, "descriptive",
                             headline=headline, value=val, code=code)
    else:
        ms = sorted(fdf["month"].unique())
        latest_month = ms[-1]
        val = float(fdf.loc[fdf["month"] == latest_month, col].sum())
        headline = f"{label} ({scope}), {latest_month}: {val:,.1f}"
        if intent.compare_basis:
            steps = _BASIS_STEPS[intent.compare_basis]
            if len(ms) > steps:
                m_prior = ms[-1 - steps]
                prior = float(fdf.loc[fdf["month"] == m_prior, col].sum())
                if prior:
                    headline += (f" ({(val - prior) / prior * 100:+.1f}% "
                                 f"{BASIS_LABELS[intent.compare_basis]}, {m_prior}: {prior:,.1f})")
            else:
                caveats.append("The requested comparison month predates the available "
                               "history, so the comparison is omitted.")
        code = (f"df = load('{res.source}')\n"
                f"df = filter(df, {intent.filters})\n"
                f"value = df[df.month == '{latest_month}']['{col}'].sum()")
        art = AnswerArtifact(intent.question, intent.question_class, TIER_VERIFIED, "descriptive",
                             headline=headline, value=val, code=code)
    art.resolution = res
    art.caveats.extend(caveats)
    return art


def retrieval(intent: Intent, res: sl.Resolution) -> AnswerArtifact:
    acc = sl.apply_filters(sl.load_accounts(), intent.filters)

    if intent.template == "whitespace":
        out = acc[(acc["decile"] >= 8) & (acc["months_since_activity"] >= 3)] \
            .sort_values("revenue_ttm", ascending=False) \
            [["account_id", "name", "region", "segment", "revenue_ttm", "decile",
              "months_since_activity", "calls_90d"]].reset_index(drop=True)
        code = ("acc = load('accounts')  # account-grain source only\n"
                f"acc = filter(acc, {intent.filters})\n"
                "out = acc[(acc.decile >= 8) & (acc.months_since_activity >= 3)]")
        headline = (f"{len(out)} whitespace accounts: decile 8+ on trailing-twelve-month revenue "
                    f"with no activity in 3+ months")
    else:
        out = acc.sort_values("revenue_ttm", ascending=False).head(15) \
            [["account_id", "name", "region", "segment", "revenue_ttm", "decile", "calls_90d"]] \
            .reset_index(drop=True)
        code = ("acc = load('accounts')\n"
                f"acc = filter(acc, {intent.filters})\n"
                "out = acc.nlargest(15, 'revenue_ttm')")
        headline = f"Top {len(out)} accounts by trailing-twelve-month revenue"

    art = AnswerArtifact(intent.question, intent.question_class, TIER_VERIFIED, "retrieval",
                         headline=headline, table=out, code=code)
    art.resolution = res
    # computed from the source registry's grain metadata
    with_grain = [s["name"] for s in sl.SOURCES.values() if s.get("account_grain")]
    without_grain = [s["name"] for s in sl.SOURCES.values() if not s.get("account_grain")]
    if without_grain:
        art.caveats.append(f"Account-level data exists in {', '.join(with_grain)} only; "
                           f"{', '.join(without_grain)} has no account grain.")
    return art
