"""Diagnostic decomposition: 'which segments account for the change, and how much each.'

Deterministic arithmetic on additive metrics — this is the checkable answer to
most of what people mean by 'why'. Contributions across each dimension sum
exactly to the total delta; the engine ranks dimensions by concentration.
"""
from __future__ import annotations

import pandas as pd

from .. import semantic_layer as sl
from ..provenance import AnswerArtifact, TIER_VERIFIED
from ..triage import BASIS_LABELS, Intent

_BASIS_STEPS = {"prior_month": 1, "prior_quarter": 3, "yoy": 12}


def _contributions(df: pd.DataFrame, col: str, dim: str, m0: str, m1: str) -> pd.DataFrame:
    a = df[df["month"] == m0].groupby(dim)[col].sum()
    b = df[df["month"] == m1].groupby(dim)[col].sum()
    out = pd.DataFrame({"period_start": a, "period_end": b}).fillna(0.0)
    out["delta"] = out["period_end"] - out["period_start"]
    total = out["delta"].sum()
    out["share_of_change"] = out["delta"] / total if total != 0 else 0.0
    return out.sort_values("delta").reset_index().rename(columns={dim: "value"}).assign(dimension=dim)


def decompose(intent: Intent, res: sl.Resolution) -> AnswerArtifact:
    df = sl.apply_filters(sl.load_fact(res.source), intent.filters)
    col = sl.column_for(res.metric, res.variant)
    ms = sorted(df["month"].unique())

    # anchor month: the window's end when one was asked for, else the latest
    anchor = intent.window.months[-1] if intent.window else ms[-1]
    if anchor not in ms:
        earlier = [m for m in ms if m <= anchor]
        anchor = earlier[-1] if earlier else ms[-1]
    basis = intent.compare_basis or "prior_quarter"
    steps = _BASIS_STEPS[basis]
    i1 = ms.index(anchor)
    i0 = max(0, i1 - steps)
    m0, m1 = ms[i0], anchor
    basis_label = BASIS_LABELS[basis] + \
        (" — clamped to available history" if i1 - steps < 0 else "")

    total0 = df[df["month"] == m0][col].sum()
    total1 = df[df["month"] == m1][col].sum()
    delta = total1 - total0

    # a dimension stays available for breakdown when its filter keeps >1 value
    dims = [d for d in sl.DIMENSIONS
            if d not in intent.filters or isinstance(intent.filters[d], (list, tuple))]
    tables = {dim: _contributions(df, col, dim, m0, m1) for dim in dims}
    label = sl.METRICS[res.metric]["variants"][res.variant]["label"]
    move = (f"{label} moved {delta:+,.1f} from {m0} to {m1} "
            f"({(delta/total0*100 if total0 else 0):+.1f}%, {basis_label})")

    if tables:
        # rank dimensions by concentration: largest single |contribution| / |total delta|
        conc = {dim: (t["delta"].abs().max() / abs(delta) if delta else 0)
                for dim, t in tables.items()}
        lead_dim = max(conc, key=conc.get)
        lead = tables[lead_dim].iloc[tables[lead_dim]["delta"].abs().idxmax()]
        headline = (f"{move}. Largest single contributor: {lead_dim} = {lead['value']} "
                    f"({lead['delta']:+,.1f}, {lead['share_of_change']*100:.0f}% of the change).")
        note = ("Decomposition shows where the change sits, not what caused it. "
                "When a registered event overlaps this window, the Causal Studio "
                "can test attribution.")
        table = pd.concat(tables.values(), ignore_index=True) \
            [["dimension", "value", "period_start", "period_end", "delta", "share_of_change"]]
    else:
        # every dimension is pinned to a single value: the total move IS the answer
        lead_dim = None
        headline = f"{move}."
        note = ("Every dimension in this scope is pinned to a single value, so there is "
                "no finer breakdown to show — the total move above is the whole story. "
                "Widen the scope to decompose it.")
        table = None

    code = (f"df = load('{res.source}'); df = filter(df, {intent.filters})\n"
            f"for dim in {list(tables)}:\n"
            f"    delta[dim] = pivot(df, '{col}', dim, '{m1}') - pivot(df, '{col}', dim, '{m0}')\n"
            f"# comparison basis: {basis_label}; contributions per dimension sum to the total delta")

    art = AnswerArtifact(intent.question, intent.question_class, TIER_VERIFIED, "decomposition",
                         headline=headline, value=float(delta), code=code, table=table)
    art.resolution = res
    art.extras = {"m0": m0, "m1": m1, "lead_dim": lead_dim, "tables": tables, "note": note}
    # link registered events that overlap this window, cover this metric, and
    # don't contradict the active filters
    overlapping = [{"id": eid, "name": ev["name"]}
                   for eid, ev in sl.EVENTS.items()
                   if ev["start"] <= m1
                   and res.metric in ev.get("metrics", [])
                   and _scope_compatible(ev["scope"], intent.filters)]
    if overlapping:
        art.extras["overlapping_events"] = overlapping
    return art


def _scope_compatible(event_scope: dict, filters: dict) -> bool:
    """An event is offered only when the analysis filters don't exclude it."""
    for dim, val in event_scope.items():
        if dim in filters:
            allowed = filters[dim] if isinstance(filters[dim], (list, tuple)) else [filters[dim]]
            if val not in allowed:
                return False
    return True
