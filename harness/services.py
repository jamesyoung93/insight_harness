"""Cross-cutting services: divergence detection, caveat blocks, monitoring, telemetry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import semantic_layer as sl
from .provenance import AnswerArtifact

FEEDBACK_LOG = Path(__file__).parent.parent / "data" / "feedback_log.jsonl"

# canonical wording per metric for machine-composed questions (drill-through,
# refusal reframes); every composed question must round-trip the parser
PRIMARY_KEYWORD = {"revenue": "revenue", "units": "units", "calls": "calls",
                   "new_customers": "new customers", "trx": "TRx", "nrx": "NRx",
                   "nbrx": "NBRx", "trx_share": "TRx market share",
                   "samples": "samples", "speaker_attendance": "speaker attendance",
                   "new_writers": "new writers"}


def breakdown_question(metric: str, filters: dict, window_n: int | None = None,
                       basis: str | None = None) -> str:
    """The canonical diagnostic question for a monitored scope. Drill-through
    asks a real question so the step lands in history and reproduces. Optional
    tile controls stay in the question text, keeping navigation reproducible."""
    kw = PRIMARY_KEYWORD.get(metric, "revenue")
    target = next((d for d in ("specialty", "payer_channel", "territory", "district",
                               "region", "segment", "channel") if d not in filters),
                  "specialty")
    target_label = {"specialty": "specialties", "payer_channel": "payer channels",
                    "territory": "territories", "district": "districts",
                    "region": "regions", "segment": "segments",
                    "channel": "channels"}[target]
    values: list = []
    for v in filters.values():
        values.extend(v if isinstance(v, (list, tuple)) else [v])
    where = f" in {' and '.join(values)}" if values else ""
    window = ""
    if window_n:
        unit = "month" if window_n == 1 else "months"
        window = f" over the last {window_n} {unit}"
    basis_text = {
        "prior_month": " vs prior month",
        "prior_quarter": " vs prior quarter",
        "yoy": " vs same month last year",
    }.get(basis, "")
    return f"Which {target_label} account for the {kw} change{where}{window}{basis_text}?"


# --------------------------------------------------------------------------- #
# Divergence: would the alternate source/variant move this answer materially?
# --------------------------------------------------------------------------- #
def check_divergence(art: AnswerArtifact, intent) -> None:
    """Recompute the artifact's own quantity under each registered alternate;
    flag material forks. The alternate must be computed the same way the
    answer was (level vs level, delta vs delta, same window) — a fork that
    can't be recomputed like-for-like is skipped, never approximated."""
    res = art.resolution
    if art.value is None or res is None or not res.alternates:
        return
    if art.engine not in ("descriptive", "decomposition"):
        # causal designs disclose variant forks via their computed sensitivity
        return
    base = art.value
    if base == 0:
        return
    window = getattr(intent, "window", None)
    for kind, alt_id in res.alternates:
        source = alt_id if kind == "source" else res.source
        variant = alt_id if kind == "variant" else res.variant
        note = ""
        try:
            df = sl.apply_filters(sl.load_fact(source), intent.filters)
            col = sl.column_for(res.metric, variant)
            months = sorted(df["month"].unique())
            if art.engine == "decomposition":
                m0, m1 = art.extras["m0"], art.extras["m1"]
                if m0 not in months:
                    continue  # alternate can't reproduce this comparison window
                if m1 not in months:
                    m1, note = months[-1], f"window clamped to {months[-1]} (reporting lag)"
                alt_val = float(sl.aggregate_metric(df[df["month"] == m1], res.metric, variant)
                                - sl.aggregate_metric(df[df["month"] == m0], res.metric, variant))
            elif window and not intent.trend:
                wmonths = [m for m in window.months if m in set(months)]
                if not wmonths:
                    continue
                if len(wmonths) < len(window.months):
                    note = "window clamped to available months (reporting lag)"
                alt_val = sl.aggregate_metric(df[df["month"].isin(wmonths)],
                                              res.metric, variant)
            else:
                target = window.months[-1] if window else months[-1]
                if target not in months:
                    earlier = [m for m in months if m <= target]
                    if not earlier:
                        continue
                    target = earlier[-1]
                alt_val = sl.aggregate_metric(df[df["month"] == target], res.metric, variant)
                if kind == "source" and sl.SOURCES[source]["lag_months"]:
                    note = f"latest available month is {target} (reporting lag)"
        except Exception:
            continue
        rel = (alt_val - base) / base
        art.divergence.append({
            "fork": f"{kind}: {alt_id}",
            "label": (sl.SOURCES[alt_id]["name"] if kind == "source"
                      else sl.METRICS[res.metric]["variants"][alt_id]["label"]),
            "value": alt_val, "rel_diff": rel,
            "material": abs(rel) > sl.materiality(), "note": note,
        })


# --------------------------------------------------------------------------- #
# Caveats: built from registry metadata, never from model musing
# --------------------------------------------------------------------------- #
def build_caveats(art: AnswerArtifact) -> None:
    res = art.resolution
    if res is None:
        return
    src = sl.SOURCES[res.source]
    art.caveats.extend(src["notes"][:2] if src["kind"] == "panel-projected" else [])
    variants = sl.METRICS[res.metric]["variants"]
    if len(variants) > 1:
        others = [v["label"] for k, v in variants.items() if k != res.variant]
        art.caveats.append(f"Named variants exist for this metric ({', '.join(others)}); "
                           f"answer shown uses the governed default unless overridden.")
    if any(d["material"] for d in art.divergence):
        art.caveats.append("A materially different answer exists under an alternate "
                           "source/definition — see 'Same question, different answer'.")


# --------------------------------------------------------------------------- #
# Watchlists: user-pinned metric+scope, evaluated with the same materiality
# logic as the anomaly feed
# --------------------------------------------------------------------------- #
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def load_watchlist() -> list:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def add_watch(metric: str, filters: dict, label: str) -> bool:
    """Pin a metric+scope. Returns False when it is already watched."""
    watches = load_watchlist()
    if any(w.get("metric") == metric and w.get("filters") == filters for w in watches):
        return False
    watches.append({"metric": metric, "filters": filters, "label": label,
                    "added": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    WATCHLIST_PATH.parent.mkdir(exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(watches, indent=2))
    return True


def remove_watch(metric: str, filters: dict) -> None:
    """Remove by identity (same equality add_watch dedupes on), so a stale or
    concurrently-modified list can never make Remove hit the wrong entry."""
    watches = [w for w in load_watchlist()
               if not (w.get("metric") == metric and w.get("filters") == filters)]
    WATCHLIST_PATH.parent.mkdir(exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(watches, indent=2))


def watch_feed(watches: list, z_threshold: float = 2.0) -> pd.DataFrame:
    """Evaluate each watched scope: latest month vs its trailing mean, flagged
    with the same z logic the anomaly feed uses."""
    rows = []
    for w in watches:
        metric = w.get("metric")
        filters = w.get("filters", {})
        if metric not in sl.METRICS:
            # keep the row visible so the watch can still be removed
            rows.append({"label": w.get("label") or str(metric), "metric": str(metric),
                         "scope": sl.scope_string(filters), "latest": None,
                         "trailing_mean": None, "z": None, "impact": None, "flagged": False,
                         "metric_id": metric, "filters": filters})
            continue
        col = sl.column_for(metric, sl.default_variant(metric))
        df = sl.apply_filters(sl.load_fact(sl.METRICS[metric]["default_source"]), filters)
        s = df.groupby("month")[col].sum().sort_index()
        scope = sl.scope_string(filters)
        label = w.get("label") or f"{sl.METRICS[metric]['label']} · {scope}"
        if len(s) < 5:
            rows.append({"label": label, "metric": sl.METRICS[metric]["label"], "scope": scope,
                         "latest": float(s.iloc[-1]) if len(s) else 0.0, "trailing_mean": None,
                         "z": None, "impact": None, "flagged": False,
                         "metric_id": metric, "filters": filters})
            continue
        hist = s.iloc[-7:-1]
        mu, sd = hist.mean(), hist.std()
        z = (s.iloc[-1] - mu) / sd if sd else 0.0
        rows.append({"label": label, "metric": sl.METRICS[metric]["label"], "scope": scope,
                     "latest": round(float(s.iloc[-1]), 1), "trailing_mean": round(float(mu), 1),
                     "z": round(float(z), 2), "impact": round(abs(float(s.iloc[-1] - mu)), 1),
                     "flagged": bool(abs(z) >= z_threshold),
                     "metric_id": metric, "filters": filters})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Monitoring: impact-ranked anomaly feed (materiality filter, not p-value spam)
# --------------------------------------------------------------------------- #
def anomaly_feed(z_threshold: float = 2.0) -> pd.DataFrame:
    rows = []
    df = sl.load_fact("source_a")
    latest = sorted(df["month"].unique())[-1]
    for metric in ("revenue", "units", "calls"):
        col = sl.column_for(metric, sl.default_variant(metric))
        for dim in ("region", "segment"):
            g = df.groupby(["month", dim])[col].sum().reset_index()
            for val in g[dim].unique():
                s = g[g[dim] == val].set_index("month")[col].sort_index()
                hist = s.iloc[-7:-1]
                if len(hist) < 4:
                    continue
                mu, sd = hist.mean(), hist.std()
                z = (s.iloc[-1] - mu) / sd if sd else 0.0
                if abs(z) >= z_threshold:
                    rows.append({"month": latest, "metric": sl.METRICS[metric]["label"],
                                 "scope": f"{dim}={val}", "latest": round(s.iloc[-1], 1),
                                 "trailing_mean": round(mu, 1), "z": round(z, 2),
                                 "impact": round(abs(s.iloc[-1] - mu), 1),
                                 "metric_id": metric, "dim": dim, "value": val})
    out = pd.DataFrame(rows)
    return out.sort_values("impact", ascending=False).reset_index(drop=True) if len(out) else out


# --------------------------------------------------------------------------- #
# Telemetry: corrections are the evidence base for scope expansion
# --------------------------------------------------------------------------- #
def log_feedback(art: AnswerArtifact, verdict: str, note: str = "") -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec = {"ts": ts,
           "question": art.question, "class": art.question_class, "tier": art.tier,
           "engine": art.engine, "result_hash": art.result_hash,
           "data_version": art.data_version, "verdict": verdict, "note": note}
    FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with FEEDBACK_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return ts


def log_review(art: AnswerArtifact) -> str:
    """Analyst sign-off on a Hypothesis-tier design is provenance too."""
    return log_feedback(art, "analyst_reviewed")


def feedback_history() -> pd.DataFrame:
    if not FEEDBACK_LOG.exists():
        return pd.DataFrame(columns=["ts", "question", "class", "tier", "verdict", "note"])
    rows = []
    for line in FEEDBACK_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # one truncated line must not take down the record
    return pd.DataFrame(rows)
