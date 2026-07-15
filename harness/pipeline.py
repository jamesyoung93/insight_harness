"""Pipeline orchestrator + evaluation harness.

pipeline.answer() is the single entry point: triage -> resolve -> engine ->
divergence -> caveats. The evaluation harness runs a golden question set with
ground truth computed through an INDEPENDENT code path, abstention tests
(refusing is a scored behavior), and reproducibility checks (same question
twice must hash identically).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import semantic_layer as sl
from . import services, tiles, triage
from .engines import basic, causal_advisor, decomposition
from .provenance import AnswerArtifact, TIER_ABSTAINED, _stable_hash

EVAL_HISTORY = Path(__file__).parent.parent / "data" / "eval_history.jsonl"


def answer(question: str, source: str | None = None, variant: str | None = None,
           api_key: str | None = None, model: str | None = None,
           basis: str | None = None) -> AnswerArtifact:
    translation = {"translator": "rules", "validated": True}
    intent = None
    if api_key:
        from . import llm_translator
        t0 = time.perf_counter()
        try:
            intent, translation = llm_translator.translate(
                question, api_key, model or llm_translator.DEFAULT_MODEL)
        except llm_translator.TranslationError as e:
            # product-voice caption; the raw error stays in the artifact JSON for audit
            reason = ("The language model's translation didn't validate against the metric "
                      "registry, so the built-in parser answered this question."
                      if e.kind == "rejected" else
                      "The language model translator wasn't available for this question, "
                      "so the built-in parser answered it.")
            translation = {"translator": "rules", "validated": True,
                           "fallback_reason": reason, "fallback_detail": str(e),
                           "fallback_kind": e.kind}
        translation["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    if intent is None:
        intent = triage.parse(question)
    if basis and intent.question_class in (triage.DIAGNOSTIC, triage.DESCRIPTIVE):
        intent.compare_basis = basis
    return answer_intent(intent, source, variant, translation)


def answer_intent(intent: triage.Intent, source: str | None = None,
                  variant: str | None = None, translation: dict | None = None) -> AnswerArtifact:
    """Resolve → engine → divergence → caveats for an already-parsed intent.
    Every answer surface (Ask, drill-through, Causal Studio) goes through here,
    so every answer is a full artifact."""
    translation = translation or {"translator": "rules", "validated": True}

    if intent.question_class in (triage.OUT_OF_SCOPE, triage.PREDICTIVE) or (
            intent.question_class == triage.CAUSAL and intent.event_id is None):
        art = AnswerArtifact(intent.question, intent.question_class, TIER_ABSTAINED, "abstention",
                             headline="Declined: " + intent.reason)
        art.data_version = sl.data_version()
        art.extras["intent"] = intent
        art.extras["translation"] = translation
        art.extras["reframes"] = _reframes(intent)
        return art

    res = sl.resolve(intent.metric or "revenue", source, variant)

    if intent.question_class == triage.CAUSAL:
        art = causal_advisor.propose(intent, res)
    elif intent.question_class == triage.DIAGNOSTIC:
        art = decomposition.decompose(intent, res)
    elif intent.question_class == triage.RETRIEVAL:
        art = basic.retrieval(intent, res)
    else:
        art = basic.descriptive(intent, res)

    if art.tier != TIER_ABSTAINED:  # an engine may refuse (e.g. uncovered window)
        services.check_divergence(art, intent)
        services.build_caveats(art)
    art.data_version = sl.data_version()
    art.extras["intent"] = intent
    art.extras["translation"] = translation
    return art


def _reframes(intent: triage.Intent) -> list[str]:
    """Concrete re-askable questions offered with a scoped refusal. Each must
    round-trip through the parser to a supported intent."""
    kw = services.PRIMARY_KEYWORD.get(intent.metric or "revenue", "revenue")
    if intent.question_class == triage.PREDICTIVE:
        return [f"Trend {kw} by month", f"Which segments account for the {kw} change?"]
    if intent.question_class == triage.CAUSAL:
        return [f"Which segments account for the {kw} change?", f"Trend {kw} by month"]
    return ["What is revenue in the West region?",
            "Which segments account for the revenue change?"]


# --------------------------------------------------------------------------- #
# Golden question set — ground truth via an independent code path
# --------------------------------------------------------------------------- #
def _gt_value(metric: str, filters: dict, months: list | None = None) -> float:
    """Independent recomputation: same registry lookups, different code path.
    Uses the governed default variant so the set stays valid under admin
    configuration changes."""
    df = sl.apply_filters(sl.load_fact("source_a"), filters)
    if months is None:
        months = [df["month"].max()]
    return sl.aggregate_metric(df[df["month"].isin(months)], metric, sl.default_variant(metric))


def _q_months(year: int, q: int) -> list:
    return [f"{year}-{m:02d}" for m in range(3 * q - 2, 3 * q + 1)]


# a fixed watch evaluated by the golden set — the West/Enterprise shock cell
_GOLDEN_WATCH = [{"metric": "revenue", "filters": {"region": "West", "segment": "Enterprise"},
                  "label": "Watched: West Enterprise revenue"}]

GOLDEN = [
    {"id": "G01", "question": "What is revenue in the West region?", "type": "value",
     "truth": lambda: _gt_value("revenue", {"region": "West"})},
    {"id": "G02", "question": "Total units for the Enterprise segment", "type": "value",
     "truth": lambda: _gt_value("units", {"segment": "Enterprise"})},
    {"id": "G03", "question": "Trend calls by month in the North region", "type": "value",
     "truth": lambda: _gt_value("calls", {"region": "North"})},
    {"id": "G04", "question": "Which segments account for the revenue change?", "type": "check",
     "check": lambda art: art.extras.get("lead_dim") in ("region", "segment")
     and any((t["value"].astype(str).str.contains("West").any() and t["delta"].min() < 0)
             for d, t in art.extras["tables"].items() if d == "region"),
     "label": "decomposition isolates the West decline"},
    {"id": "G05", "question": "List whitespace accounts with no activity in the East", "type": "check",
     "check": lambda art: art.table is not None and len(art.table) > 0
     and (art.table["decile"] >= 8).all() and (art.table["months_since_activity"] >= 3).all(),
     "label": "retrieval respects the whitespace criteria row-by-row"},
    {"id": "G06", "question": "What was the impact of the partner enablement program in the East?",
     "type": "causal",
     "truth_pct": lambda: sl.ground_truth()["events"]["east_program"]["true_effect_pct"],
     "tolerance": 0.03},
    {"id": "G07", "question": "Forecast revenue for next quarter", "type": "abstain"},
    {"id": "G08", "question": "Why is morale down this quarter?", "type": "abstain"},
    {"id": "G09", "question": "What is our customer happiness index?", "type": "abstain"},
    {"id": "G10", "question": "What is revenue in the South region?", "type": "divergence",
     "label": "source fork surfaced; material flags consistent with the threshold"},
    # windows
    {"id": "G11", "question": "What was revenue in Q1 2026 in the West region?", "type": "value",
     "truth": lambda: _gt_value("revenue", {"region": "West"}, _q_months(2026, 1))},
    {"id": "G12", "question": "Trend revenue last 6 months", "type": "check",
     "check": lambda art: art.chart_df is not None
     and art.chart_df["month"].tolist() == sl.months()[-6:],
     "label": "trend restricted to the asked window"},
    # multi-value filters
    {"id": "G13", "question": "What is revenue in the East and West regions?", "type": "value",
     "truth": lambda: _gt_value("revenue", {"region": ["East", "West"]})},
    # comparison basis
    {"id": "G14", "question": "Which segments account for the revenue change vs same month last year?",
     "type": "check",
     "check": lambda art: sl.months().index(art.extras["m1"])
     - sl.months().index(art.extras["m0"]) == 12
     and "same month last year" in art.headline,
     "label": "year-over-year basis honored and disclosed in the headline"},
    # watch alerts (evaluated at the monitoring slider's most sensitive setting)
    {"id": "G15", "question": "(watchlist) West Enterprise revenue watch", "type": "watch",
     "z": 1.5,
     "check": lambda feed: len(feed) == 1 and bool(feed.iloc[0]["flagged"])
     and float(feed.iloc[0]["z"]) < 0
     and all(feed["flagged"] == (feed["z"].abs() >= 1.5)),
     "label": "watched shock cell flags a decline, consistently with the threshold"},
    # abstentions for the new capabilities
    {"id": "G16", "question": "What was revenue in Q1 2023?", "type": "abstain"},
    {"id": "G17", "question": "What was our NPS over the last 6 months?", "type": "abstain"},
    # tile/Ask parity: the glanceable layer must be the same artifact, not a
    # separately implemented dashboard calculation
    {"id": "G18", "question": "(tile) TRx · Latest · MoM", "type": "tile",
     "tile_id": "trx", "window": "Latest", "basis": "MoM",
     "region": tiles.ALL_REGIONS},
    {"id": "G19", "question": "(tile) NRx · R3M · YoY · South", "type": "tile",
     "tile_id": "nrx", "window": "R3M", "basis": "YoY", "region": "South"},
    {"id": "G20", "question": "(tile) TRx market share · R6M · QoQ",
     "type": "tile", "tile_id": "trx_share", "window": "R6M",
     "basis": "QoQ", "region": tiles.ALL_REGIONS},
]


def run_golden(record: bool = True) -> pd.DataFrame:
    rows = []
    for g in GOLDEN:
        if g["type"] == "tile":
            kwargs = {"window": g["window"], "basis": g["basis"], "region": g["region"]}
            question = tiles.canonical_question(g["tile_id"], **kwargs)
            direct = answer_intent(tiles.intent_for(g["tile_id"], **kwargs))
            asked = answer(question)
            repeated = answer(question)
            same_resolution = direct.resolution == asked.resolution
            same_data = direct.data_version == asked.data_version == sl.data_version()
            same_hash = direct.result_hash == asked.result_hash
            rows.append({"id": g["id"], "question": question, "class": "Tile parity",
                         "tier": direct.tier,
                         "pass": bool(same_hash and same_resolution and same_data),
                         "reproducible": asked.result_hash == repeated.result_hash,
                         "detail": "tile and opened question share hash, resolution, and data"})
            continue
        if g["type"] == "watch":
            feed1 = services.watch_feed(_GOLDEN_WATCH, g["z"])
            feed2 = services.watch_feed(_GOLDEN_WATCH, g["z"])
            rows.append({"id": g["id"], "question": g["question"], "class": "Monitoring",
                         "tier": "Verified", "pass": bool(g["check"](feed1)),
                         "reproducible": _stable_hash(feed1) == _stable_hash(feed2),
                         "detail": g["label"]})
            continue
        art1 = answer(g["question"])
        art2 = answer(g["question"])
        reproducible = art1.result_hash == art2.result_hash
        if g["type"] == "value":
            truth = g["truth"]()
            ok = art1.value is not None and abs(art1.value - truth) / truth < 1e-6
            detail = f"expected {truth:,.1f}, got {art1.value:,.1f}" if art1.value is not None else "no value"
        elif g["type"] == "check":
            ok = bool(g["check"](art1)); detail = g["label"]
        elif g["type"] == "causal":
            truth = g["truth_pct"]()
            ok = art1.value is not None and abs(art1.value - truth) <= g["tolerance"]
            detail = (f"expected {truth*100:+.1f}%, DiD estimate "
                      f"{(art1.value or 0)*100:+.1f}% (tol ±{g['tolerance']*100:.0f}pp)")
        elif g["type"] == "abstain":
            ok = art1.tier == TIER_ABSTAINED; detail = "correct behavior is a scoped refusal"
        elif g["type"] == "divergence":
            ok = (len(art1.divergence) > 0
                  and any(d["fork"].startswith("source") for d in art1.divergence)
                  and all(d["material"] == (abs(d["rel_diff"]) > sl.materiality())
                          for d in art1.divergence))
            detail = g["label"]
        rows.append({"id": g["id"], "question": g["question"], "class": art1.question_class,
                     "tier": art1.tier, "pass": ok, "reproducible": reproducible, "detail": detail})
    res = pd.DataFrame(rows)
    if record:
        _record_run(res)
    return res


def _record_run(res: pd.DataFrame) -> None:
    refusals = res[res["tier"] == TIER_ABSTAINED]
    fb = services.feedback_history()
    votes = fb[fb["verdict"].isin(["correct", "wrong"])] if len(fb) else fb
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data_version": sl.data_version(),
           "pass_rate": float(res["pass"].mean()),
           "reproducible_rate": float(res["reproducible"].mean()),
           "correct_refusal_rate": float(refusals["pass"].mean()) if len(refusals) else 1.0,
           "correction_rate": float((votes["verdict"] == "wrong").mean()) if len(votes) else 0.0,
           "n": int(len(res))}
    EVAL_HISTORY.parent.mkdir(exist_ok=True)
    with EVAL_HISTORY.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def eval_history() -> pd.DataFrame:
    cols = ["ts", "data_version", "pass_rate", "reproducible_rate",
            "correct_refusal_rate", "correction_rate", "n"]
    if not EVAL_HISTORY.exists():
        return pd.DataFrame(columns=cols)
    rows = []
    for line in EVAL_HISTORY.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
