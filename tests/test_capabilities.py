"""Capability tests: time windows, multi-value filters, comparison bases,
watchlists, governance configuration, run history, and translator polish."""
import json

import pytest

from harness import pipeline, services, triage
from harness import semantic_layer as sl
from harness.llm_translator import TranslationError, _validate


# --------------------------------------------------------------------------- #
# Time windows
# --------------------------------------------------------------------------- #
def test_quarter_window_aggregates_and_discloses():
    art = pipeline.answer("What was revenue in Q1 2026 in the West region?")
    assert "Q1 2026" in art.headline
    df = sl.apply_filters(sl.load_fact("source_a"), {"region": "West"})
    expected = float(df[df["month"].isin(["2026-01", "2026-02", "2026-03"])]
                     [sl.column_for("revenue", sl.default_variant("revenue"))].sum())
    assert abs(art.value - expected) < 1e-6


def test_last_n_window_restricts_trend():
    art = pipeline.answer("Trend revenue last 6 months")
    assert art.chart_df["month"].tolist() == sl.months()[-6:]
    assert "last 6 months" in art.headline


def test_out_of_range_window_is_a_scoped_refusal():
    art = pipeline.answer("What was revenue in Q1 2023?")
    assert art.tier == "Abstained"
    assert "outside that range" in art.headline
    art = pipeline.answer("What was revenue in March 2023 in the East?")
    assert art.tier == "Abstained"


def test_window_reclamps_against_the_resolved_source():
    """A window valid on the default calendar may be uncovered on an
    overridden source — refuse or disclose, never report a wrong number."""
    # June 2026 doesn't exist on the lagged panel feed: scoped refusal
    art = pipeline.answer("What was revenue in June 2026?", source="source_b")
    assert art.tier == "Abstained"
    assert "External panel feed" in art.headline
    # Q2 2026 is partially covered there: clamped and disclosed
    art = pipeline.answer("What was revenue in Q2 2026?", source="source_b")
    assert art.tier == "Verified"
    assert "partial" in art.headline
    assert any("clamped" in c for c in art.caveats)
    # trend flavor of the same refusal must not crash either
    art = pipeline.answer("Trend revenue by month for June 2026", source="source_b")
    assert art.tier == "Abstained"


def test_fully_pinned_scope_decomposes_to_total_only():
    q = "Which segments account for the revenue change in West and Enterprise and Direct?"
    art = pipeline.answer(q)
    assert art.tier == "Verified" and art.engine == "decomposition"
    assert art.extras["tables"] == {}
    assert "no finer breakdown" in art.extras["note"]


def test_window_plus_basis_is_computed_or_disclosed():
    # single-month window: the comparison is computed against the shifted month
    art = pipeline.answer("What was revenue in May 2026 vs prior month?")
    assert "vs prior month" in art.headline
    # multi-month window: the comparison is omitted with a computed caveat
    art = pipeline.answer("What was revenue in Q1 2026 vs prior month?")
    assert any("comparison is omitted" in c for c in art.caveats)


def test_anomaly_feed_follows_governed_default_variant():
    base = services.anomaly_feed(1.5)
    sl.set_governance(default_variants={"revenue": "gross"})
    changed = services.anomaly_feed(1.5)
    rev_base = base[base["metric"] == "Revenue"]["latest"].sum()
    rev_changed = changed[changed["metric"] == "Revenue"]["latest"].sum()
    if len(base[base["metric"] == "Revenue"]) and len(changed[changed["metric"] == "Revenue"]):
        assert rev_changed != rev_base  # gross ≠ net


def test_llm_refusal_reason_is_never_model_authored():
    raw = _raw(question_class="Predictive", metric=None,
               reason="Based on the trend, revenue will likely reach 999 next quarter.")
    intent, meta = _validate("Forecast revenue", raw)
    assert "999" not in intent.reason  # the model's guess is not rendered
    assert intent.reason == triage.refusal_reason(triage.PREDICTIVE)
    assert "999" in meta["model_reason"]  # but it stays auditable


def test_oversized_last_n_clamps_with_disclosure():
    n_avail = len(sl.months())
    art = pipeline.answer(f"Trend revenue last {n_avail + 10} months")
    assert art.chart_df["month"].tolist() == sl.months()
    assert "available" in art.headline


def test_partial_quarter_disclosed():
    # data ends 2026-06, so Q3 2026 has no months; Q3 2024 starts mid-quarter
    w, refusal = triage.resolve_window({"kind": "quarter", "q": 3, "year": 2024})
    assert refusal is None and w.months == ["2024-07", "2024-08", "2024-09"]
    w, refusal = triage.resolve_window({"kind": "quarter", "q": 3, "year": 2026})
    assert w is None and "outside that range" in refusal


# --------------------------------------------------------------------------- #
# Multi-value filters
# --------------------------------------------------------------------------- #
def test_multi_value_filter_parses_and_computes():
    intent = triage.parse("What is revenue in the East and West regions?")
    assert intent.filters == {"region": ["East", "West"]}
    art = pipeline.answer_intent(intent)
    df = sl.load_fact("source_a")
    latest = sorted(df["month"].unique())[-1]
    expected = float(df[(df["region"].isin(["East", "West"])) & (df["month"] == latest)]
                     [sl.column_for("revenue", sl.default_variant("revenue"))].sum())
    assert abs(art.value - expected) < 1e-6
    assert "region in [East, West]" in art.headline


def test_multi_value_filtered_dim_stays_available_for_breakdown():
    art = pipeline.answer("Which regions account for the revenue change in the East and West regions?")
    assert "region" in art.extras["tables"]
    assert set(art.extras["tables"]["region"]["value"]) == {"East", "West"}


# --------------------------------------------------------------------------- #
# Comparison basis
# --------------------------------------------------------------------------- #
def test_basis_parses_and_sets_window_months():
    art = pipeline.answer("Which segments account for the revenue change vs prior month?")
    ms = sl.months()
    assert ms.index(art.extras["m1"]) - ms.index(art.extras["m0"]) == 1
    assert "vs prior month" in art.headline


def test_basis_override_via_pipeline_param():
    art = pipeline.answer("Which segments account for the revenue change?", basis="yoy")
    ms = sl.months()
    assert ms.index(art.extras["m1"]) - ms.index(art.extras["m0"]) == 12


def test_descriptive_basis_appends_comparison():
    art = pipeline.answer("What is revenue in the West region vs last year?")
    assert "vs same month last year" in art.headline


def test_trend_yoy_emits_aligned_reference_and_latest_comparison():
    art = pipeline.answer("Trend revenue last 6 months vs same month last year")
    current_col = sl.METRICS["revenue"]["variants"][sl.default_variant("revenue")]["label"]
    reference_cols = [c for c in art.chart_df.columns if c not in ("month", current_col)]
    assert len(reference_cols) == 1
    reference_col = reference_cols[0]

    df = sl.load_fact("source_a")
    value_col = sl.column_for("revenue", sl.default_variant("revenue"))
    monthly = df.groupby("month")[value_col].sum()
    months = sl.months()
    for row in art.chart_df.itertuples(index=False, name=None):
        month, current, reference = row
        reference_month = months[months.index(month) - 12]
        assert current == pytest.approx(float(monthly[month]))
        assert reference == pytest.approx(float(monthly[reference_month]))

    comparison = art.extras["comparison"]
    latest_month = art.chart_df.iloc[-1]["month"]
    reference_month = months[months.index(latest_month) - 12]
    assert comparison["basis"] == "yoy"
    assert comparison["basis_label"] == "vs same month last year"
    assert comparison["available"] is True
    assert comparison["current_month"] == latest_month
    assert comparison["reference_month"] == reference_month
    assert comparison["current_value"] == pytest.approx(float(monthly[latest_month]))
    assert comparison["reference_value"] == pytest.approx(float(monthly[reference_month]))
    assert comparison["delta"] == pytest.approx(
        float(monthly[latest_month] - monthly[reference_month]))
    assert comparison["delta_pct"] == pytest.approx(
        float((monthly[latest_month] - monthly[reference_month])
              / monthly[reference_month]))


@pytest.mark.parametrize("phrase,basis,steps", [
    ("vs prior month", "prior_month", 1),
    ("vs prior quarter", "prior_quarter", 3),
    ("vs same month last year", "yoy", 12),
])
def test_descriptive_point_exposes_structured_comparison(phrase, basis, steps):
    art = pipeline.answer(f"What was revenue in May 2026 {phrase}?")
    comparison = art.extras["comparison"]
    months = sl.months()
    reference_month = months[months.index("2026-05") - steps]
    df = sl.load_fact("source_a")
    col = sl.column_for("revenue", sl.default_variant("revenue"))
    monthly = df.groupby("month")[col].sum()

    assert comparison["basis"] == basis
    assert comparison["available"] is True
    assert comparison["current_month"] == "2026-05"
    assert comparison["current_value"] == pytest.approx(art.value)
    assert comparison["reference_month"] == reference_month
    assert comparison["reference_value"] == pytest.approx(float(monthly[reference_month]))
    assert comparison["delta"] == pytest.approx(
        comparison["current_value"] - comparison["reference_value"])
    assert comparison["delta_pct"] == pytest.approx(
        comparison["delta"] / comparison["reference_value"])


def test_trend_comparison_discloses_insufficient_reference_history():
    art = pipeline.answer("Trend revenue by month in Q1 2025 vs same month last year")
    reference_cols = [c for c in art.chart_df.columns if c not in ("month", "Net revenue")]
    assert len(reference_cols) == 1
    assert art.chart_df[reference_cols[0]].isna().all()
    comparison = art.extras["comparison"]
    assert comparison["basis"] == "yoy"
    assert comparison["current_month"] == "2025-03"
    assert comparison["available"] is False
    assert comparison["reference_month"] is None
    assert comparison["reference_value"] is None
    assert any("predates the available history" in caveat
               and "latest-point comparison is unavailable" in caveat
               for caveat in art.caveats)


# --------------------------------------------------------------------------- #
# LLM translation contract for the new capabilities
# --------------------------------------------------------------------------- #
def _raw(**kw):
    base = {"question_class": "Descriptive", "metric": "revenue", "filters": {},
            "trend": False, "event_id": None, "template": None, "reason": ""}
    base.update(kw)
    return json.dumps(base)


def test_llm_window_validated_and_resolved():
    intent, _ = _validate("q", _raw(window={"kind": "quarter", "q": 1, "year": 2026}))
    assert intent.window.months == ["2026-01", "2026-02", "2026-03"]


def test_llm_out_of_range_window_becomes_refusal_not_error():
    intent, meta = _validate("q", _raw(window={"kind": "quarter", "q": 1, "year": 2023}))
    assert intent.question_class == triage.OUT_OF_SCOPE
    assert meta["translator"] == "llm"


def test_llm_invalid_window_and_basis_rejected():
    with pytest.raises(TranslationError) as e:
        _validate("q", _raw(window={"kind": "fortnight"}))
    assert e.value.kind == "rejected"
    with pytest.raises(TranslationError) as e:
        _validate("q", _raw(compare_basis="vs_vibes"))
    assert e.value.kind == "rejected"


def test_llm_array_filters_validated():
    intent, _ = _validate("q", _raw(filters={"region": ["West", "East"]}))
    assert intent.filters == {"region": ["East", "West"]}  # sorted, deterministic
    with pytest.raises(TranslationError):
        _validate("q", _raw(filters={"region": ["East", "Atlantis"]}))
    with pytest.raises(TranslationError) as e:  # unhashable values must reject, not crash
        _validate("q", _raw(filters={"region": [{"x": 1}]}))
    assert e.value.kind == "rejected"


def test_translator_fallback_meta_without_network(monkeypatch):
    from harness import llm_translator

    def boom(*a, **kw):
        raise TranslationError("nope", kind="rejected")
    monkeypatch.setattr(llm_translator, "translate", boom)
    art = pipeline.answer("What is revenue in the West region?", api_key="test-key")
    tr = art.extras["translation"]
    assert tr["fallback_kind"] == "rejected"
    assert isinstance(tr["latency_ms"], int)
    assert tr["translator"] == "rules"  # visible fallback


# --------------------------------------------------------------------------- #
# Watchlists
# --------------------------------------------------------------------------- #
def test_watchlist_add_dedupe_remove_and_feed():
    assert services.add_watch("revenue", {"region": "West"}, "Revenue · region=West")
    assert not services.add_watch("revenue", {"region": "West"}, "dup")
    assert services.add_watch("calls", {}, "Sales calls · all scopes")
    feed = services.watch_feed(services.load_watchlist(), 1.5)
    assert len(feed) == 2
    assert all(feed["flagged"] == (feed["z"].abs() >= 1.5))
    services.remove_watch("revenue", {"region": "West"})  # remove by identity
    remaining = services.load_watchlist()
    assert len(remaining) == 1 and remaining[0]["metric"] == "calls"


def test_stale_watch_stays_visible_and_removable():
    services.add_watch("revenue", {"region": "West"}, "ok")
    watches = services.load_watchlist()
    watches.append({"metric": "retired_metric", "filters": {}, "label": "old"})
    services.WATCHLIST_PATH.write_text(json.dumps(watches))
    feed = services.watch_feed(services.load_watchlist(), 2.0)
    assert len(feed) == 2  # the stale watch is visible, not silently hidden
    services.remove_watch("retired_metric", {})
    assert len(services.load_watchlist()) == 1


# --------------------------------------------------------------------------- #
# Governance administration
# --------------------------------------------------------------------------- #
def test_governance_changes_apply_log_and_keep_golden_green():
    # raising materiality makes the ~3% South source fork immaterial, while
    # the much larger gross-vs-net variant fork rightly stays flagged
    def source_forks(art):
        return [d for d in art.divergence if d["fork"].startswith("source")]

    art = pipeline.answer("What is revenue in the South region?")
    assert any(d["material"] for d in source_forks(art))
    sl.set_governance(materiality_rel=0.10)
    art = pipeline.answer("What is revenue in the South region?")
    assert source_forks(art) and not any(d["material"] for d in source_forks(art))

    # flipping the default variant changes resolution and stays disclosed
    sl.set_governance(default_variants={"revenue": "gross"})
    art = pipeline.answer("What is revenue in the West region?")
    assert art.resolution.variant == "gross"
    assert "Gross revenue" in art.headline

    # both changes are logged, and the accuracy record stays green under them
    log = sl.governance_log()
    assert len(log) == 2 and all("ts" in r and "change" in r for r in log)
    res = pipeline.run_golden(record=False)
    assert res["pass"].all() and res["reproducible"].all()


def test_run_golden_records_history():
    pipeline.run_golden(record=True)
    pipeline.run_golden(record=True)
    hist = pipeline.eval_history()
    assert len(hist) == 2
    assert (hist["pass_rate"] == 1.0).all()
    assert set(hist.columns) >= {"ts", "data_version", "pass_rate",
                                 "reproducible_rate", "correct_refusal_rate"}
