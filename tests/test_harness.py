"""Unit-level regression tests for review findings: metric clamping in causal
designs, LLM validation guardrails, engine-aware divergence, resolution
disclosure, event-overlap filtering, and telemetry robustness."""
import json

import pandas as pd

from harness import pipeline, services
from harness import semantic_layer as sl
from harness.llm_translator import _validate


def test_causal_question_on_unregistered_design_metric_answers_and_discloses():
    """'Why did calls drop in the West?' used to crash the brief render: the
    advisor clamped the metric but the resolution kept the asked one."""
    art = pipeline.answer("Why did calls drop in the West?")
    assert art.engine == "causal_advisor"
    assert art.resolution.metric == "revenue"  # what actually ran
    assert any("design runs on revenue" in c for c in art.caveats)
    # sensitivity keys must be resolvable against the resolution's metric
    for v in art.extras["sensitivity"]:
        assert v in sl.METRICS[art.resolution.metric]["variants"]


def test_llm_causal_intent_without_metric_defaults_to_revenue():
    raw = json.dumps({"question_class": "Causal", "metric": None, "filters": {},
                      "trend": False, "event_id": "east_program", "template": None,
                      "reason": ""})
    intent, meta = _validate("What was the impact of the east program?", raw)
    assert intent.metric == "revenue"
    art = pipeline.answer_intent(intent, translation=meta)
    assert art.engine == "causal_advisor"


def test_divergence_is_engine_aware():
    # descriptive: level vs level (the original behavior)
    desc = pipeline.answer("What is revenue in the South region?")
    assert any(d["material"] for d in desc.divergence)

    # decomposition: the alternate is recomputed as the same m0->m1 delta
    deco = pipeline.answer("Which segments account for the revenue change?")
    gross_forks = [d for d in deco.divergence if d["label"] == "Gross revenue"]
    assert gross_forks, "variant fork missing on decomposition"
    df = sl.load_fact("source_a")
    m0, m1 = deco.extras["m0"], deco.extras["m1"]
    expected = float(df[df["month"] == m1]["revenue_gross"].sum()
                     - df[df["month"] == m0]["revenue_gross"].sum())
    assert abs(gross_forks[0]["value"] - expected) < 1e-6

    # causal: no fabricated forks; variant sensitivity is computed in-engine
    causal = pipeline.answer("What was the impact of the partner enablement program in the East?")
    assert causal.divergence == []
    assert causal.extras["sensitivity"]


def test_resolve_disclosure_covers_clamped_overrides():
    r = sl.resolve("calls", None, "net")  # 'net' isn't registered for calls
    assert r.variant == "std"
    assert "not registered" in r.reason

    r = sl.resolve("calls", "source_b", None)  # calls exist in the warehouse only
    assert r.source == "source_a"
    assert "not registered" in r.reason

    r = sl.resolve("revenue", "source_b", "bogus")  # one valid + one clamped axis
    assert r.source == "source_b" and r.variant == "net"
    assert "override" in r.reason and "not registered" in r.reason


def test_event_overlap_respects_metric_and_scope():
    # calls: no registered event covers the metric
    art = pipeline.answer("Which regions account for the calls change?")
    assert "overlapping_events" not in art.extras
    # revenue in North: both events' scopes contradict the filter
    art = pipeline.answer("Which segments account for the revenue change in the North region?")
    assert "overlapping_events" not in art.extras
    # revenue unfiltered: both revenue events overlap
    art = pipeline.answer("Which segments account for the revenue change?")
    ids = {e["id"] for e in art.extras["overlapping_events"]}
    assert ids == {"east_program", "west_shock"}


def test_feedback_history_survives_a_corrupt_line(tmp_path, monkeypatch):
    log = tmp_path / "feedback_log.jsonl"
    good = {"ts": "2026-07-12T00:00:00+00:00", "question": "q", "class": "Descriptive",
            "tier": "Verified", "engine": "descriptive", "result_hash": "abc",
            "data_version": "v", "verdict": "correct", "note": ""}
    log.write_text(json.dumps(good) + "\n" + '{"ts": "2026-07-12T00:0' + "\n"
                   + json.dumps(good | {"verdict": "wrong"}) + "\n")
    monkeypatch.setattr(services, "FEEDBACK_LOG", log)
    hist = services.feedback_history()
    assert len(hist) == 2
    assert set(hist["verdict"]) == {"correct", "wrong"}


def test_artifact_json_is_complete_for_every_engine():
    for q in ("What is revenue in the West region?",
              "Which segments account for the revenue change?",
              "List whitespace accounts with no activity",
              "What was the impact of the partner enablement program in the East?",
              "Forecast revenue for next quarter"):
        d = json.loads(pipeline.answer(q).to_json())
        for k in ("question", "tier", "engine", "code", "resolution", "caveats",
                  "divergence", "extras", "data_version", "result_hash"):
            assert k in d, f"{k} missing from artifact JSON for {q!r}"
        assert "intent" in d["extras"]
