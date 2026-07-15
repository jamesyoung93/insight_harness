"""Headless UI suite via streamlit.testing.v1.AppTest.

Covers: every page renders; every question class round-trips; refusals are
scoped with clickable reframes; drill-through navigation works end to end;
downloads exist on answers; history replays; analyst sign-off records; the
accuracy check passes through the UI.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from harness import pipeline, profiles, services, tiles

REPO = Path(__file__).parent.parent
APP = str(REPO / "app.py")

PAGES = ["Home", "Monitoring", "Causal Studio", "Semantic Layer", "Reliability",
         "How answers are produced"]


def app(timeout=30):
    return AppTest.from_file(APP, default_timeout=timeout)


def ask(at, q):
    box = next(t for t in at.text_input if t.label == "Question")
    box.set_value(q)
    return at.run()


def rendered_text(at):
    parts = [m.value for m in at.markdown]
    parts += [s.value for s in at.subheader]
    parts += [c.value for c in at.caption]
    parts += [i.value for i in at.info]
    parts += [s.value for s in at.success]
    return " ".join(parts)


def button_labeled(at, label):
    matches = [b for b in at.button if label in (b.label or "")]
    assert matches, f"no button labeled {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Rendering and round-trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = app().run()
    at.radio(key="nav").set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"


def test_home_shows_live_kpis_before_a_question_is_asked():
    at = app().run()
    values = {metric.label: metric.value for metric in at.metric}
    persona = profiles.PERSONAS_BY_ID["executive"]
    expected = {tiles.TILES_BY_ID[tile_id].label for tile_id in persona.default_tile_ids}
    assert expected <= set(values)
    assert all(value not in (None, "—") for value in values.values())
    charts = at.get("vega_lite_chart") or at.get("arrow_vega_lite_chart")
    assert len(charts) >= len(expected)
    assert at.selectbox(key="home_persona").value == "executive"
    assert at.radio(key="home_window").value == persona.default_window
    assert at.radio(key="home_basis").value == persona.default_basis
    assert at.selectbox(key="home_region").value == persona.default_scope
    assert next(t for t in at.text_input if t.label == "Question").value == ""


def test_persona_change_applies_scope_controls_and_ordered_tile_defaults():
    at = app().run()
    at.selectbox(key="home_persona").set_value("sales_rep").run()
    persona = profiles.PERSONAS_BY_ID["sales_rep"]
    assert at.radio(key="home_window").value == persona.default_window
    assert at.radio(key="home_basis").value == persona.default_basis
    assert at.selectbox(key="home_region").value == persona.default_scope
    labels = [metric.label.removesuffix(" · West") for metric in at.metric]
    expected = [tiles.TILES_BY_ID[tile_id].label for tile_id in persona.default_tile_ids]
    assert labels == expected


def test_tile_customization_is_session_local_reorderable_and_resettable():
    at = app().run()
    assert "Details" not in [metric.label for metric in at.metric]
    at.selectbox(key="home_executive_add_choice").set_value("calls").run()
    at.button(key="home_executive_add").click().run()
    assert "Details" in [metric.label for metric in at.metric]

    at.button(key="home_executive_calls_up").click().run()
    labels = [metric.label for metric in at.metric]
    assert labels[-2:] == ["Details", "Commercial TRx"]

    at.radio(key="home_window").set_value("R3M").run()
    at.button(key="home_executive_save").click().run()
    at.selectbox(key="home_persona").set_value("sales_rep").run()
    at.selectbox(key="home_persona").set_value("executive").run()
    assert at.radio(key="home_window").value == "R3M"
    assert "Details" in [metric.label for metric in at.metric]

    at.button(key="home_executive_reset").click().run()
    persona = profiles.PERSONAS_BY_ID["executive"]
    assert at.radio(key="home_window").value == persona.default_window
    assert at.radio(key="home_basis").value == persona.default_basis
    assert "Details" not in [metric.label for metric in at.metric]

    fresh = app().run()
    assert "Details" not in [metric.label for metric in fresh.metric]


def test_tile_open_question_reproduces_the_full_artifact():
    at = app().run()
    button_labeled(at, "Open question").click().run()
    question = at.session_state["ask_q"]
    expected = pipeline.answer(question)
    assert at.session_state["nav"] == "Home"
    assert expected.result_hash in rendered_text(at)
    assert at.session_state["history"][-1]["hash"] == expected.result_hash


def test_home_scope_control_recomputes_unpinned_tiles():
    at = app().run()
    baseline = next(metric.value for metric in at.metric if metric.label == "TRx")
    at.selectbox(key="home_region").set_value("East").run()
    scoped = next(metric.value for metric in at.metric if metric.label == "TRx · East")
    assert scoped != baseline


def test_tile_breakdown_keeps_the_selected_controls():
    at = app().run()
    at.radio(key="home_window").set_value("R3M").run()
    at.radio(key="home_basis").set_value("YoY").run()
    at.selectbox(key="home_region").set_value("East").run()
    button_labeled(at, "Break this down").click().run()
    question = at.session_state["ask_q"]
    assert "last 3 months" in question
    assert "same month last year" in question
    assert "East" in question
    assert "Largest single contributor" in rendered_text(at)


def test_home_controls_survive_navigation_round_trip():
    at = app().run()
    at.radio(key="home_window").set_value("R6M").run()
    at.radio(key="home_basis").set_value("YoY").run()
    at.selectbox(key="home_region").set_value("North").run()
    at.radio(key="nav").set_value("Monitoring").run()
    at.radio(key="nav").set_value("Home").run()
    assert at.radio(key="home_window").value == "R6M"
    assert at.radio(key="home_basis").value == "YoY"
    assert at.selectbox(key="home_region").value == "North"


def test_no_key_makes_llm_requirement_visible():
    at = app().run()
    text = rendered_text(at)
    assert any("Language-model translation is off" in w.value for w in at.warning)
    assert "No API credential is bundled" in text
    assert any("required for LLM translation" in t.label for t in at.text_input)


def test_session_key_updates_llm_status_without_exposing_key():
    at = app().run()
    key_input = next(t for t in at.text_input if "required for LLM translation" in t.label)
    key_input.set_value("sk-ant-test-placeholder").run()
    text = rendered_text(at)
    assert "Language-model translation is enabled" in text
    assert "credential entered for this app session" in text
    assert "sk-ant-test-placeholder" not in text


def test_deployment_key_reports_credential_source(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")
    at = app().run()
    text = rendered_text(at)
    assert "Language-model translation is enabled" in text
    assert "configured by the deployment owner" in text
    assert "sk-ant-test-placeholder" not in text


@pytest.mark.parametrize("q,expect", [
    ("What is revenue in the West region?", "Net revenue"),
    ("Trend calls by month in the North region", "Sales calls"),
    ("Which segments account for the revenue change?", "Largest single contributor"),
    ("List whitespace accounts with no activity", "whitespace accounts"),
    ("What was the impact of the partner enablement program in the East?",
     "difference-in-differences"),
    ("Why did revenue drop in the West?", "difference-in-differences"),
])
def test_question_classes_round_trip(q, expect):
    at = ask(app().run(), q)
    assert not at.exception
    assert expect in rendered_text(at), f"expected {expect!r} in answer for {q!r}"


@pytest.mark.parametrize("q", [
    "Forecast revenue for next quarter",
    "Why is morale down this quarter?",
    "What is our customer happiness index?",
])
def test_refusals_render_as_scoped_refusals(q):
    at = ask(app().run(), q)
    assert not at.exception
    assert "Scoped refusal" in rendered_text(at)


def test_refusal_reframe_chip_reasks():
    at = ask(app().run(), "Forecast revenue for next quarter")
    chip = next(b for b in at.button if (b.label or "").startswith("Trend "))
    chip.click().run()
    assert not at.exception
    assert at.session_state["ask_q"] == chip.label
    assert "Sales calls" in rendered_text(at) or "Revenue" in rendered_text(at)
    assert len(at.subheader) > 0  # a real answer rendered, not a refusal


# --------------------------------------------------------------------------- #
# Artifact actions
# --------------------------------------------------------------------------- #
def test_downloads_exist_on_answers():
    at = ask(app().run(), "What is revenue in the West region?")
    labels = [d.label for d in at.get("download_button")]
    assert any("JSON" in l for l in labels), labels

    at = ask(app().run(), "List whitespace accounts with no activity")
    labels = [d.label for d in at.get("download_button")]
    assert any("JSON" in l for l in labels), labels
    assert any("CSV" in l for l in labels), labels

    # refusals are answers too: the artifact stays exportable
    at = ask(app().run(), "Forecast revenue for next quarter")
    labels = [d.label for d in at.get("download_button")]
    assert any("JSON" in l for l in labels), labels


def test_waterfall_renders_on_decomposition():
    at = ask(app().run(), "Which segments account for the revenue change?")
    # Streamlit 1.59 renamed the AppTest element from the Arrow-prefixed
    # identifier; accept both so the declared >=1.40 range remains testable.
    charts = at.get("vega_lite_chart") or at.get("arrow_vega_lite_chart")
    assert len(charts) >= 1


# --------------------------------------------------------------------------- #
# The analyst loop: drill-through navigation
# --------------------------------------------------------------------------- #
def test_monitoring_breaks_down_to_ask():
    at = app().run()
    at.radio(key="nav").set_value("Monitoring").run()
    button_labeled(at, "Break this down").click().run()
    assert at.session_state["nav"] == "Home"
    assert "account for" in at.session_state["ask_q"]
    assert "Largest single contributor" in rendered_text(at)


def test_decomposition_drills_into_causal_studio():
    at = ask(app().run(), "Which segments account for the revenue change?")
    button_labeled(at, "Test attribution in Causal Studio").click().run()
    assert at.session_state["nav"] == "Causal Studio"
    text = rendered_text(at)
    assert "difference-in-differences" in text
    assert "Assumption checks" in text


def test_analyst_signoff_stamps_and_records(tmp_path):
    at = app().run()
    at.radio(key="nav").set_value("Causal Studio").run()
    button_labeled(at, "Propose a design").click().run()
    button_labeled(at, "Mark as analyst-reviewed").click().run()
    assert not at.exception
    assert "Analyst-reviewed" in rendered_text(at)
    hist = services.feedback_history()
    assert (hist["verdict"] == "analyst_reviewed").any()


# --------------------------------------------------------------------------- #
# Context survives navigation
# --------------------------------------------------------------------------- #
def test_home_question_context_survives_navigation_round_trip():
    at = ask(app().run(), "What is revenue in the West region?")
    at.radio(key="nav").set_value("Semantic Layer").run()
    at.radio(key="nav").set_value("Home").run()
    assert not at.exception
    box = next(t for t in at.text_input if t.label == "Question")
    assert box.value == "What is revenue in the West region?"
    assert any("Net revenue" in s.value for s in at.subheader)


def test_studio_context_survives_navigation_round_trip():
    at = app().run()
    at.radio(key="nav").set_value("Causal Studio").run()
    at.selectbox(key="studio_event").set_value("west_shock").run()
    button_labeled(at, "Propose a design").click().run()
    assert "Competitor entry" in rendered_text(at)
    at.radio(key="nav").set_value("Monitoring").run()
    at.radio(key="nav").set_value("Causal Studio").run()
    assert not at.exception
    assert "Competitor entry" in rendered_text(at)  # selection and brief kept


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def test_history_records_and_replays():
    at = app().run()
    at = ask(at, "What is revenue in the West region?")
    at = ask(at, "Trend calls by month in the North region")
    history = at.session_state["history"]
    assert [e["question"] for e in history] == [
        "What is revenue in the West region?",
        "Trend calls by month in the North region"]
    button_labeled(at, "What is revenue in the West region?").click().run()
    assert not at.exception
    assert any("Net revenue" in s.value for s in at.subheader)


# --------------------------------------------------------------------------- #
# Accuracy record through the UI
# --------------------------------------------------------------------------- #
def test_accuracy_check_via_ui():
    at = app(timeout=180).run()
    at.radio(key="nav").set_value("Reliability").run()
    button_labeled(at, "Run accuracy check").click()
    at.run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values.get("Pass rate") == "100%"
    assert values.get("Reproducible") == "100%"
    assert values.get("Correct refusals") == "100%"


# --------------------------------------------------------------------------- #
# Watchlists
# --------------------------------------------------------------------------- #
def test_watch_pin_and_remove_flow():
    at = ask(app().run(), "What is revenue in the West region?")
    button_labeled(at, "Watch").click().run()
    assert services.load_watchlist(), "watch was not persisted"

    at.radio(key="nav").set_value("Monitoring").run()
    assert "Revenue · region=West" in rendered_text(at)
    button_labeled(at, "Remove").click().run()
    assert services.load_watchlist() == []
    assert "Nothing watched yet" in rendered_text(at)


# --------------------------------------------------------------------------- #
# Comparison basis
# --------------------------------------------------------------------------- #
def test_basis_selector_recomputes_decomposition():
    at = ask(app().run(), "Which segments account for the revenue change?")
    assert "vs prior quarter" in rendered_text(at)
    at.selectbox(key="ask_basis").set_value("same month last year").run()
    assert not at.exception
    assert any("same month last year" in s.value for s in at.subheader)


# --------------------------------------------------------------------------- #
# Governance administration
# --------------------------------------------------------------------------- #
def test_admin_changes_materiality_and_logs_it():
    from harness import semantic_layer as sl
    at = app().run()
    at.radio(key="nav").set_value("Semantic Layer").run()
    at.number_input[0].set_value(10.0).run()
    button_labeled(at, "Apply governance changes").click().run()
    assert not at.exception
    assert abs(sl.materiality() - 0.10) < 1e-9
    assert sl.governance_log(), "governance change was not logged"
    assert "10.0%" in rendered_text(at)  # registry caption reflects the new threshold
