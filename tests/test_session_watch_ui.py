"""Watch UI uses session-owned saved insights and never writes shared state."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from harness import saved_insights, services


APP = str(Path(__file__).parent.parent / "app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=60)


def _ask(at, question):
    next(item for item in at.text_input if item.label == "Question").set_value(question)
    return at.run()


def _button(at, label):
    return next(button for button in at.button if label in (button.label or ""))


def _text(at):
    values = [item.value for item in at.markdown]
    values += [item.value for item in at.caption]
    return " ".join(values)


def test_watch_is_session_local_evaluated_and_removable():
    first = _ask(_app().run(), "What is TRx in the West region?")
    _button(first, "Watch").click().run()
    store = first.session_state[saved_insights.SESSION_STORE_KEY]
    assert len(store) == 1
    saved = store.all()[0]
    assert saved.metric == "trx"
    assert saved.filters == {"region": "West"}
    assert saved.source == "source_a"
    assert saved.variant == "units"
    assert services.load_watchlist() == []

    second = _app().run()
    second.radio(key="nav").set_value("Monitoring").run()
    assert saved.label not in _text(second)
    assert "Nothing watched yet" in _text(second)

    first.radio(key="nav").set_value("Monitoring").run()
    assert not first.exception
    assert saved.label in _text(first)
    _button(first, "Remove").click().run()
    assert len(first.session_state[saved_insights.SESSION_STORE_KEY]) == 0
    assert "Nothing watched yet" in _text(first)
