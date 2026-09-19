"""Runs the Streamlit script end to end without a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_without_exceptions():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    assert not app.exception
    assert any("SafetyNet" in title.value for title in app.title)


def test_switching_to_the_decision_tree_still_renders():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    app.sidebar.radio[0].set_value("tree").run()
    assert not app.exception
