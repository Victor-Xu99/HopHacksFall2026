"""Runs the Streamlit script end to end without a browser."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data.mimic import mimic_available

APP = str(Path(__file__).resolve().parent.parent / "app.py")
NEEDS_MIMIC = pytest.mark.skipif(not mimic_available(), reason="MIMIC-IV demo not extracted")


def launch(timeout: int = 300) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def select_mimic(app: AppTest) -> AppTest:
    cohort = app.radio("source_radio")
    cohort.set_value(next(o for o in cohort.options if "MIMIC" in o)).run()
    return app


def test_app_renders_without_exceptions():
    app = launch()
    assert not app.exception
    assert any("SafetyNet" in title.value for title in app.title)


def test_switching_to_the_decision_tree_still_renders():
    app = launch()
    app.radio("model_family").set_value("tree").run()
    assert not app.exception


def test_threshold_queue_mode_renders():
    app = launch()
    app.radio("queue_mode").set_value("Score threshold").run()
    assert not app.exception


@NEEDS_MIMIC
def test_real_mimic_source_renders():
    app = select_mimic(launch(timeout=600))
    assert not app.exception
    # The note reader must be excluded rather than fed empty features.
    assert any("no clinical notes" in info.value for info in app.info)


@NEEDS_MIMIC
def test_real_mimic_source_with_tree_renders():
    app = select_mimic(launch(timeout=600))
    app.radio("model_family").set_value("tree").run()
    assert not app.exception


@NEEDS_MIMIC
def test_real_mimic_reports_dark_triggers():
    """Coverage must call out triggers this dataset cannot support."""
    app = select_mimic(launch(timeout=600))
    assert any("Not derivable" in error.value for error in app.error)
