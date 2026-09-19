"""Runs the Streamlit script end to end without a browser."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data.mimic import mimic_available
from src.data.safetyhops import hops_available
from src.data.sql_store import available as sql_available

APP = str(Path(__file__).resolve().parent.parent / "app.py")
NEEDS_MIMIC = pytest.mark.skipif(not mimic_available(), reason="MIMIC-IV demo not extracted")
NEEDS_SQL = pytest.mark.skipif(
    not sql_available(), reason="SafetyNet canonical layer not reachable"
)
NEEDS_HOPS = pytest.mark.skipif(not hops_available(), reason="SafetyHops database not reachable")


def launch(timeout: int = 300) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def select_cohort(app: AppTest, *tokens: str) -> AppTest:
    cohort = app.radio("source_radio")
    matching = next(o for o in cohort.options if all(t in o for t in tokens))
    cohort.set_value(matching).run()
    return app


def select_mimic(app: AppTest) -> AppTest:
    return select_cohort(app, "MIMIC")


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
    app.radio("queue_mode").set_value("A score cutoff").run()
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


@NEEDS_SQL
def test_sql_mimic_cohort_renders_and_still_drops_the_note_reader():
    app = select_cohort(launch(timeout=600), "MIMIC", "SQL Server")
    assert not app.exception
    assert any("no clinical notes" in info.value for info in app.info)


@NEEDS_SQL
def test_sql_synthetic_cohort_keeps_the_note_reader():
    app = select_cohort(launch(timeout=600), "Synthetic", "SQL Server")
    assert not app.exception
    assert not any("no clinical notes" in info.value for info in app.info)


@NEEDS_HOPS
def test_safetyhops_sql_source_renders():
    app = select_cohort(launch(timeout=600), "SafetyHops")
    assert not app.exception
    assert any("synthetic encounters" in success.value.lower() for success in app.success)
