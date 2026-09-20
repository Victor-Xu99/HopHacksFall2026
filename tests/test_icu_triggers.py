"""ICU transfer triggers against MIMIC-shaped unit names.

These names used to miss the watcher because it looked only for the token
'icu'. MIMIC writes 'Coronary Care Unit (CCU)' and 'Neuro Stepdown'.
"""

from __future__ import annotations

import pytest

from src.data.mimic import load_mimic_cases, mimic_available
from src.domain.models import PatientCase, PatientEvent
from src.engine.watcher import StructuredDataWatcher

WATCHER = StructuredDataWatcher(anchor="admission")


def _stay(*events: PatientEvent) -> PatientCase:
    return PatientCase(
        patient_id="icu-smoke",
        age=70,
        gender="F",
        events=[
            PatientEvent(
                event_id="adm",
                event_type="admission",
                timestamp="2026-09-01T08:00:00",
                value="Inpatient admission",
                details="",
            ),
            *events,
        ],
        is_harm_event=False,
        scenario="hospital",
        label_source="fixture",
    )


def _transfer(event_id: str, when: str, unit: str, details: str) -> PatientEvent:
    return PatientEvent(
        event_id=event_id,
        event_type="transfer",
        timestamp=when,
        value=unit,
        details=details,
    )


@pytest.mark.parametrize(
    "unit",
    [
        "Medical Intensive Care Unit (MICU)",
        "Coronary Care Unit (CCU)",
        "Neuro Stepdown",
    ],
)
def test_mimic_icu_unit_names_fire_unplanned_escalation(unit: str) -> None:
    case = _stay(
        _transfer(
            "t1",
            "2026-09-01T20:00:00",
            unit,
            "Unplanned escalation to intensive care from Med/Surg.",
        )
    )
    features = WATCHER.extract_features(case)
    assert features["unplanned_icu_transfer"] == 1.0
    assert features["repeat_icu_escalation"] == 0.0


def test_planned_ed_to_icu_is_not_an_escalation() -> None:
    case = _stay(
        _transfer(
            "t1",
            "2026-09-01T08:10:00",
            "Medical Intensive Care Unit (MICU)",
            "Planned or direct admission to intensive care from Emergency Department.",
        )
    )
    assert WATCHER.extract_features(case)["unplanned_icu_transfer"] == 0.0


def test_two_floor_to_icu_moves_set_repeat_escalation() -> None:
    case = _stay(
        _transfer(
            "t1",
            "2026-09-01T18:00:00",
            "Medical Intensive Care Unit (MICU)",
            "Unplanned escalation to intensive care from Med/Surg.",
        ),
        _transfer(
            "t2",
            "2026-09-03T04:00:00",
            "Coronary Care Unit (CCU)",
            "Unplanned escalation to intensive care from Med/Surg.",
        ),
    )
    features = WATCHER.extract_features(case)
    assert features["unplanned_icu_transfer"] == 1.0
    assert features["repeat_icu_escalation"] == 1.0


@pytest.mark.skipif(not mimic_available(), reason="MIMIC-IV demo not extracted")
def test_mimic_demo_fires_unplanned_icu_transfer() -> None:
    cases = load_mimic_cases()
    fired = sum(
        1 for case in cases if WATCHER.extract_features(case)["unplanned_icu_transfer"] > 0
    )
    assert fired > 0, "unplanned_icu_transfer still never fires on the MIMIC demo"
