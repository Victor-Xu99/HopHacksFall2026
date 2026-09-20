"""Print ICU trigger coverage. Uses MIMIC if present, otherwise fixtures."""

from __future__ import annotations

from collections import Counter

from src.data.mimic import load_mimic_cases, mimic_available
from src.engine.watcher import StructuredDataWatcher
from src.domain.models import PatientCase, PatientEvent

ICU_FEATURES = ("unplanned_icu_transfer", "repeat_icu_escalation")


def _fixture_cases() -> list[PatientCase]:
    def stay(patient_id: str, unit: str, details: str) -> PatientCase:
        return PatientCase(
            patient_id=patient_id,
            age=70,
            gender="F",
            events=[
                PatientEvent(
                    event_id="a",
                    event_type="admission",
                    timestamp="2026-09-01T08:00:00",
                    value="Inpatient admission",
                    details="",
                ),
                PatientEvent(
                    event_id="t",
                    event_type="transfer",
                    timestamp="2026-09-01T20:00:00",
                    value=unit,
                    details=details,
                ),
            ],
        )

    singles = [
        stay(
            "micu",
            "Medical Intensive Care Unit (MICU)",
            "Unplanned escalation to intensive care from Med/Surg.",
        ),
        stay(
            "ccu",
            "Coronary Care Unit (CCU)",
            "Unplanned escalation to intensive care from Med/Surg.",
        ),
        stay(
            "stepdown",
            "Neuro Stepdown",
            "Unplanned escalation to intensive care from Med/Surg.",
        ),
    ]
    repeat = stay(
        "repeat",
        "Medical Intensive Care Unit (MICU)",
        "Unplanned escalation to intensive care from Med/Surg.",
    )
    repeat.events.append(
        PatientEvent(
            event_id="t2",
            event_type="transfer",
            timestamp="2026-09-03T04:00:00",
            value="Coronary Care Unit (CCU)",
            details="Unplanned escalation to intensive care from Med/Surg.",
        )
    )
    return singles + [repeat]


def main() -> None:
    watcher = StructuredDataWatcher(anchor="admission")
    if mimic_available():
        cases = load_mimic_cases()
        origin = "MIMIC-IV demo"
    else:
        cases = _fixture_cases()
        origin = "MIMIC-shaped fixtures (demo CSVs not on disk)"

    fired: Counter[str] = Counter()
    for case in cases:
        features = watcher.extract_features(case)
        for name in ICU_FEATURES:
            if features[name] > 0:
                fired[name] += 1

    print(f"{origin}: {len(cases)} stays")
    for name in ICU_FEATURES:
        count = fired[name]
        status = "ALIVE" if count else "DEAD"
        print(f"  {name:28s} {count:4d}  {status}")
    if fired["unplanned_icu_transfer"] == 0 or fired["repeat_icu_escalation"] == 0:
        raise SystemExit("an ICU trigger never fired")


if __name__ == "__main__":
    main()
