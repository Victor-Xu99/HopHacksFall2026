"""Adapter from the SafetyHops SQL database (Synthea-style synthetic encounters).

The database on this machine is named `SafetyHops`, not `SafetyHop`. Tables live
in `dbo` (`patients`, `encounters`, `observations`, `medications`, `procedures`,
`conditions`) and are not the canonical `core` layer.

One encounter is one case. Labels come from harm-coded conditions on that
encounter, never from labs or meds, so the watcher is not scoring the label it
was given.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine

from src.data.sql_store import DEFAULT_SERVER, connection_url
from src.domain.models import PatientCase, PatientEvent

DEFAULT_DATABASE = "SafetyHops"

# Chronic / routine conditions stay unlabeled. These ten are the harm archetypes
# planted in this synthetic warehouse.
HARM_CONDITIONS = frozenset(
    {
        "Postoperative hemorrhage",
        "Hospital-acquired surgical site infection",
        "Severe iatrogenic hypoglycemia",
        "Acute kidney injury due to contrast and medication",
        "Warfarin coagulopathy / adverse bleeding",
        "Benzodiazepine oversedation / toxic encephalopathy",
        "Adverse reaction to anticoagulant therapy",
        "Accidental puncture or laceration during a procedure",
        "Opioid oversedation and respiratory depression",
        "Severe hyperkalemia with cardiac toxicity",
    }
)

READMISSION_WINDOW_DAYS = 30


def hops_available(server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE) -> bool:
    try:
        engine = create_engine(connection_url(server, database))
        with engine.connect() as connection:
            return connection.exec_driver_sql("SELECT OBJECT_ID('dbo.encounters', 'U')").scalar() is not None
    except Exception:
        return False


def _read(engine, table: str, columns: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT {columns} FROM dbo.{table}", engine)


def _when(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_safetyhops_cases(
    server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE
) -> List[PatientCase]:
    if not hops_available(server, database):
        raise FileNotFoundError(
            f"Cannot read dbo.encounters on {server}/{database}. "
            "The warehouse on this machine is `SafetyHops` (with an s)."
        )

    engine = create_engine(connection_url(server, database))
    try:
        patients = _read(engine, "patients", "Id, BIRTHDATE, GENDER")
        encounters = _read(engine, "encounters", "Id, START, STOP, PATIENT, ENCOUNTERCLASS, DESCRIPTION")
        observations = _read(engine, "observations", "DATE, ENCOUNTER, DESCRIPTION, VALUE, UNITS")
        medications = _read(engine, "medications", "START, ENCOUNTER, DESCRIPTION, REASONDESCRIPTION")
        procedures = _read(engine, "procedures", "START, ENCOUNTER, DESCRIPTION, REASONDESCRIPTION")
        conditions = _read(engine, "conditions", "START, ENCOUNTER, DESCRIPTION")
    finally:
        engine.dispose()

    patients = patients.rename(columns={"Id": "patient_id"})
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")

    encounters = encounters.rename(columns={"Id": "encounter_id", "PATIENT": "patient_id"})
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
    encounters["STOP"] = pd.to_datetime(encounters["STOP"], errors="coerce")
    encounters = encounters.sort_values(["patient_id", "START"])
    encounters["prev_stop"] = encounters.groupby("patient_id")["STOP"].shift()
    gap = (encounters["START"] - encounters["prev_stop"]).dt.total_seconds() / 86400
    encounters["readmission_gap_days"] = gap.where(gap.between(0, READMISSION_WINDOW_DAYS))

    demographics = patients.set_index("patient_id")
    harm_by_encounter = (
        conditions[conditions["DESCRIPTION"].isin(HARM_CONDITIONS)]
        .groupby("ENCOUNTER")["DESCRIPTION"]
        .apply(lambda s: sorted(set(s)))
    )

    events_by_encounter: Dict[str, List[PatientEvent]] = {eid: [] for eid in encounters["encounter_id"]}

    def add(encounter_id: str, event: Optional[PatientEvent]) -> None:
        if event is not None and encounter_id in events_by_encounter:
            events_by_encounter[encounter_id].append(event)

    seq = 0
    for row in observations.itertuples(index=False):
        timestamp = _when(row.DATE)
        if timestamp is None or not row.ENCOUNTER:
            continue
        seq += 1
        units = _clean(row.UNITS)
        details = f"{_clean(row.VALUE)} {units}".strip()
        add(
            row.ENCOUNTER,
            PatientEvent(
                event_id=f"{row.ENCOUNTER}-lab-{seq}",
                event_type="lab",
                timestamp=timestamp,
                value=_clean(row.DESCRIPTION),
                details=details,
            ),
        )

    seq = 0
    for row in medications.itertuples(index=False):
        timestamp = _when(row.START)
        if timestamp is None or not row.ENCOUNTER:
            continue
        seq += 1
        reason = _clean(row.REASONDESCRIPTION)
        add(
            row.ENCOUNTER,
            PatientEvent(
                event_id=f"{row.ENCOUNTER}-med-{seq}",
                event_type="medication",
                timestamp=timestamp,
                value=_clean(row.DESCRIPTION),
                details=f"Administered. {reason}".strip(),
            ),
        )

    seq = 0
    for row in procedures.itertuples(index=False):
        timestamp = _when(row.START)
        if timestamp is None or not row.ENCOUNTER:
            continue
        seq += 1
        reason = _clean(row.REASONDESCRIPTION)
        add(
            row.ENCOUNTER,
            PatientEvent(
                event_id=f"{row.ENCOUNTER}-proc-{seq}",
                event_type="procedure",
                timestamp=timestamp,
                value=_clean(row.DESCRIPTION),
                details=reason or "Coded procedure.",
            ),
        )

    cases: List[PatientCase] = []
    for row in encounters.itertuples(index=False):
        if pd.isna(row.START):
            continue
        person = demographics.loc[row.patient_id] if row.patient_id in demographics.index else None
        age = 0
        gender = "U"
        if person is not None:
            gender = _clean(person["GENDER"]) or "U"
            if pd.notna(person["BIRTHDATE"]):
                age = max(0, int((row.START - person["BIRTHDATE"]).days / 365.25))

        encounter_class = _clean(row.ENCOUNTERCLASS) or "inpatient"
        description = _clean(row.DESCRIPTION) or "Encounter"
        details = f"{encounter_class} admission. {description}."
        if pd.notna(row.readmission_gap_days):
            details = (
                f"Readmission within {READMISSION_WINDOW_DAYS} days: "
                f"{row.readmission_gap_days:.1f} days after the previous discharge. {details}"
            )

        admission = PatientEvent(
            event_id=f"{row.encounter_id}-admission",
            event_type="admission",
            timestamp=row.START.isoformat(),
            value="Inpatient admission" if encounter_class == "inpatient" else description,
            details=details,
        )
        events = [admission] + events_by_encounter.get(row.encounter_id, [])
        events.sort(key=lambda e: e.timestamp)

        harm_names = harm_by_encounter.get(row.encounter_id, [])
        scenario = harm_names[0] if harm_names else encounter_class

        cases.append(
            PatientCase(
                patient_id=f"{row.patient_id}-{row.encounter_id}",
                age=age,
                gender=gender,
                events=events,
                is_harm_event=bool(harm_names),
                scenario=scenario,
                label_source="safetyhops_condition_proxy",
            )
        )
    return cases
