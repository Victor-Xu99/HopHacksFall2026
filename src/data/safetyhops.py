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

import logging
import os

from src.domain.models import PatientCase, PatientEvent

logger = logging.getLogger(__name__)

# Override with SAFETYNET_SQL_SERVER in your shell / .env to match your local
# SQL Server instance name.  The teammate default is .\SQLEXPRESS; locally you
# may need .\XUSHOE or just the bare instance name.
DEFAULT_SERVER = os.environ.get("SAFETYNET_SQL_SERVER", r".\SQLEXPRESS")
DEFAULT_DATABASE = "SafetyHops"
DRIVER = "ODBC Driver 17 for SQL Server"


def _connection_url(server: str, database: str) -> str:
    return (
        f"mssql+pyodbc://@{server}/{database}"
        f"?driver={DRIVER.replace(' ', '+')}&trusted_connection=yes"
    )

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


def _driver_message(exc: BaseException) -> str:
    """The ODBC driver's complaint, without SQLAlchemy's wrapper and repeated clauses."""
    text = str(getattr(exc, "orig", exc))
    for marker in ("[SQL Server]", "[ODBC Driver 17 for SQL Server]"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    return text.split(";")[0].strip(" '\")") or exc.__class__.__name__


def hops_available(server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE) -> bool:
    try:
        from sqlalchemy import create_engine  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        # A missing package and an unreachable server both end in "not available",
        # but only one of them is fixed with pip.
        logger.warning(
            "SafetyHops unavailable: %s is not installed. Run `pip install sqlalchemy pyodbc`.",
            exc.name,
        )
        return False

    try:
        engine = create_engine(_connection_url(server, database))
        with engine.connect() as connection:
            found = connection.exec_driver_sql("SELECT OBJECT_ID('dbo.encounters', 'U')").scalar()
        if found is None:
            logger.warning(
                "SafetyHops unavailable: connected to %s/%s but dbo.encounters does not exist.",
                server,
                database,
            )
            return False
        return True
    except Exception as exc:
        # The driver's own message is the diagnostic; the traceback through
        # SQLAlchemy's connect stack is noise unless you are debugging this file.
        logger.warning(
            "SafetyHops unavailable: cannot reach %s/%s (%s). Set SAFETYNET_SQL_SERVER to "
            "your instance name: default instances take the bare machine name, named "
            "instances take .\\NAME.",
            server,
            database,
            _driver_message(exc),
        )
        logger.debug("SafetyHops connection traceback", exc_info=True)
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

    from sqlalchemy import create_engine  # noqa: PLC0415
    engine = create_engine(_connection_url(server, database))
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
