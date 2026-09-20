"""Hospital CSV ingest: stays, labs, meds, transfers.

Rows are appended to SafetyNet SQL. Ranking is not done here — that waits until
a stay's discharged column is filled in.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.hospital_adapt import load_mapped_tables, normalize_name
from src.domain.models import PatientCase, PatientEvent

HOSPITAL_SOURCE = "hospital"
EXTRACT_FILES = ("stays.csv", "labs.csv", "meds.csv", "transfers.csv")

STAY_COLUMNS = ("encounter_id", "age", "sex", "admitted", "discharged")
LAB_COLUMNS = ("encounter_id", "time", "name", "value", "units")
MED_COLUMNS = ("encounter_id", "time", "name")
TRANSFER_COLUMNS = ("encounter_id", "time", "to_unit")


def _parse_time(value: str) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "")).isoformat()


def _event_id(event_type: str, timestamp: str, value: str, details: str) -> str:
    raw = f"{event_type}|{timestamp}|{value}|{details}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _event(event_type: str, timestamp: str, value: str, details: str = "") -> PatientEvent:
    return PatientEvent(
        event_id=_event_id(event_type, timestamp, value, details),
        event_type=event_type,
        timestamp=timestamp,
        value=value,
        details=details,
    )


def _age(value: str) -> int:
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def load_extract_dir(directory: Path) -> Tuple[List[PatientCase], Dict[str, object]]:
    """Build cases from a folder of tables. Filenames and headers may vary."""
    tables, report = load_mapped_tables(Path(directory))
    stays, labs, meds, transfers = tables["stays"], tables["labs"], tables["meds"], tables["transfers"]

    cases: Dict[str, PatientCase] = {}
    for row in stays.itertuples(index=False):
        admitted = _parse_time(row.admitted)
        if admitted is None:
            raise ValueError(f"Stay {row.encounter_id} is missing admitted time.")
        discharged = _parse_time(row.discharged)
        case = PatientCase(
            patient_id=str(row.encounter_id).strip(),
            age=_age(row.age),
            gender=str(row.sex).strip(),
            events=[_event("admission", admitted, "Inpatient admission", "")],
            is_harm_event=False,
            scenario="hospital",
            label_source="hospital",
            discharged_at=discharged,
        )
        cases[case.patient_id] = case

    for row in labs.itertuples(index=False):
        encounter_id = str(row.encounter_id).strip()
        if encounter_id not in cases:
            continue
        when = _parse_time(row.time)
        if when is None:
            continue
        units = str(row.units).strip()
        value = str(row.value).strip()
        details = f"{value} {units}".strip()
        cases[encounter_id].events.append(
            _event("lab", when, normalize_name(row.name), details)
        )

    for row in meds.itertuples(index=False):
        encounter_id = str(row.encounter_id).strip()
        if encounter_id not in cases:
            continue
        when = _parse_time(row.time)
        if when is None:
            continue
        cases[encounter_id].events.append(
            _event("medication", when, normalize_name(row.name), "")
        )

    for row in transfers.itertuples(index=False):
        encounter_id = str(row.encounter_id).strip()
        if encounter_id not in cases:
            continue
        when = _parse_time(row.time)
        if when is None:
            continue
        cases[encounter_id].events.append(_event("transfer", when, str(row.to_unit).strip(), ""))

    return list(cases.values()), report


def ingest_extract(
    directory: Path,
    engine=None,
) -> Dict[str, object]:
    """Upsert stays and append new events. Returns census plus newly discharged ids."""
    from src.data.sql_store import append_events, discharged_unscored_ids, stay_census, upsert_stays

    source = HOSPITAL_SOURCE
    cases, mapping = load_extract_dir(directory)
    upsert_stays(cases, source, engine=engine)
    appended = append_events(cases, source, engine=engine)
    census = stay_census(source, engine=engine)
    pending = discharged_unscored_ids(source, engine=engine)
    return {
        "source": source,
        "stays_upserted": len(cases),
        "events_appended": appended,
        **census,
        "ready_to_rank": pending,
        "mapping": mapping,
    }


def rank_discharged(engine=None, model_type: str = "logistic") -> Dict[str, object]:
    """Score hospital stays that have a discharge time and no score yet."""
    from src.data.sql_store import (
        CaseScore,
        FeatureContribution,
        discharged_unscored_ids,
        read_cases,
        write_scores,
    )
    from src.service import trained_model

    source = "hospital"
    pending_ids = set(discharged_unscored_ids(source, engine=engine))
    if not pending_ids:
        return {"scored": 0, "case_ids": []}

    hospital_cases = [case for case in read_cases(source, engine=engine) if case.patient_id in pending_ids]
    model = trained_model("hospital", model_type)

    scores = []
    for case in hospital_cases:
        evidence = model.get_evidence(case)
        scores.append(
            CaseScore(
                source_case_id=case.patient_id,
                score=float(model.predict_score(case)),
                contributions=[
                    FeatureContribution(
                        feature=item.feature,
                        value=item.value,
                        weight=item.weight,
                        contribution=item.contribution,
                    )
                    for item in evidence
                ],
            )
        )
    write_scores(scores, source, model_type, engine=engine)
    return {
        "scored": len(scores),
        "case_ids": [item.source_case_id for item in scores],
        "scores": {item.source_case_id: item.score for item in scores},
    }


def write_extract_dir(folder: Path, files: Dict[str, bytes]) -> Path:
    """Write the four named CSVs into folder. Keys must be stays.csv, labs.csv, ..."""
    missing = [name for name in EXTRACT_FILES if name not in files or not files[name]]
    if missing:
        raise ValueError(f"Upload is missing: {', '.join(missing)}")
    folder.mkdir(parents=True, exist_ok=True)
    for name in EXTRACT_FILES:
        (folder / name).write_bytes(files[name])
    return folder


def ingest_and_rank(directory: Path, engine=None, model_type: str = "logistic") -> Dict[str, object]:
    ingest = ingest_extract(directory, engine=engine)
    ranked = rank_discharged(engine=engine, model_type=model_type)
    return {**ingest, "ranked": ranked}
