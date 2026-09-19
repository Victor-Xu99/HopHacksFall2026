"""Adapter from the MIMIC-IV clinical database demo to our PatientCase model.

Real de-identified records from Beth Israel Deaconess: 100 patients, 275
admissions, open access under ODbL 1.0. What this release can and cannot support:

  labs, meds, transfers, ICU stays, procedures, readmissions   real, available
  free-text clinical notes                                     excluded from the demo
  adjudicated harm labels                                      absent from MIMIC entirely

So the structured watcher runs for real here, the note reader has nothing to read,
and the label has to be a proxy. Nothing in this module invents any of the three.

One admission is one case. Harm is a property of an episode of care rather than of
a person, and readmission only means anything across admissions.

THE LABEL IS A PROXY. `is_harm_event` comes from ICD complication-of-care codes
(ICD-10 T80-T88 and Y62-Y84, ICD-9 996-999, E870-E879, E930-E949), the same family
of codes behind the AHRQ Patient Safety Indicators. It is not adjudication:

  - Administrative coding is known to under-capture harm badly. Trigger-tool
    studies find several times more adverse events than coding or voluntary
    reports do, so absence of a code is weak evidence of absence of harm.
  - "Adverse effects in therapeutic use" codes cover expected drug side effects
    as well as genuine harm. Coding does not resolve the expected-versus-
    unanticipated distinction that this project is built around.
  - Coders read the notes. The label therefore derives from the free text, while
    the features derive from orders, labs, and transfers -- different sources,
    which is what makes this a real prediction task rather than a lookup. But
    procedure codes are assigned by those same coders, so that one feature family
    shares a source with the label. Treat it as the weakest link here.

Diagnosis codes are deliberately never emitted as events. They are the label.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.domain.models import PatientCase, PatientEvent

MIMIC_ROOT_DEFAULT = "data/mimic-iv-clinical-database-demo-2.2"

# Care units whose names mark a higher level of care.
ICU_UNIT_TOKENS = (
    "intensive care",
    "icu",
    "ccu",
    "coronary care",
    "neuro stepdown",
    "neuro intermediate",
)

# Units a patient passes through on a planned route into intensive care. An ICU
# arrival from one of these is an expected step, not a deterioration.
PLANNED_ROUTE_TOKENS = ("emergency department", "pacu", "discharge lounge")

# Labs the watcher has clinical rules for, keyed by the label MIMIC uses, valued
# by the canonical name in watcher.LAB_RULES.
LAB_LABEL_MAP: Dict[str, str] = {
    "Hemoglobin": "Hemoglobin",
    "Creatinine": "Creatinine",
    "Potassium": "Potassium",
    "Lactate": "Lactate",
    "INR(PT)": "INR",
    "Platelet Count": "Platelets",
    "White Blood Cells": "WBC",
    "Glucose": "Glucose",
    "Bilirubin, Total": "Bilirubin",
    "Alanine Aminotransferase (ALT)": "ALT",
    "Troponin T": "Troponin",
}

# Only what was actually put into the patient. emar records plenty of orders that
# were "Not Given", and counting those as administrations would be wrong.
EMAR_ADMINISTERED = ("Administered", "Administered in Other Location", "Started")

# inputevents d_items category for blood products. Matching on the word "blood"
# instead would pull in blood-pressure alarms and "Blood Transfusion Consent".
BLOOD_PRODUCT_CATEGORY = "Blood Products/Colloids"
BLOOD_PRODUCT_EXCLUDE = ("consent", "autologous")

# ICD complication-of-care code families used as the proxy label.
ICD10_COMPLICATION = r"^(T8[0-8]|Y6[2-9]|Y7\d|Y8[0-4])"
ICD9_COMPLICATION = r"^(99[6-9]|E87[0-9]|E9[3-4]\d)"

READMISSION_WINDOW_DAYS = 30


def _antidote_pattern() -> str:
    from src.engine.watcher import ANTIDOTES

    return "|".join(sorted(ANTIDOTES, key=len, reverse=True))


ANTIDOTE_PATTERN = _antidote_pattern()


def read_table(
    root: Path, name: str, usecols: Optional[Sequence[str]] = None, **kwargs
) -> pd.DataFrame:
    """Read a MIMIC table by module-qualified name, e.g. 'hosp/labevents'."""
    path = Path(root) / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Extract the MIMIC-IV demo so that {path.parent} exists."
        )
    return pd.read_csv(path, usecols=usecols, low_memory=False, **kwargs)


def mimic_available(root: str = MIMIC_ROOT_DEFAULT) -> bool:
    return (Path(root) / "hosp" / "admissions.csv.gz").exists()


def _is_icu(unit: object) -> bool:
    lowered = str(unit).lower()
    return any(token in lowered for token in ICU_UNIT_TOKENS)


def _on_planned_route(unit: object) -> bool:
    lowered = str(unit).lower()
    return any(token in lowered for token in PLANNED_ROUTE_TOKENS)


def _iso(value: pd.Timestamp) -> Optional[str]:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _event(hadm_id: int, seq: int, event_type: str, when: object, value: str, details: str):
    timestamp = _iso(when)
    if timestamp is None:
        return None
    return PatientEvent(
        event_id=f"{hadm_id}-{event_type}-{seq}",
        event_type=event_type,
        timestamp=timestamp,
        value=str(value),
        details=str(details),
    )


def harm_labels(root: Path) -> pd.Series:
    """Proxy label per hadm_id from complication-of-care diagnosis codes."""
    dx = read_table(root, "hosp/diagnoses_icd", usecols=["hadm_id", "icd_code", "icd_version"])
    code = dx["icd_code"].astype(str).str.strip().str.upper()
    is_complication = ((dx["icd_version"] == 10) & code.str.match(ICD10_COMPLICATION)) | (
        (dx["icd_version"] == 9) & code.str.match(ICD9_COMPLICATION)
    )
    return dx.assign(flag=is_complication).groupby("hadm_id")["flag"].any()


def load_mimic_cases(
    root: str = MIMIC_ROOT_DEFAULT, limit: Optional[int] = None
) -> List[PatientCase]:
    """Build one PatientCase per hospital admission."""
    root_path = Path(root)

    patients = read_table(root_path, "hosp/patients", usecols=["subject_id", "gender", "anchor_age"])
    admissions = read_table(
        root_path,
        "hosp/admissions",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions = admissions.sort_values(["subject_id", "admittime"])
    admissions["prev_discharge"] = admissions.groupby("subject_id")["dischtime"].shift()
    gap = (admissions["admittime"] - admissions["prev_discharge"]).dt.total_seconds() / 86400
    admissions["readmission_gap_days"] = gap.where(gap.between(0, READMISSION_WINDOW_DAYS))

    if limit is not None:
        admissions = admissions.head(limit)
    keep = set(admissions["hadm_id"])

    labels = harm_labels(root_path)
    events_by_admission = _collect_events(root_path, keep)
    demographics = patients.set_index("subject_id")

    cases: List[PatientCase] = []
    for row in admissions.itertuples():
        events = events_by_admission.get(row.hadm_id, [])

        admission_details = f"{row.admission_type} admission."
        if pd.notna(row.readmission_gap_days):
            admission_details = (
                f"Readmission within {READMISSION_WINDOW_DAYS} days: "
                f"{row.readmission_gap_days:.1f} days after the previous discharge. "
                f"{row.admission_type} admission."
            )
        admission_event = _event(
            row.hadm_id, 0, "admission", row.admittime, "Inpatient admission", admission_details
        )
        if admission_event is not None:
            events = [admission_event] + events

        person = demographics.loc[row.subject_id]
        cases.append(
            PatientCase(
                patient_id=f"{row.subject_id}-{row.hadm_id}",
                age=int(person["anchor_age"]),
                gender=str(person["gender"]),
                events=events,
                is_harm_event=bool(labels.get(row.hadm_id, False)),
                scenario=str(row.admission_type),
                label_source="icd_complication_proxy",
            )
        )
    return cases


def _collect_events(root: Path, keep: set) -> Dict[int, List[PatientEvent]]:
    collected: Dict[int, List[PatientEvent]] = {}

    def add(event: Optional[PatientEvent], hadm_id: int) -> None:
        if event is not None:
            collected.setdefault(hadm_id, []).append(event)

    # --- transfers, with a proxy for whether an escalation was unplanned --------
    transfers = read_table(
        root, "hosp/transfers", usecols=["hadm_id", "eventtype", "careunit", "intime"]
    )
    transfers = transfers[transfers["hadm_id"].isin(keep)].copy()
    transfers["intime"] = pd.to_datetime(transfers["intime"])
    transfers = transfers.sort_values(["hadm_id", "intime"])
    transfers["previous_unit"] = transfers.groupby("hadm_id")["careunit"].shift()

    for seq, row in enumerate(transfers.itertuples()):
        if not _is_icu(row.careunit):
            continue
        first_unit = pd.isna(row.previous_unit)
        # MIMIC does not record whether a transfer was planned. Arriving in
        # intensive care straight from the ED, from recovery, or as the first unit
        # of the stay is the expected route; arriving from a general ward part-way
        # through the stay is the pattern that means deterioration. This is a
        # proxy, and it is the weakest inference in this adapter.
        # An escalation means arriving from a general ward. Coming from the ED, from
        # recovery, or from another intensive care unit is a planned route or a
        # lateral move, not a deterioration.
        planned = (
            first_unit
            or row.eventtype == "admit"
            or _on_planned_route(row.previous_unit)
            or _is_icu(row.previous_unit)
        )
        origin = "direct" if first_unit else str(row.previous_unit)
        details = (
            f"Planned or direct admission to intensive care from {origin}."
            if planned
            else f"Unplanned escalation to intensive care from {origin}."
        )
        add(_event(row.hadm_id, seq, "transfer", row.intime, row.careunit, details), row.hadm_id)

    # --- labs, restricted to those the watcher has rules for --------------------
    labitems = read_table(root, "hosp/d_labitems", usecols=["itemid", "label"])
    wanted = labitems[labitems["label"].isin(LAB_LABEL_MAP)]
    itemid_to_name = {
        int(r.itemid): LAB_LABEL_MAP[r.label] for r in wanted.itertuples() if pd.notna(r.itemid)
    }
    labs = read_table(
        root, "hosp/labevents", usecols=["hadm_id", "itemid", "charttime", "valuenum"]
    )
    labs = labs[labs["hadm_id"].isin(keep) & labs["itemid"].isin(itemid_to_name)].copy()
    labs = labs.dropna(subset=["valuenum"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    for seq, row in enumerate(labs.itertuples()):
        add(
            _event(
                int(row.hadm_id),
                seq,
                "lab",
                row.charttime,
                itemid_to_name[int(row.itemid)],
                f"{row.valuenum:g}",
            ),
            int(row.hadm_id),
        )

    # --- medications actually administered --------------------------------------
    emar = read_table(
        root, "hosp/emar", usecols=["hadm_id", "charttime", "medication", "event_txt"]
    )
    emar = emar[
        emar["hadm_id"].isin(keep)
        & emar["event_txt"].isin(EMAR_ADMINISTERED)
        & emar["medication"].notna()
    ].copy()
    emar["charttime"] = pd.to_datetime(emar["charttime"])
    for seq, row in enumerate(emar.itertuples()):
        add(
            _event(
                int(row.hadm_id),
                seq,
                "medication",
                row.charttime,
                row.medication,
                f"Administered ({row.event_txt}).",
            ),
            int(row.hadm_id),
        )

    # --- blood products, which live in inputevents rather than emar -------------
    items = read_table(root, "icu/d_items", usecols=["itemid", "label", "category"])
    blood = items[items["category"] == BLOOD_PRODUCT_CATEGORY]
    blood = blood[
        ~blood["label"].fillna("").str.lower().str.contains("|".join(BLOOD_PRODUCT_EXCLUDE))
    ]
    blood_names = {int(r.itemid): str(r.label) for r in blood.itertuples()}
    inputs = read_table(
        root, "icu/inputevents", usecols=["hadm_id", "itemid", "starttime", "amount", "amountuom"]
    )
    inputs = inputs[inputs["hadm_id"].isin(keep) & inputs["itemid"].isin(blood_names)].copy()
    inputs["starttime"] = pd.to_datetime(inputs["starttime"])
    for seq, row in enumerate(inputs.itertuples()):
        amount = f"{row.amount:g} {row.amountuom}" if pd.notna(row.amount) else "amount not recorded"
        add(
            _event(
                int(row.hadm_id),
                seq,
                "medication",
                row.starttime,
                blood_names[int(row.itemid)],
                f"Transfused {amount}.",
            ),
            int(row.hadm_id),
        )

    # --- procedures ------------------------------------------------------------
    procedures = read_table(
        root, "hosp/procedures_icd", usecols=["hadm_id", "chartdate", "icd_code", "icd_version"]
    )
    titles = read_table(
        root, "hosp/d_icd_procedures", usecols=["icd_code", "icd_version", "long_title"]
    )
    procedures = procedures[procedures["hadm_id"].isin(keep)].merge(
        titles, on=["icd_code", "icd_version"], how="left"
    )
    # procedures_icd carries a date but no time, so these land at midday. Ordering
    # against same-day labs is therefore approximate.
    procedures["when"] = pd.to_datetime(procedures["chartdate"]) + pd.Timedelta(hours=12)
    for seq, row in enumerate(procedures.itertuples()):
        title = row.long_title if pd.notna(row.long_title) else f"ICD-{row.icd_version} {row.icd_code}"
        add(
            _event(
                int(row.hadm_id),
                seq,
                "procedure",
                row.when,
                title,
                f"Coded procedure, date only (ICD-{row.icd_version} {row.icd_code}).",
            ),
            int(row.hadm_id),
        )

    for hadm_id in collected:
        collected[hadm_id].sort(key=lambda e: e.timestamp)
    return collected
