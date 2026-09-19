"""Guess file roles and column names from a hospital folder.

The hospital does not have to use our filenames or headers. We still need four
kinds of tables: stays, labs, meds, transfers.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

ROLES = ("stays", "labs", "meds", "transfers")

FILE_HINTS: Dict[str, Tuple[str, ...]] = {
    "stays": ("stay", "encounter", "adt", "admission", "census"),
    "labs": ("lab", "observation", "result", "component"),
    "meds": ("med", "mar", "medication", "drug"),
    "transfers": ("transfer", "unit_move", "unit", "department", "ward"),
}

FIELD_ALIASES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "stays": {
        "encounter_id": ("encounter_id", "pat_enc_csn_id", "csn", "visit_id", "encounter", "hadm_id"),
        "age": ("age",),
        "sex": ("sex", "gender"),
        "admitted": ("admitted", "hosp_admsn_time", "admit_time", "admission_time", "start"),
        "discharged": ("discharged", "hosp_disch_time", "discharge_time", "disch_time", "stop"),
    },
    "labs": {
        "encounter_id": ("encounter_id", "pat_enc_csn_id", "csn", "visit_id", "encounter", "hadm_id"),
        "time": ("time", "date", "result_time", "taken_time", "observation_time"),
        "name": ("name", "component_name", "lab_name", "description", "test"),
        "value": ("value", "ord_value", "result", "numeric_value"),
        "units": ("units", "unit"),
    },
    "meds": {
        "encounter_id": ("encounter_id", "pat_enc_csn_id", "csn", "visit_id", "encounter", "hadm_id"),
        "time": ("time", "date", "taken_time", "start", "administered_at"),
        "name": ("name", "medication_name", "drug", "description", "med"),
    },
    "transfers": {
        "encounter_id": ("encounter_id", "pat_enc_csn_id", "csn", "visit_id", "encounter", "hadm_id"),
        "time": ("time", "date", "effective_time", "start"),
        "to_unit": ("to_unit", "department_name", "unit", "ward", "department"),
    },
}

REQUIRED: Dict[str, Tuple[str, ...]] = {
    "stays": ("encounter_id", "admitted"),
    "labs": ("encounter_id", "time", "name", "value"),
    "meds": ("encounter_id", "time", "name"),
    "transfers": ("encounter_id", "time", "to_unit"),
}

NAME_SYNONYMS = {
    "narcan": "Naloxone",
    "naloxone hcl": "Naloxone",
    "naloxone hydrochloride": "Naloxone",
    "creat": "Creatinine",
    "creatinine serpl": "Creatinine",
    "wbc count": "WBC",
    "white blood cell": "WBC",
    "lactate venous": "Lactate",
    "lactic acid": "Lactate",
}


def normalize_name(value: str) -> str:
    key = " ".join(str(value).strip().lower().split())
    return NAME_SYNONYMS.get(key, str(value).strip())


def _norm_col(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")


def _match_field(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    normalized = {_norm_col(col): col for col in columns}
    for alias in aliases:
        key = _norm_col(alias)
        if key in normalized:
            return normalized[key]
    return None


def map_columns(frame: pd.DataFrame, role: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    aliases = FIELD_ALIASES[role]
    mapping: Dict[str, str] = {}
    rename: Dict[str, str] = {}
    for field, options in aliases.items():
        source = _match_field(frame.columns, options)
        if source is not None:
            mapping[field] = source
            rename[source] = field
    missing = [field for field in REQUIRED[role] if field not in mapping]
    if missing:
        raise ValueError(
            f"{role} table is missing {', '.join(missing)}. "
            f"Columns we saw: {', '.join(frame.columns)}"
        )
    mapped = frame.rename(columns=rename)
    keep = list(aliases)
    for field in keep:
        if field not in mapped.columns:
            mapped[field] = ""
    return mapped[keep], mapping


def _filename_score(name: str, role: str) -> int:
    stem = Path(name).stem.lower()
    score = 0
    if stem == role or stem == f"{role}s":
        score += 5
    for hint in FILE_HINTS[role]:
        if hint in stem:
            score += 2
    return score


def _column_score(columns: Iterable[str], role: str) -> int:
    score = 0
    for field, aliases in FIELD_ALIASES[role].items():
        if _match_field(columns, aliases):
            score += 3 if field in REQUIRED[role] else 1
    return score


def list_tables(folder: Path) -> List[Path]:
    folder = Path(folder)
    files = list(folder.glob("*.csv"))
    files.extend(folder.glob("*/*.csv"))
    # Prefer unique paths, skip macOS junk
    unique = []
    seen = set()
    for path in files:
        if path.name.startswith(".") or path.name.startswith("~"):
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def classify_folder(folder: Path) -> Tuple[Dict[str, Path], Dict[str, object]]:
    """Assign each CSV to stays/labs/meds/transfers. Raises if a role is missing."""
    candidates = []
    for path in list_tables(folder):
        header = pd.read_csv(path, dtype=str, nrows=0)
        columns = list(header.columns)
        scores = {role: _filename_score(path.name, role) + _column_score(columns, role) for role in ROLES}
        candidates.append((path, columns, scores))
    if not candidates:
        raise ValueError("That folder has no CSV files.")

    assigned: Dict[str, Path] = {}
    leftover = list(candidates)
    for _ in ROLES:
        best_role, best_path, best_score = "", None, 0
        for path, _columns, scores in leftover:
            for role, score in scores.items():
                if role in assigned:
                    continue
                if score > best_score:
                    best_role, best_path, best_score = role, path, score
        if best_path is None or best_score <= 0:
            break
        assigned[best_role] = best_path
        leftover = [item for item in leftover if item[0] != best_path]

    missing = [role for role in ROLES if role not in assigned]
    if missing:
        found = ", ".join(f"{path.name}→{role}" for role, path in assigned.items()) or "nothing"
        raise ValueError(f"Could not find tables for: {', '.join(missing)}. Mapped so far: {found}.")

    report = {
        "files": {role: assigned[role].name for role in ROLES},
        "columns": {},
    }
    return assigned, report


def load_mapped_tables(folder: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    assigned, report = classify_folder(folder)
    tables: Dict[str, pd.DataFrame] = {}
    for role, path in assigned.items():
        raw = pd.read_csv(path, dtype=str).fillna("")
        mapped, column_map = map_columns(raw, role)
        tables[role] = mapped
        report["columns"][role] = column_map
    return tables, report


def unpack_zip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(dest)
    csvs = list(dest.glob("*.csv")) + list(dest.glob("*/*.csv"))
    if not csvs:
        raise ValueError("That zip has no CSV files.")
    parents = {path.parent for path in dest.glob("*.csv")}
    nested = {path.parent for path in dest.glob("*/*.csv")}
    if not parents and len(nested) == 1:
        return next(iter(nested))
    return dest


def write_bundle(folder: Path, files: Iterable[Tuple[str, bytes]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, payload in files:
        filename = Path(name).name
        if not filename.lower().endswith(".csv") and not filename.lower().endswith(".zip"):
            continue
        (folder / filename).write_bytes(payload)
        written += 1
    if written == 0:
        raise ValueError("Upload a folder of CSVs or one zip of CSVs.")
    zips = list(folder.glob("*.zip"))
    if len(zips) == 1 and written == 1:
        return unpack_zip(zips[0], folder / "_unzipped")
    return folder
