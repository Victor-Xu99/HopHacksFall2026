"""Break the one-to-one tie between a harm label and its giveaway evidence.

The warehouse was planted so that every harm archetype carries a pathognomonic
pair: postoperative hemorrhage always has PRBC and a laparotomy reopening,
oversedation always has naloxone and an intubation. Nothing else ever carries
them. A model trained on that learns the generator's planting rule, scores 0.95,
and would collapse on real data where naloxone is given to patients who turn out
fine and half of all harm is never recognised at all.

Two mutations, both of which leave `conditions` untouched so the labels are
exactly the ones a reviewer assigned:

  silent harm      strip the signature evidence from a share of harm
                   encounters. These become harm that left no smoking gun,
                   which is the population the OIG report is actually about.
                   A further slice also loses its abnormal labs, leaving harm
                   that is genuinely undetectable -- a ceiling on recall that
                   an honest demo should show rather than hide.

  benign triggers  give a share of non-harm encounters the same signature
                   evidence, with a reason that explains it away: protamine at
                   the end of bypass, naloxone reversing procedural sedation,
                   vitamin K for a nutritional deficiency. The watcher already
                   assumes these exist; the data never contained any.

Every change is logged to dbo.safetynet_deleak_log with the full original row,
so --revert restores the warehouse exactly. Injected rows carry CODE =
'SN-DELEAK', a column no adapter reads.

Usage:
    python -m scripts.deleak_safetyhops --dry-run
    python -m scripts.deleak_safetyhops
    python -m scripts.deleak_safetyhops --revert
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from src.data.safetyhops import (
    DEFAULT_DATABASE,
    DEFAULT_SERVER,
    HARM_CONDITIONS,
    _connection_url,
)

SENTINEL = "SN-DELEAK"
LOG_TABLE = "dbo.safetynet_deleak_log"

# The planted pair per archetype: the drug and the procedure that currently
# appear on harm encounters and nowhere else.
SIGNATURE_MEDS = (
    "Packed Red Blood Cells (PRBC)",
    "Vancomycin 1g",
    "Dextrose 50% Injectable Solution",
    "Sodium bicarbonate 8.4%",
    "Phytonadione 10 MG/ML Injectable Solution",
    "Flumazenil 0.1 MG/ML Injectable Solution",
    "Protamine Sulfate 10 MG/ML Injectable Solution",
    "Naloxone 0.4 MG/ML Injectable Solution",
    "Cefazolin 1g",
    "Calcium gluconate 10% IV Solution",
)

SIGNATURE_PROCS = (
    "Reopening of recent laparotomy site to control hemorrhage",
    "Incision and drainage of wound infection",
    "Intravenous catheter insertion",
    "Hemodialysis catheter placement",
    "Control of hemorrhage",
    "Bag-valve mask ventilation",
    "Endoscopic control of gastric bleeding",
    "Emergency endotracheal intubation",
    "Repair of accidental surgical laceration",
    "Continuous cardiac monitoring",
)

# Every signature drug needs a benign counterpart or it stays a perfect tell.
# A reversal agent with a routine explanation is the whole point: the reviewer
# has to read the reason, and so does the model.
BENIGN_MEDS: Tuple[Tuple[str, str], ...] = (
    ("Naloxone 0.4 MG/ML Injectable Solution",
     "Reversal of procedural sedation at end of endoscopy; patient recovered uneventfully."),
    ("Protamine Sulfate 10 MG/ML Injectable Solution",
     "Routine heparin reversal at separation from cardiopulmonary bypass."),
    ("Phytonadione 10 MG/ML Injectable Solution",
     "Nutritional vitamin K repletion for prolonged poor oral intake."),
    ("Calcium gluconate 10% IV Solution",
     "Scheduled calcium repletion for hypocalcaemia of chronic kidney disease."),
    ("Dextrose 50% Injectable Solution",
     "Protocol treatment of asymptomatic hypoglycaemia; resolved, no sequelae."),
    ("Packed Red Blood Cells (PRBC)",
     "Elective transfusion for chronic symptomatic anaemia."),
    ("Flumazenil 0.1 MG/ML Injectable Solution",
     "Reversal of midazolam after planned sedation; expected and documented."),
    ("Cefazolin 1g",
     "Standard single-dose surgical antibiotic prophylaxis on induction."),
    ("Vancomycin 1g",
     "Surgical prophylaxis for a penicillin-allergic patient with MRSA colonisation."),
    ("Sodium bicarbonate 8.4%",
     "Correction of chronic metabolic acidosis of established kidney disease."),
)

# Coded reoperations are the hardest confounder and the most honest one: a
# procedure code says a cavity was reopened, never whether that was the plan.
# A planned second-look laparotomy trips the same trigger as a catastrophe, and
# a watcher that cannot tell them apart should be shown failing to.
BENIGN_PROCS: Tuple[Tuple[str, str], ...] = (
    ("Intravenous catheter insertion", "Routine peripheral access for scheduled infusion."),
    ("Continuous cardiac monitoring", "Standard telemetry for an observation admission."),
    ("Emergency endotracheal intubation", "Planned airway management for elective general anaesthesia."),
    ("Bag-valve mask ventilation", "Pre-oxygenation during planned induction of anaesthesia."),
    ("Reopening of recent laparotomy site to control hemorrhage",
     "Planned second-look laparotomy after staged damage-control surgery; no new bleeding found."),
    ("Control of hemorrhage",
     "Expected intraoperative haemostasis during a scheduled resection."),
    ("Endoscopic control of gastric bleeding",
     "Elective endoscopic banding of known varices; prophylactic, no acute bleed."),
    ("Repair of accidental surgical laceration",
     "Planned repair of a deliberate serosal incision made for exposure."),
    ("Incision and drainage of wound infection",
     "Scheduled drainage of a pre-existing chronic abscess present on admission."),
    ("Hemodialysis catheter placement",
     "Elective access placement for maintenance dialysis in established renal failure."),
)

# Values that read as critical to the watcher, for non-harm encounters whose
# results drifted and came back. Keyed to how src/engine/watcher.py reads labs.
BENIGN_CRITICAL_LABS: Tuple[Tuple[str, str, str], ...] = (
    ("Hemoglobin", "6.8", "g/dL"),
    ("Creatinine", "2.7", "mg/dL"),
    ("Potassium", "6.6", "mmol/L"),
    ("Glucose", "48", "mg/dL"),
    ("INR", "4.7", "{INR}"),
)

LOG_DDL = f"""
IF OBJECT_ID('{LOG_TABLE}', 'U') IS NULL
CREATE TABLE {LOG_TABLE} (
    id          INT IDENTITY(1,1) PRIMARY KEY,
    run_id      VARCHAR(36)   NOT NULL,
    applied_at  DATETIME2(0)  NOT NULL,
    action      VARCHAR(10)   NOT NULL,
    table_name  VARCHAR(50)   NOT NULL,
    encounter   VARCHAR(50)   NULL,
    row_json    NVARCHAR(MAX) NOT NULL
);
"""


# ----------------------------------------------------------------- helpers


def _shift(timestamp: str, hours: int) -> str:
    """An injected event has to land after admission or the watcher ignores it."""
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp("2023-01-01T00:00:00")
    return (parsed + timedelta(hours=hours)).isoformat()


def _applied_runs(connection) -> List[str]:
    return [
        row[0]
        for row in connection.execute(
            text(f"SELECT DISTINCT run_id FROM {LOG_TABLE}")
        ).fetchall()
    ]


def _log(connection, run_id: str, action: str, table: str, encounter: str, row: dict) -> None:
    connection.execute(
        text(
            f"INSERT INTO {LOG_TABLE} (run_id, applied_at, action, table_name, encounter, row_json) "
            "VALUES (:r, :t, :a, :tab, :e, :j)"
        ),
        {
            "r": run_id,
            "t": datetime.now(),
            "a": action,
            "tab": table,
            "e": encounter,
            "j": json.dumps(row, default=str),
        },
    )


# ------------------------------------------------------------------ revert


def revert(connection) -> Dict[str, int]:
    runs = _applied_runs(connection)
    if not runs:
        return {}

    restored = 0
    for table, encounter, row_json in connection.execute(
        text(
            f"SELECT table_name, encounter, row_json FROM {LOG_TABLE} "
            "WHERE action = 'delete' ORDER BY id"
        )
    ).fetchall():
        row = json.loads(row_json)
        columns = ", ".join(row)
        params = ", ".join(f":{c}" for c in row)
        connection.execute(text(f"INSERT INTO dbo.{table} ({columns}) VALUES ({params})"), row)
        restored += 1

    removed = 0
    for table in ("medications", "procedures", "observations"):
        result = connection.execute(text(f"DELETE FROM dbo.{table} WHERE CODE = :s"), {"s": SENTINEL})
        removed += result.rowcount or 0

    connection.execute(text(f"DELETE FROM {LOG_TABLE}"))
    return {"restored": restored, "removed": removed, "runs": len(runs)}


# ------------------------------------------------------------------- apply


def apply(
    connection,
    run_id: str,
    silent_share: float,
    dark_share: float,
    benign_share: float,
    seed: int,
    dry_run: bool,
) -> Dict[str, int]:
    rng = random.Random(seed)

    encounters = pd.read_sql("SELECT Id, START, PATIENT FROM dbo.encounters", connection)
    conditions = pd.read_sql("SELECT ENCOUNTER, DESCRIPTION FROM dbo.conditions", connection)

    harm_ids: Set[str] = set(
        conditions.loc[conditions["DESCRIPTION"].isin(HARM_CONDITIONS), "ENCOUNTER"]
    )
    harm = [e for e in encounters.itertuples() if e.Id in harm_ids]
    benign = [e for e in encounters.itertuples() if e.Id not in harm_ids]
    rng.shuffle(harm)
    rng.shuffle(benign)

    n_silent = int(len(harm) * silent_share)
    silent = harm[:n_silent]
    dark = silent[: int(len(harm) * dark_share)]
    dark_ids = {e.Id for e in dark}
    chosen_benign = benign[: int(len(benign) * benign_share)]

    stats = {
        "harm": len(harm),
        "benign": len(benign),
        "silenced": len(silent),
        "darkened": len(dark),
        "benign_triggered": len(chosen_benign),
        "rows_deleted": 0,
        "rows_inserted": 0,
    }
    if dry_run:
        return stats

    silent_ids = [e.Id for e in silent]

    # --- strip the signature evidence from the silenced harm encounters -----
    for table, signatures in (("medications", SIGNATURE_MEDS), ("procedures", SIGNATURE_PROCS)):
        for chunk_start in range(0, len(silent_ids), 400):
            chunk = silent_ids[chunk_start : chunk_start + 400]
            enc_params = {f"e{i}": v for i, v in enumerate(chunk)}
            sig_params = {f"s{i}": v for i, v in enumerate(signatures)}
            where = (
                f"ENCOUNTER IN ({', '.join(':' + k for k in enc_params)}) "
                f"AND DESCRIPTION IN ({', '.join(':' + k for k in sig_params)})"
            )
            doomed = pd.read_sql(
                text(f"SELECT * FROM dbo.{table} WHERE {where}"),
                connection,
                params={**enc_params, **sig_params},
            )
            for row in doomed.to_dict("records"):
                _log(connection, run_id, "delete", table, row.get("ENCOUNTER"), row)
            deleted = connection.execute(
                text(f"DELETE FROM dbo.{table} WHERE {where}"), {**enc_params, **sig_params}
            )
            stats["rows_deleted"] += deleted.rowcount or 0

    # --- and take the abnormal labs off the fully dark ones -----------------
    dark_list = list(dark_ids)
    for chunk_start in range(0, len(dark_list), 400):
        chunk = dark_list[chunk_start : chunk_start + 400]
        enc_params = {f"e{i}": v for i, v in enumerate(chunk)}
        where = f"ENCOUNTER IN ({', '.join(':' + k for k in enc_params)})"
        observations = pd.read_sql(
            text(f"SELECT * FROM dbo.observations WHERE {where}"), connection, params=enc_params
        )
        for row in observations.to_dict("records"):
            _log(connection, run_id, "delete", "observations", row.get("ENCOUNTER"), row)
        deleted = connection.execute(text(f"DELETE FROM dbo.observations WHERE {where}"), enc_params)
        stats["rows_deleted"] += deleted.rowcount or 0

    # --- give benign encounters the evidence, with an innocent reason -------
    # Round-robin rather than random choice: a signature the sampler happened to
    # skip would stay a perfect tell, which is the bug this script exists to fix.
    med_rows, proc_rows, obs_rows = [], [], []
    for index, encounter in enumerate(chosen_benign):
        drug, reason = BENIGN_MEDS[index % len(BENIGN_MEDS)]
        med_rows.append(
            {
                "START": _shift(encounter.START, rng.randint(4, 30)),
                "STOP": "",
                "PATIENT": encounter.PATIENT,
                "PAYER": "",
                "ENCOUNTER": encounter.Id,
                "CODE": SENTINEL,
                "DESCRIPTION": drug,
                "BASE_COST": "0",
                "PAYER_COVERAGE": "0",
                "DISPENSES": "1",
                "TOTALCOST": "0",
                "REASONCODE": "",
                "REASONDESCRIPTION": reason,
            }
        )
        if rng.random() < 0.85:
            procedure, proc_reason = BENIGN_PROCS[index % len(BENIGN_PROCS)]
            proc_rows.append(
                {
                    "START": _shift(encounter.START, rng.randint(2, 20)),
                    "STOP": "",
                    "PATIENT": encounter.PATIENT,
                    "ENCOUNTER": encounter.Id,
                    "SYSTEM": "SNOMED-CT",
                    "CODE": SENTINEL,
                    "DESCRIPTION": procedure,
                    "BASE_COST": "0",
                    "REASONCODE": "",
                    "REASONDESCRIPTION": proc_reason,
                }
            )
        if rng.random() < 0.45:
            lab, value, units = rng.choice(BENIGN_CRITICAL_LABS)
            obs_rows.append(
                {
                    "DATE": _shift(encounter.START, rng.randint(6, 40)),
                    "PATIENT": encounter.PATIENT,
                    "ENCOUNTER": encounter.Id,
                    "CATEGORY": "laboratory",
                    "CODE": SENTINEL,
                    "DESCRIPTION": lab,
                    "VALUE": value,
                    "UNITS": units,
                    "TYPE": "numeric",
                }
            )

    for table, rows in (
        ("medications", med_rows),
        ("procedures", proc_rows),
        ("observations", obs_rows),
    ):
        if not rows:
            continue
        columns = ", ".join(rows[0])
        params = ", ".join(f":{c}" for c in rows[0])
        statement = text(f"INSERT INTO dbo.{table} ({columns}) VALUES ({params})")
        for chunk_start in range(0, len(rows), 500):
            connection.execute(statement, rows[chunk_start : chunk_start + 500])
        stats["rows_inserted"] += len(rows)
        _log(connection, run_id, "insert", table, None, {"count": len(rows)})

    return stats


# -------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument(
        "--silent-share", type=float, default=0.45,
        help="Share of harm encounters that lose their signature evidence.",
    )
    parser.add_argument(
        "--dark-share", type=float, default=0.15,
        help="Share of harm encounters that also lose their abnormal labs.",
    )
    parser.add_argument(
        "--benign-share", type=float, default=0.18,
        help="Share of non-harm encounters that gain explained-away evidence.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true", help="Revert a previous run, then reapply.")
    args = parser.parse_args(argv)

    engine = create_engine(_connection_url(args.server, args.database))
    with engine.begin() as connection:
        connection.execute(text(LOG_DDL))

        if args.revert:
            result = revert(connection)
            if not result:
                print("nothing to revert.")
            else:
                print(
                    f"reverted {result['runs']} run(s): restored {result['restored']} rows, "
                    f"removed {result['removed']} injected rows."
                )
            return 0

        if _applied_runs(connection):
            if not args.force:
                print("already de-leaked. Use --revert to undo, or --force to redo.")
                return 1
            revert(connection)
            print("previous run reverted.")

        run_id = str(uuid.uuid4())
        stats = apply(
            connection,
            run_id=run_id,
            silent_share=args.silent_share,
            dark_share=args.dark_share,
            benign_share=args.benign_share,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        print(f"{'planned' if args.dry_run else 'applied'} run {run_id[:8]}")
        print(f"  harm encounters            {stats['harm']}")
        print(f"  non-harm encounters        {stats['benign']}")
        print(f"  harm stripped of signature {stats['silenced']}")
        print(f"  ...also stripped of labs   {stats['darkened']}")
        print(f"  non-harm given evidence    {stats['benign_triggered']}")
        if not args.dry_run:
            print(f"  rows deleted               {stats['rows_deleted']}")
            print(f"  rows inserted              {stats['rows_inserted']}")
            print("\nre-run the audit:  python -m scripts.audit_leakage --source safetyhops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
