# Hospital CSV pipeline

SafetyNet does not take a Google Doc or a copy of the hospital chart system.
Upload **one folder** of tables (or a zip of that folder). Filenames and column
headers can be theirs. SafetyNet classifies stays / labs / meds / transfers and
maps columns. A stay is stored immediately and ranked only after a discharge
time is present.

## Files

All four must share `encounter_id`.

**stays.csv**

| encounter_id | age | sex | admitted | discharged |
|---|---|---|---|---|
| 88421 | 67 | F | 2026-09-01T07:12:00 | |

Leave `discharged` empty while the patient is still in house.

**labs.csv** — `encounter_id,time,name,value,units`

**meds.csv** — `encounter_id,time,name`

**transfers.csv** — `encounter_id,time,to_unit`

## How a night works

1. Drop the latest four files in a folder (new rows are enough; repeats are ignored).
2. Ingest. Open stays are stored. Events append.
3. Any stay whose `discharged` just filled in is scored. In-house stays are not ranked.

Sample files (Maria still in house vs after discharge):

```
tests/fixtures/hospital/in_house/
tests/fixtures/hospital/discharged/
```

**In the UI:** Streamlit sidebar → **Hospital extract**, or the teal React page →
**Import a hospital folder**. Drop the folder’s CSVs or one zip. The mapping
report shows which file/column we treated as encounter id, discharge, and so on.

Sample files to try first:

```
tests/fixtures/hospital/in_house/
tests/fixtures/hospital/discharged/
```

```bash
python -m scripts.ingest_hospital --dir tests/fixtures/hospital/in_house
python -m scripts.ingest_hospital --dir tests/fixtures/hospital/discharged
```

Or `POST /api/hospital/upload` with the four files, or `POST /api/hospital/ingest?directory=...`.

The teal dashboard source **Hospital extract (rank after discharge)** lists only
stays that already have a discharge time. The ranker is fit on the synthetic
cohort because hospital extracts do not carry training labels.
