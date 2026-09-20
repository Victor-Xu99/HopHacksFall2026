# SafetyNet

Triage completed hospital stays so a reviewer finds more preventable harm in a
fixed amount of time. It is retrospective quality assurance, not a diagnostic
tool.

The HHS OIG found that about a quarter of Medicare inpatients are harmed, and
about 43% of that harm is preventable — roughly **10% of admissions**. Those
events sit in labs, meds, and transfers that nobody has time to read in full.
SafetyNet ranks the charts.

## What a judge should notice first

The banner at the top of the UI is the pitch:

1. **We caught our own leakage.** SafetyHops first scored ROC AUC 0.95 because
   every harm label carried its own evidence. After silent harm and benign
   triggers, we report about **0.80**.
2. **We fixed the base rate.** The warehouse plants harm at 46%. We thin to
   **10%** (OIG preventable harm) by dropping cases, never duplicating them.
3. **Hospital charts have no labels.** They are ranked with a model trained on
   the labeled synthetic cohort. Extra detail still lets you switch the
   scoring source.

Chips in the header show **Scoring** (the list you are looking at) and
**Trained on** (the labeled cohort that fit the model).

## One command

```bash
pip install -r requirements.txt
python -m scripts.launch
```

Then open http://127.0.0.1:5173. That starts FastAPI and the React UI together.
You need Node.js on PATH for the frontend.

Copy `.env.example` to `.env` and set `SAFETYNET_SQL_SERVER` to your instance
(`XUSHOE` on a default instance, `.\SQLEXPRESS` on a named one). Point
`SAFETYNET_SQL_DATABASE` at `SafetyNetQA` and keep `SAFETYHOPS_SQL_DATABASE=SafetyHops`.

## What the three layers do

| Layer | Module | Job |
| --- | --- | --- |
| 1 Structured watcher | `src/engine/watcher.py` | Red flags: antidotes, unplanned ICU, labs, takebacks |
| 2 Note reader | `src/engine/nlp_reader.py` | Framing in free text (not used on hospital CSV extracts) |
| 3 Scoring | `src/engine/model.py` | Logistic or shallow tree, with per-feature contributions |

Hospital files land in SQL Server (`core` schema). Ranking waits until
`discharged` is filled in. Sample CSVs: `tests/fixtures/hospital/mixed_dates`.

## Other commands

```bash
python -m pytest tests -q
python -m scripts.icu_smoke
streamlit run app.py
```

Setup details and SQL loading: [docs/SETUP.md](docs/SETUP.md).
Hospital CSV shape: [docs/HOSPITAL_EXTRACT.md](docs/HOSPITAL_EXTRACT.md).
