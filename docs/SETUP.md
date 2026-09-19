# Setup

Everything here is reproducible from a public dataset, so there is no shared
database to connect to and no credentials to pass around. You download the same
open-access archive everyone else does and build your own copy locally.

## What you need

- **Python 3.13** (3.11+ should be fine).
- **Nothing else** to run the app on synthetic data or on the MIMIC CSVs.
- **SQL Server 2022 Express** plus the **ODBC Driver 17 for SQL Server** only if you
  want the SQL-backed path. Both are free. The app hides the SQL cohorts and the
  tests skip themselves when SQL Server is absent, so this is genuinely optional.

## 1. Install

```bash
git clone https://github.com/Victor-Xu99/HopHacksFall2026.git
cd HopHacksFall2026
pip install -r requirements.txt
```

## 2. Run it on synthetic data

This needs no dataset and no database:

```bash
streamlit run app.py
```

The sidebar defaults to the synthetic cohort. Use the **Note reader** tab to paste
your own text and watch it get classified — that tab works regardless of which
cohort is loaded.

## 3. Get the real records

The MIMIC-IV **demo** is open access under ODbL. It is 100 real de-identified
patients and needs no credentialing, unlike full MIMIC-IV:

<https://physionet.org/content/mimic-iv-demo/2.2/>

Download the zip and extract it so this path exists:

```
data/mimic-iv-clinical-database-demo-2.2/hosp/admissions.csv.gz
```

`data/` is gitignored. Do not commit any of it, even though this tier is
open access.

Once it is in place, the app offers **MIMIC-IV demo (real records)** in the sidebar,
and this prints trigger coverage, per-trigger lift, and held-out metrics:

```bash
python -m scripts.run_mimic
```

Note that this release **excludes clinical notes** by design — they ship as a
separate credentialed dataset. On real records the note reader is therefore
excluded from the model rather than fed empty features.

## 4. Optional: load into SQL Server

Confirm SQL Server is reachable first. The default instance name used throughout is
`.\SQLEXPRESS`:

```bash
sqlcmd -S ".\SQLEXPRESS" -E -Q "SELECT @@VERSION"
```

Then load. This creates the database, one schema per MIMIC module, and one table
per CSV, and verifies every row count against the source:

```bash
# from an already-extracted data/ directory
python -m scripts.load_mimic_sql

# or extract and load in one step
python -m scripts.load_mimic_sql --zip C:\path\to\mimic-iv-clinical-database-demo-2.2.zip

# a single table, useful for a quick smoke test
python -m scripts.load_mimic_sql --only hosp/patients hosp/admissions

# a different target database
python -m scripts.load_mimic_sql --database SafetyNet
```

Expect roughly **20 minutes** for all 31 tables and ~1.4M rows. It is almost
entirely I/O-bound on row-by-row inserts rather than CPU-bound, so it will look
idle. It is safe to re-run; tables are replaced, not duplicated.

Windows authentication is used throughout. No password is stored anywhere in this
repo, and none is needed.

## 5. Tests

```bash
python -m pytest tests -q
```

Tests that require the MIMIC files or SQL Server skip cleanly when those are
missing, so a clean checkout with no dataset still passes.

## Troubleshooting

**A new database does not appear in SSMS.** Object Explorer caches the tree and
does not notice databases created outside of it. Right-click `Databases` → Refresh.

**`Cannot open server` or `Login failed`.** Check the instance name. A named
instance needs the backslash form `.\SQLEXPRESS`, not `localhost` alone. Confirm the
service is running with `Get-Service MSSQL$SQLEXPRESS`.

**`Data source name not found` from pyodbc.** The ODBC driver is a separate system
install from the Python package. Verify with:

```powershell
Get-OdbcDriver -Platform 64-bit | Where-Object Name -like "*SQL Server*"
```

**Remote access from another machine.** Not configured, and not recommended. TCP/IP
is disabled by default on Express, which is a reasonable default. Build a local copy
with the loader instead — it is faster than configuring firewall rules and avoids
copying patient records across a network.
