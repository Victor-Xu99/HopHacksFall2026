"""Load hospital CSVs into SafetyNet SQL and rank stays that now have a discharge time.

python -m scripts.ingest_hospital --dir tests/fixtures/hospital/in_house
python -m scripts.ingest_hospital --dir tests/fixtures/hospital/discharged
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.hospital_extract import ingest_and_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest hospital CSVs and rank new discharges.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("tests/fixtures/hospital/in_house"),
        help="Folder with stays.csv, labs.csv, meds.csv, transfers.csv",
    )
    parser.add_argument("--model", default="logistic", choices=("logistic", "tree"))
    args = parser.parse_args()
    result = ingest_and_rank(args.dir, model_type=args.model)
    print(result)


if __name__ == "__main__":
    main()
