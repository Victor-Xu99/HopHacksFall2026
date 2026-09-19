"""One-off reconnaissance of the MIMIC-IV demo: what can the watcher actually see?"""

from pathlib import Path

import pandas as pd

from src.data.mimic import ANTIDOTE_PATTERN, MIMIC_ROOT_DEFAULT, read_table


def main() -> None:
    root = Path(MIMIC_ROOT_DEFAULT)
    print(f"root: {root}\n")

    patients = read_table(root, "hosp/patients")
    admissions = read_table(root, "hosp/admissions")
    print(f"patients {len(patients)}   admissions {len(admissions)}")
    print("admission types:", admissions["admission_type"].value_counts().to_dict())
    print()

    transfers = read_table(root, "hosp/transfers")
    print("careunits:")
    print(transfers["careunit"].value_counts().head(20).to_string())
    print("\neventtypes:", transfers["eventtype"].value_counts().to_dict())
    print()

    icustays = read_table(root, "icu/icustays")
    print(f"icustays {len(icustays)} across {icustays['hadm_id'].nunique()} admissions")
    print("first careunit:", icustays["first_careunit"].value_counts().to_dict())
    print()

    emar = read_table(root, "hosp/emar", usecols=["hadm_id", "charttime", "medication", "event_txt"])
    print(f"emar rows {len(emar)}   distinct medications {emar['medication'].nunique()}")
    print("event_txt:", emar["event_txt"].value_counts().head(8).to_dict())
    hits = emar[emar["medication"].fillna("").str.lower().str.contains(ANTIDOTE_PATTERN, regex=True)]
    print(f"\nantidote administrations in emar: {len(hits)}")
    if len(hits):
        print(hits["medication"].value_counts().to_string())
        print("distinct admissions with an antidote:", hits["hadm_id"].nunique())
    print()

    prescriptions = read_table(root, "hosp/prescriptions", usecols=["hadm_id", "drug"])
    pres_hits = prescriptions[
        prescriptions["drug"].fillna("").str.lower().str.contains(ANTIDOTE_PATTERN, regex=True)
    ]
    print(f"antidote orders in prescriptions: {len(pres_hits)}")
    if len(pres_hits):
        print(pres_hits["drug"].value_counts().to_string())
    print()

    blood = prescriptions[
        prescriptions["drug"].fillna("").str.lower().str.contains("transfus|packed red|prbc|platelet|plasma")
    ]
    print(f"blood-product orders: {len(blood)}", blood["drug"].unique()[:10])
    print()

    labitems = read_table(root, "hosp/d_labitems")
    labevents = read_table(
        root,
        "hosp/labevents",
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "valueuom", "flag", "ref_range_lower", "ref_range_upper"],
    )
    labevents = labevents.merge(labitems[["itemid", "label", "fluid"]], on="itemid", how="left")
    print(f"labevents {len(labevents)}   distinct labs {labevents['label'].nunique()}")
    print("flag values:", labevents["flag"].value_counts(dropna=False).to_dict())
    print("\nlabs we have rules for, as named in MIMIC:")
    for name in ("Hemoglobin", "Creatinine", "Potassium", "Lactate", "INR(PT)", "Platelet Count", "White Blood Cells", "Glucose", "Troponin T", "Bilirubin, Total", "Alanine Aminotransferase (ALT)"):
        subset = labevents[labevents["label"] == name]
        if len(subset):
            print(f"  {name:32s} n={len(subset):6d} uom={subset['valueuom'].dropna().unique()[:2]}")
        else:
            print(f"  {name:32s} ABSENT")
    print()

    procedures = read_table(root, "hosp/procedures_icd")
    d_proc = read_table(root, "hosp/d_icd_procedures")
    procedures = procedures.merge(d_proc, on=["icd_code", "icd_version"], how="left")
    print(f"procedures_icd rows {len(procedures)}  columns {list(procedures.columns)}")
    print("sample titles:")
    for title in procedures["long_title"].dropna().head(8):
        print("  -", title[:90])
    print()

    services = read_table(root, "hosp/services")
    print("services:", services["curr_service"].value_counts().to_dict())


if __name__ == "__main__":
    main()
