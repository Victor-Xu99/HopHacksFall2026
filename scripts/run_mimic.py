"""Run the pipeline against real MIMIC-IV demo admissions.

Reports what the dataset can support before reporting any model numbers, because
a trigger that cannot fire is more important to know about than one that does.
"""

from collections import Counter

import numpy as np

from src.data.mimic import MIMIC_ROOT_DEFAULT, load_mimic_cases, mimic_available
from src.engine.model import HarmScoringModel
from src.engine.watcher import FEATURE_DESCRIPTIONS, StructuredDataWatcher


def main() -> None:
    if not mimic_available():
        raise SystemExit(f"MIMIC demo not found under {MIMIC_ROOT_DEFAULT}")

    cases = load_mimic_cases()
    positives = sum(c.is_harm_event for c in cases)
    event_total = sum(len(c.events) for c in cases)
    print(f"admissions {len(cases)}   events {event_total}   mean {event_total / len(cases):.0f}/admission")
    print(f"proxy-labeled harm {positives}/{len(cases)} = {positives / len(cases):.1%}")
    print(f"note events: {sum(len(c.events_of_type('note')) for c in cases)} (demo excludes notes)\n")

    watcher = StructuredDataWatcher(anchor="admission")

    print("--- trigger coverage on real data ---")
    fired = Counter()
    for case in cases:
        for name, value in watcher.extract_features(case).items():
            if value > 0:
                fired[name] += 1
    for name in FEATURE_DESCRIPTIONS:
        count = fired.get(name, 0)
        verdict = "NEVER FIRES" if count == 0 else f"{count / len(cases):6.1%}"
        print(f"  {name:36s} {count:4d} admissions  {verdict}")

    print("\n--- lift: how often each trigger coincides with a coded complication ---")
    base = positives / len(cases)
    rows = []
    for name in FEATURE_DESCRIPTIONS:
        with_trigger = [c for c in cases if watcher.extract_features(c).get(name, 0.0) > 0]
        if len(with_trigger) < 5:
            continue
        rate = sum(c.is_harm_event for c in with_trigger) / len(with_trigger)
        rows.append((rate / base, rate, len(with_trigger), name))
    for lift, rate, n, name in sorted(rows, reverse=True):
        print(f"  {name:36s} n={n:4d}  harm rate {rate:5.1%}  lift {lift:4.2f}x  (base {base:.1%})")

    print("\n--- structured-only model trained on the ICD proxy label ---")
    for model_type in ("logistic", "tree"):
        model = HarmScoringModel([watcher], model_type=model_type, threshold=0.4, random_state=3)
        report = model.train(cases)
        print(f"\n{report.summary()}")
        print(f"  cross-validated ROC AUC {report.cv_roc_auc_mean:.3f} +/- {report.cv_roc_auc_std:.3f}")
        print(f"  confusion (tn, fp, fn, tp) = {report.confusion}   n_train={report.n_train} n_test={report.n_test}")
        if model_type == "logistic":
            print("  learned weights:")
            for row in model.feature_table():
                print(f"    {row['feature']:36s} {row['log_odds_weight']:+.3f}")

    print("\n--- top of the review queue (logistic) ---")
    model = HarmScoringModel([watcher], model_type="logistic", threshold=0.4, random_state=3)
    model.train(cases)
    scored = sorted(cases, key=model.predict_score, reverse=True)
    for case in scored[:5]:
        print(f"\n  {case.patient_id}  score {model.predict_score(case):.2f}  coded complication={case.is_harm_event}")
        for contribution in model.get_evidence(case)[:4]:
            print(f"      {contribution.feature:34s} {contribution.contribution:+.2f}")
        for note in watcher.explain(case)[:4]:
            print(f"      trigger: {note}")

    print("\n--- precision at review capacity ---")
    scores = np.array([model.predict_score(c) for c in cases])
    labels = np.array([c.is_harm_event for c in cases])
    order = np.argsort(-scores)
    for k in (10, 25, 50, 100):
        top = labels[order[:k]]
        print(f"  reviewing top {k:3d}: precision {top.mean():5.1%}, catches {top.sum():2d}/{labels.sum()} coded complications")


if __name__ == "__main__":
    main()
