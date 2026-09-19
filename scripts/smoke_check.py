"""Headless end-to-end check: generate, train, score, explain.

Run with `python -m scripts.smoke_check` from the repo root.
"""

from collections import defaultdict

from src.data.generator import generate_dataset
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher


def main() -> None:
    dataset = generate_dataset(num_cases=600, harm_ratio=0.15, hard_negative_ratio=0.35, seed=7)
    watcher, reader = StructuredDataWatcher(), ClinicalNoteReader()

    for model_type in ("logistic", "tree"):
        model = HarmScoringModel([watcher, reader], model_type=model_type, threshold=0.4)
        report = model.train(dataset)
        print("\n" + "=" * 70)
        print(report.summary())
        print(f"cv roc auc {report.cv_roc_auc_mean:.3f} +/- {report.cv_roc_auc_std:.3f}")
        print(f"confusion (tn, fp, fn, tp) = {report.confusion}")

        print("\ntop learned weights:")
        for row in model.feature_table()[:8]:
            print("  ", row)

        scores = defaultdict(list)
        for case in dataset:
            scores[case.scenario].append(model.predict_score(case))
        print("\nmean score by scenario:")
        for scenario, values in sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(f"   {scenario:34s} n={len(values):4d} mean={sum(values) / len(values):.3f}")

        harm_case = next(c for c in dataset if c.is_harm_event)
        hard_negative = next(c for c in dataset if c.scenario == "consented_known_complication")
        for label, case in (("HARM", harm_case), ("HARD NEGATIVE", hard_negative)):
            print(f"\n{label} {case.scenario} score={model.predict_score(case):.3f}")
            for contribution in model.get_evidence(case)[:5]:
                print(
                    f"   {contribution.feature:34s} value={contribution.value:>6.2f} "
                    f"contribution={contribution.contribution:+.3f}"
                )
            for note in watcher.explain(case):
                print(f"   trigger: {note}")
            for finding in reader.explain(case)[:3]:
                print(f"   note[{finding.label} {finding.probability:.0%}]: {finding.text[:90]}")

    print("\n" + "=" * 70)
    print("note classifier top unanticipated terms:", reader.top_terms("unanticipated", 8))
    print("note classifier top expected-risk terms:", reader.top_terms("expected_risk", 8))


if __name__ == "__main__":
    main()
