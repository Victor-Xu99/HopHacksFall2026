"""Score the note reader, and price it against a keyword list that needs no training.

Usage:
    python -m scripts.eval_note_reader
    python -m scripts.eval_note_reader --skip-ablation
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from src.engine.note_eval import compare_on_contrast, compare_sentence_level, run_ablation


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args(argv)

    reports = compare_sentence_level(n_folds=args.folds, random_state=args.seed)

    print("\nsentence level, stratified %d-fold over the seed corpus" % args.folds)
    print("=" * 72)
    for report in reports:
        print(f"\n{report.summary()}")
        print(f"{'class':<16}{'precision':>11}{'recall':>9}{'f1':>8}{'n':>6}")
        for label, scores in report.per_class.items():
            print(
                f"{label:<16}{scores['precision']:>11.3f}{scores['recall']:>9.3f}"
                f"{scores['f1']:>8.3f}{scores['support']:>6}"
            )

    trained, lexicon = reports
    delta = trained.macro_f1_mean - lexicon.macro_f1_mean
    print(f"\nclassifier over lexicon: {delta:+.3f} macro F1")
    print("\nconfusion, trained classifier (rows are truth)")
    print(trained.confusion_table())

    print("\n\nheld-out contrast pairs, same event with framing flipped")
    print("=" * 72)
    contrast = compare_on_contrast()
    for report in contrast:
        print(report.summary())

    for report in contrast:
        print(f"\nmisses, {report.name} ({len(report.misses)} of {report.n_pairs * 2})")
        for text, truth, predicted in report.misses[:8]:
            print(f"  want {truth:<14} got {predicted:<14} {text[:58]}")

    if not args.skip_ablation:
        print("\n\nend to end, synthetic cohort")
        print("=" * 72)
        for arm in run_ablation(seed=args.seed):
            print(arm.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
