"""Report label leakage for a cohort, so an AUC is read next to what produced it.

Usage:
    python -m scripts.audit_leakage --source safetyhops
    python -m scripts.audit_leakage --source synthetic
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from src.engine.leakage import audit
from src.engine.watcher import StructuredDataWatcher


def _load(source: str):
    """Returns (cases, watcher). Kept local so the audit has no app dependency."""
    if source == "safetyhops":
        from src.data.safetyhops import load_safetyhops_cases

        return load_safetyhops_cases(), StructuredDataWatcher(anchor="admission")
    if source == "mimic":
        from src.data.mimic import load_mimic_cases

        return load_mimic_cases(), StructuredDataWatcher(anchor="admission")
    if source == "synthetic":
        from src.data.generator import generate_dataset

        return generate_dataset(num_cases=600, seed=7), StructuredDataWatcher()
    raise SystemExit(f"unknown source {source!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="safetyhops", choices=("safetyhops", "mimic", "synthetic")
    )
    args = parser.parse_args(argv)

    cases, watcher = _load(args.source)
    report = audit(cases, watcher)

    print(f"\n{args.source}: {report.n_cases} cases, base rate {report.base_rate:.1%}")
    print(f"perfect-separation ceiling is {1 / report.base_rate:.2f}x lift\n")

    print(f"{'feature':<38}{'fires':>7}{'lift':>8}{'of max':>9}")
    for finding in report.features:
        flag = "  <-- LEAKING" if finding.leaking else ""
        print(
            f"{finding.feature:<38}{finding.support:>7}"
            f"{finding.lift:>7.2f}x{finding.separation:>8.0%}{flag}"
        )
    for name in report.dead_features:
        print(f"{name:<38}{0:>7}{'never fires':>17}")

    if report.giveaways:
        print(f"\nevent values that all but determine the label:")
        for value in report.giveaways[:12]:
            print(
                f"  {value.label_rate:>6.1%} labeled  n={value.support:<5} "
                f"{value.event_type}: {value.value[:54]}"
            )

    print(f"\n{report.summary()}")
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
