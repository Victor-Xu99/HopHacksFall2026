"""Resample a cohort to a realistic harm prevalence.

The synthetic warehouse plants harm conditions at roughly 46%, which quietly
contradicts the premise of the whole project. Harm is worth detecting because
it is rare and therefore missed; a cohort that is half harm is not the problem
anyone has, and every metric read off it is distorted:

  - PR AUC nulls at the positive rate, so 0.74 against a 46% base rate is a
    1.6x lift, while the same separation against a 10% base rate is far more.
  - "Precision at review capacity" is meaningless when half the pile is harm,
    because reviewing at random already succeeds every other chart.

The OIG report this project is built around found roughly a quarter of Medicare
inpatients experienced harm and about 43% of that was preventable, which puts
preventable harm near 10% of admissions. That is the default target here.

Loading stays faithful to what the database holds: `load_safetyhops_cases`
still returns every encounter. Resampling is a separate, explicit step so the
number a reviewer sees is a stated analysis choice rather than something the
loader did on its own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from src.domain.models import PatientCase

# Roughly 25% of inpatients harmed, of which about 43% was preventable.
OIG_PREVENTABLE_HARM_RATE = 0.10


@dataclass
class ResampleSummary:
    """What was dropped to hit the target, so the change is reportable."""

    target_rate: float
    kept: int
    dropped: int
    original_rate: float
    resulting_rate: float

    def summary(self) -> str:
        return (
            f"resampled to {self.resulting_rate:.1%} harm "
            f"(from {self.original_rate:.1%}), kept {self.kept} of "
            f"{self.kept + self.dropped} cases"
        )


def harm_rate(cases: Sequence[PatientCase]) -> float:
    return sum(1 for c in cases if c.is_harm_event) / len(cases) if cases else 0.0


def downsample_to_prevalence(
    cases: Sequence[PatientCase],
    target_rate: float = OIG_PREVENTABLE_HARM_RATE,
    seed: int = 7,
) -> List[PatientCase]:
    """Drop cases from the over-represented class until harm sits at `target_rate`.

    Only ever drops, never duplicates. Oversampling the minority would invent
    encounters that the warehouse does not contain and would let a model see
    the same case in both train and test, which is the failure mode this
    project already went to some trouble to remove.

    The kept cases are chosen uniformly at random under `seed`, so scenario mix
    is preserved in expectation and the cohort is reproducible run to run.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be between 0 and 1, got {target_rate}")

    harm = [c for c in cases if c.is_harm_event]
    clean = [c for c in cases if not c.is_harm_event]
    if not harm or not clean:
        return list(cases)

    rng = random.Random(seed)
    if len(harm) / len(cases) > target_rate:
        # Too much harm: keep every clean case, thin the harm ones.
        keep_harm = int(round(len(clean) * target_rate / (1.0 - target_rate)))
        harm = rng.sample(harm, min(len(harm), max(1, keep_harm)))
    else:
        keep_clean = int(round(len(harm) * (1.0 - target_rate) / target_rate))
        clean = rng.sample(clean, min(len(clean), max(1, keep_clean)))

    kept = harm + clean
    rng.shuffle(kept)
    return kept


def resample_with_summary(
    cases: Sequence[PatientCase],
    target_rate: float = OIG_PREVENTABLE_HARM_RATE,
    seed: int = 7,
) -> tuple[List[PatientCase], ResampleSummary]:
    """Resample and report the change, for surfaces that show the cohort's shape."""
    kept = downsample_to_prevalence(cases, target_rate, seed)
    return kept, ResampleSummary(
        target_rate=target_rate,
        kept=len(kept),
        dropped=len(cases) - len(kept),
        original_rate=harm_rate(cases),
        resulting_rate=harm_rate(kept),
    )
