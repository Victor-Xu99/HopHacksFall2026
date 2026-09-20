"""Computes detection-rate (gain) curve data for the lift curve chart.

The curve answers: "If you review the top X% of cases by model score,
what fraction of all true harm events do you find?"

Two series are returned:
  model  - the curve for SafetyNet-ranked review
  random - the diagonal baseline (reviewing at random)

Both are normalised to [0, 1] so the chart is source-agnostic.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple


class LiftPoint(NamedTuple):
    review_share: float   # fraction of total cases reviewed (x-axis)
    harm_found: float     # fraction of total harm cases found  (y-axis)


def compute_lift_curve(
    scored: List[Tuple[float, bool]],
    n_points: int = 40,
) -> List[LiftPoint]:
    """Return the model gain curve.

    Args:
        scored: list of (score, is_harm) pairs for every case, unsorted.
        n_points: how many evenly-spaced x-axis steps to emit.

    Returns:
        List of LiftPoint from (0, 0) to (1, 1).
    """
    if not scored:
        return [LiftPoint(0.0, 0.0), LiftPoint(1.0, 1.0)]

    total = len(scored)
    total_harm = sum(1 for _, label in scored if label)
    if total_harm == 0:
        return [LiftPoint(x / n_points, 0.0) for x in range(n_points + 1)]

    # Sort highest score first — the model's recommended review order.
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)

    points: List[LiftPoint] = []
    harm_seen = 0
    for step in range(n_points + 1):
        cutoff = int(round(step / n_points * total))
        harm_seen = sum(1 for _, label in ranked[:cutoff] if label)
        points.append(
            LiftPoint(
                review_share=cutoff / total,
                harm_found=harm_seen / total_harm,
            )
        )

    return points


def random_baseline(n_points: int = 40) -> List[LiftPoint]:
    """The diagonal — what random review looks like."""
    return [LiftPoint(i / n_points, i / n_points) for i in range(n_points + 1)]


def lift_at(curve: List[LiftPoint], review_share: float) -> float:
    """Interpolate harm_found at a given review_share from the model curve."""
    for pt in curve:
        if pt.review_share >= review_share - 1e-9:
            return pt.harm_found
    return 1.0


def lift_multiple(curve: List[LiftPoint], review_share: float) -> float:
    """How many times more harm found vs random at the given review share.

    Returns 1.0 if the model is no better than random.
    """
    model_found = lift_at(curve, review_share)
    return model_found / review_share if review_share > 0 else 1.0
