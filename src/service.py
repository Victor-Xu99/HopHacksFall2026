"""JSON API for the React review dashboard.

Reuses the same watchers and scoring model as app.py. In-memory cache so SafetyHops
is not reloaded on every click.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.data.generator import generate_dataset
from src.data.mimic import load_mimic_cases, mimic_available
from src.data.safetyhops import hops_available, load_safetyhops_cases
from src.data.sql_store import available as sql_available
from src.data.sql_store import read_cases, source_counts
from src.domain.models import PatientCase
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher

SOURCE_SYNTHETIC = "synthetic"
SOURCE_MIMIC = "mimic"
SOURCE_SAFETYHOPS = "safetyhops"
SOURCE_SQL_MIMIC = "sql_mimic"
SOURCE_SQL_SYNTHETIC = "sql_synthetic"

LABEL_NAMES = {
    SOURCE_SYNTHETIC: "generator label",
    SOURCE_MIMIC: "coded complication",
    SOURCE_SAFETYHOPS: "condition-coded harm",
    SOURCE_SQL_MIMIC: "coded complication",
    SOURCE_SQL_SYNTHETIC: "generator label",
}


def list_sources() -> List[Dict[str, Any]]:
    sources = [
        {
            "id": SOURCE_SYNTHETIC,
            "label": "Synthetic (generated in memory)",
            "available": True,
        },
        {
            "id": SOURCE_SAFETYHOPS,
            "label": "SafetyHops (SQL Server)",
            "available": hops_available(),
        },
        {
            "id": SOURCE_MIMIC,
            "label": "MIMIC-IV demo (CSV files)",
            "available": mimic_available(),
        },
    ]
    if sql_available():
        counts = source_counts()
        sources.append(
            {
                "id": SOURCE_SQL_MIMIC,
                "label": "MIMIC-IV demo (SafetyNet SQL)",
                "available": bool(counts.get("mimic_iv_demo")),
            }
        )
        sources.append(
            {
                "id": SOURCE_SQL_SYNTHETIC,
                "label": "Synthetic (SafetyNet SQL)",
                "available": bool(counts.get("synthetic")),
            }
        )
    return sources


@lru_cache(maxsize=8)
def load_bundle(source: str) -> Tuple[Tuple[PatientCase, ...], StructuredDataWatcher, Optional[ClinicalNoteReader]]:
    if source == SOURCE_MIMIC:
        cases = tuple(load_mimic_cases())
        return cases, StructuredDataWatcher(anchor="admission"), None
    if source == SOURCE_SAFETYHOPS:
        cases = tuple(load_safetyhops_cases())
        return cases, StructuredDataWatcher(anchor="admission"), None
    if source == SOURCE_SQL_MIMIC:
        cases = tuple(read_cases("mimic_iv_demo"))
        has_notes = any(e.event_type == "note" for case in cases for e in case.events)
        reader = ClinicalNoteReader() if has_notes else None
        return cases, StructuredDataWatcher(anchor="admission"), reader
    if source == SOURCE_SQL_SYNTHETIC:
        cases = tuple(read_cases("synthetic"))
        has_notes = any(e.event_type == "note" for case in cases for e in case.events)
        reader = ClinicalNoteReader() if has_notes else None
        return cases, StructuredDataWatcher(), reader
    cases = tuple(generate_dataset(num_cases=400, harm_ratio=0.15, hard_negative_ratio=0.35, seed=7))
    return cases, StructuredDataWatcher(), ClinicalNoteReader()


@lru_cache(maxsize=16)
def trained_model(source: str, model_type: str) -> HarmScoringModel:
    cases, watcher, reader = load_bundle(source)
    extractors = [watcher] if reader is None else [watcher, reader]
    model = HarmScoringModel(list(extractors), model_type=model_type, threshold=0.4)
    model.train(list(cases))
    return model


def _serialize_event(event) -> Dict[str, str]:
    return {
        "event_type": event.event_type,
        "value": event.value,
        "details": event.details,
        "timestamp": event.timestamp,
    }


def _serialize_contribution(item) -> Dict[str, Any]:
    return {
        "feature": item.feature,
        "description": item.description,
        "value": item.value,
        "weight": item.weight,
        "contribution": item.contribution,
        "direction": item.direction,
    }


def review_payload(source: str, model_type: str, limit: int) -> Dict[str, Any]:
    cases, watcher, reader = load_bundle(source)
    model = trained_model(source, model_type)
    scored = sorted(
        (
            {
                "case_id": case.patient_id,
                "age": case.age,
                "gender": case.gender,
                "scenario": case.scenario,
                "score": model.predict_score(case),
                "label": case.is_harm_event,
                "event_count": len(case.events),
                "case": case,
            }
            for case in cases
        ),
        key=lambda row: row["score"],
        reverse=True,
    )

    flagged = scored[: max(1, min(limit, len(scored)))]
    tagged_on_list = sum(1 for row in flagged if row["label"])
    tagged_in_all = sum(1 for row in scored if row["label"])
    share_on_list = tagged_on_list / len(flagged) if flagged else 0.0
    share_overall = tagged_in_all / len(scored) if scored else 0.0

    reviews = []
    for row in flagged:
        case: PatientCase = row["case"]
        evidence = model.get_evidence(case)
        findings = []
        if reader is not None:
            for finding in reader.explain(case)[:6]:
                findings.append(
                    {
                        "text": finding.text,
                        "label": finding.label,
                        "probability": finding.probability,
                    }
                )
        reviews.append(
            {
                "case_id": row["case_id"],
                "age": row["age"],
                "gender": row["gender"],
                "scenario": row["scenario"],
                "score": float(row["score"]),
                "label": row["label"],
                "event_count": row["event_count"],
                "raising": [_serialize_contribution(c) for c in evidence if c.contribution > 0][:8],
                "lowering": [_serialize_contribution(c) for c in evidence if c.contribution < 0][:8],
                "triggers": watcher.explain(case)[:10],
                "notes": findings,
                "timeline": [_serialize_event(e) for e in case.sorted_events()[-12:]],
            }
        )

    report = model.report
    report_json = None
    if report is not None:
        report_json = {
            "n_train": report.n_train,
            "n_test": report.n_test,
            "roc_auc": report.roc_auc,
            "average_precision": report.average_precision,
            "pr_baseline": report.pr_baseline,
            "pr_lift": report.pr_lift,
            "recall": report.recall,
            "precision": report.precision,
            "cv_roc_auc_mean": report.cv_roc_auc_mean,
            "cv_roc_auc_std": report.cv_roc_auc_std,
            "cv_average_precision_mean": report.cv_average_precision_mean,
            "cv_average_precision_std": report.cv_average_precision_std,
            "brier": report.brier,
            "positive_rate": report.positive_rate,
            "threshold": report.threshold,
        }

    descriptions = watcher.feature_descriptions()
    fired = Counter()
    fired_and_labeled = Counter()
    for case in cases:
        for name, value in watcher.extract_features(case).items():
            if value > 0:
                fired[name] += 1
                if case.is_harm_event:
                    fired_and_labeled[name] += 1
    total = len(cases)
    base_rate = tagged_in_all / total if total else 0.0
    coverage = []
    for name, meaning in descriptions.items():
        count = fired.get(name, 0)
        rate = fired_and_labeled[name] / count if count else None
        lift = (rate / base_rate) if rate is not None and base_rate else None
        coverage.append(
            {
                "trigger": name,
                "meaning": meaning,
                "cases": count,
                "share": count / total if total else 0.0,
                "label_rate": rate,
                "lift": lift,
            }
        )
    coverage.sort(key=lambda r: (r["lift"] is None, -(r["lift"] or 0)))

    weights = model.feature_table()

    return {
        "source": source,
        "label_name": LABEL_NAMES.get(source, "harm tag"),
        "has_notes": reader is not None,
        "checked": len(scored),
        "top_reviews": len(flagged),
        "already_tagged_share": share_on_list,
        "overall_tagged_share": share_overall,
        "tagged_found": tagged_on_list,
        "tagged_total": tagged_in_all,
        "reviews": reviews,
        "report": report_json,
        "coverage": coverage,
        "weights": weights,
        "model_type": model_type,
    }
