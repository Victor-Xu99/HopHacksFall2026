import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src.data.generator import generate_dataset
from src.data.hospital_adapt import write_bundle
from src.data.hospital_extract import HOSPITAL_SOURCE, ingest_and_rank
from src.data.mimic import MIMIC_ROOT_DEFAULT, load_mimic_cases, mimic_available
from src.data.safetyhops import hops_available, load_safetyhops_cases
from src.data.sql_store import available as sql_available
from src.data.sql_store import read_cases, record_review, reviewed_case_ids, source_counts
from src.domain.models import PatientCase, PatientEvent
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher
from src.service import STALE_WAIT_DAYS, waiting_days

st.set_page_config(page_title="SafetyNet", layout="wide")
logger = logging.getLogger(__name__)

SYNTHETIC = "Synthetic cohort"
MIMIC = "MIMIC-IV demo (real records)"
SQL_MIMIC = "MIMIC-IV demo (SQL Server)"
SQL_SYNTHETIC = "Synthetic cohort (SQL Server)"
SAFETYHOPS = "SafetyHops (SQL Server)"
HOSPITAL = "Hospital extract (upload CSVs)"

# Sidebar label -> the `source` tag the cohort carries in core.cases. Everything
# read through here arrives as the same PatientCase objects as the CSV path, so
# nothing downstream of load_cases knows the difference.
SQL_SOURCES = {SQL_MIMIC: "mimic_iv_demo", SQL_SYNTHETIC: "synthetic"}
MIMIC_SOURCES = (MIMIC, SQL_MIMIC)

# Label wording differs by source and the difference matters, so it is never
# rendered as a bare "ground truth".
LABEL_NAMES = {
    SYNTHETIC: "generator label",
    MIMIC: "coded complication",
    SQL_MIMIC: "coded complication",
    SQL_SYNTHETIC: "generator label",
    SAFETYHOPS: "condition-coded harm",
    HOSPITAL: "not yet reviewed",
}


@st.cache_resource(show_spinner=False)
def sql_cohorts() -> Dict[str, int]:
    """Case count per selectable SQL cohort; empty when SQL Server is unreachable."""
    if not sql_available():
        return {}
    counts = source_counts()
    return {label: counts[tag] for label, tag in SQL_SOURCES.items() if counts.get(tag)}


@st.cache_resource(show_spinner="Loading cohort...")
def load_cases(source: str, num_cases: int, harm_ratio: float, hard_negative_ratio: float, seed: int):
    """Returns (cases, watcher, reader). The reader is None when there are no notes."""
    if source == MIMIC:
        cases = load_mimic_cases()
        # A whole admission is the unit here and coded procedures carry a date but
        # no time, so arrival is the honest baseline boundary.
        return cases, StructuredDataWatcher(anchor="admission"), None

    if source == HOSPITAL:
        cases = [case for case in read_cases(HOSPITAL_SOURCE) if case.discharged_at]
        return cases, StructuredDataWatcher(anchor="admission"), None

    if source == SAFETYHOPS:
        cases = load_safetyhops_cases()
        has_notes = any(e.event_type == "note" for case in cases for e in case.events)
        return (
            cases,
            StructuredDataWatcher(anchor="admission"),
            ClinicalNoteReader() if has_notes else None,
        )

    if source in SQL_SOURCES:
        cases = read_cases(SQL_SOURCES[source])
        anchor = "admission" if source in MIMIC_SOURCES else "procedure"
        # Whether a cohort carries notes is a property of the data, not of the
        # source name. Without any, the reader is dropped rather than asked for
        # features it would have to invent.
        has_notes = any(e.event_type == "note" for case in cases for e in case.events)
        return (
            cases,
            StructuredDataWatcher(anchor=anchor),
            ClinicalNoteReader() if has_notes else None,
        )

    cases = generate_dataset(
        num_cases=num_cases,
        harm_ratio=harm_ratio,
        hard_negative_ratio=hard_negative_ratio,
        seed=seed,
    )
    return cases, StructuredDataWatcher(), ClinicalNoteReader()


@st.cache_resource(show_spinner="Fitting the scoring model...")
def fit_model(_cases, _extractors, source: str, model_type: str, threshold: float, n: int):
    model = HarmScoringModel(_extractors, model_type=model_type, threshold=threshold)
    model.train(_cases)
    return model


@st.cache_resource(show_spinner="Scoring cohort...")
def score_cohort(_cases, _model, source: str, model_type: str, threshold: float) -> pd.DataFrame:
    rows = [
        {
            "case_id": case.patient_id,
            "age": case.age,
            "gender": case.gender,
            "score": _model.predict_score(case),
            "label": case.is_harm_event,
            "group": case.scenario,
            "events": len(case.events),
            "discharged_at": case.discharged_at or "9999-12-31",
            "waiting_days": waiting_days(case.discharged_at),
            "case_obj": case,
        }
        for case in _cases
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["score", "discharged_at"], ascending=[False, True])
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner="Measuring trigger coverage...")
def coverage_frame(_cases, _watcher, source: str) -> pd.DataFrame:
    """How often each trigger fires, and how it relates to the label."""
    descriptions = _watcher.feature_descriptions()
    fired = Counter()
    fired_and_labeled = Counter()
    for case in _cases:
        for name, value in _watcher.extract_features(case).items():
            if value > 0:
                fired[name] += 1
                if case.is_harm_event:
                    fired_and_labeled[name] += 1

    total = len(_cases)
    base_rate = sum(c.is_harm_event for c in _cases) / total if total else 0.0
    rows = []
    for name, meaning in descriptions.items():
        count = fired.get(name, 0)
        rate = fired_and_labeled[name] / count if count else float("nan")
        rows.append(
            {
                "trigger": name,
                "meaning": meaning,
                "cases": count,
                "% of cohort": count / total if total else 0.0,
                "label rate when fired": rate,
                "lift": rate / base_rate if count and base_rate else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("lift", ascending=False, na_position="last")


def render_timeline(case: PatientCase, limit: int = 12) -> None:
    events = case.sorted_events()
    by_type = Counter(e.event_type for e in events)
    st.caption(
        f"{len(events)} events: " + ", ".join(f"{n} {t}" for t, n in by_type.most_common())
    )
    if len(events) > limit:
        st.caption(f"Showing the last {limit}.")
    for event in events[-limit:]:
        st.caption(f"{event.event_type.upper()} | {event.value} - {event.details}")


def render_evidence(case: PatientCase, watcher, reader: Optional[ClinicalNoteReader], model) -> None:
    contributions = model.get_evidence(case)
    raising = [c for c in contributions if c.contribution > 0]
    lowering = [c for c in contributions if c.contribution < 0]

    left, right = st.columns(2)

    with left:
        st.markdown("**Signals raising the score**")
        if raising:
            for c in raising:
                st.write(f"- {c.description} - `+{c.contribution:.2f}` (value {c.value:g})")
        else:
            st.write("None.")

        st.markdown("**Signals lowering the score**")
        if lowering:
            for c in lowering:
                st.write(f"- {c.description} - `{c.contribution:.2f}` (value {c.value:g})")
        else:
            st.write("None.")

        breakdown = model.score_breakdown(case)
        if model.model_type == "logistic":
            st.caption(
                f"Baseline {breakdown['baseline_log_odds']:.2f} + contributions "
                f"{breakdown['sum_of_contributions']:+.2f} = logit {breakdown['logit']:.2f}, "
                f"a probability of {breakdown['probability']:.2f}."
            )
        else:
            st.caption(
                "Contributions are the change in predicted risk at each split along this "
                f"case's path through the tree, ending at {breakdown['probability']:.2f}."
            )

        structured = watcher.explain(case)
        if structured:
            st.markdown("**Structured triggers**")
            for note in structured[:10]:
                st.write(f"- {note}")
            if len(structured) > 10:
                st.caption(f"{len(structured) - 10} more not shown.")

    with right:
        st.markdown("**Note language**")
        if reader is None:
            st.info(
                "This dataset carries no clinical notes, so the note reader contributes "
                "nothing here and is excluded from the model."
            )
        else:
            findings = reader.explain(case)
            if findings:
                badge = {
                    "unanticipated": "not supposed to happen",
                    "expected_risk": "known / consented risk",
                    "neutral": "neutral",
                }
                for f in findings[:6]:
                    st.write(f"- _{f.text}_")
                    extras = []
                    if f.error_terms:
                        extras.append("process failure wording: " + ", ".join(f.error_terms))
                    if f.harm_terms:
                        extras.append("deterioration wording: " + ", ".join(f.harm_terms))
                    if f.negated:
                        extras.append("negated")
                    suffix = f" | {'; '.join(extras)}" if extras else ""
                    st.caption(f"{badge.get(f.label, f.label)} ({f.probability:.0%}){suffix}")
            else:
                st.write("No harm-adjacent language detected.")

        st.markdown("**Timeline**")
        render_timeline(case)


def render_review_queue(scored: pd.DataFrame, watcher, reader, model, source: str, queue) -> None:
    label_name = LABEL_NAMES[source]
    mode, capacity, threshold = queue

    if mode == "capacity":
        flagged = scored.head(capacity)
        caption = (
            f"Showing the {capacity} stays the tool scored highest. "
            "That is the list a reviewer would open this cycle."
        )
    else:
        flagged = scored[scored["score"] >= threshold]
        caption = (
            f"Showing every stay whose score is at least {threshold:.2f}."
        )

    if source == HOSPITAL and sql_available():
        done = set(reviewed_case_ids(HOSPITAL_SOURCE))
        flagged = flagged[~flagged["case_id"].isin(done)]

    tagged_on_list = int(flagged["label"].sum()) if len(flagged) else 0
    tagged_in_all = int(scored["label"].sum())
    share_on_list = flagged["label"].mean() if len(flagged) else 0.0
    share_overall = scored["label"].mean() if len(scored) else 0.0

    a, b, c, d = st.columns(4)
    a.metric(
        "Stays the tool checked",
        len(scored),
        help="Every patient stay in this dataset, scored from highest concern to lowest.",
    )
    b.metric(
        "Top Reviews",
        len(flagged),
        help="How many of those stays you asked to look at now. Highest scores go first.",
    )
    c.metric(
        "Already tagged as harm",
        f"{share_on_list:.0%}",
        help=(
            f"Of the stays on this review list, how many already have a harm tag in the data "
            f"({label_name}). About {share_overall:.0%} of all stays have that tag, "
            "so a random list of the same length would land near that lower number."
        ),
    )
    d.metric(
        "Tagged stays this list found",
        f"{tagged_on_list} of {tagged_in_all}",
        help=(
            f"There are {tagged_in_all} stays in the whole dataset that already have a harm tag. "
            f"This short list includes {tagged_on_list} of them."
        ),
    )
    st.caption(caption)
    st.caption(
        f"\"Already tagged as harm\" means the stay has a {label_name} in the dataset. "
        "It does not mean a doctor missed something. It means the tag was already there, "
        "and we are checking whether the short list is full of those tagged stays."
    )

    if not len(flagged):
        st.info("This review list is empty. Raise how many stays to review, or lower the score cutoff.")
        return

    for _, row in flagged.iterrows():
        header = (
            f"{row['case_id']} | age {row['age']}{row['gender']} | "
            f"score {row['score']:.2f} | harm tag: {'yes' if row['label'] else 'no'}"
        )
        wait = row.get("waiting_days")
        if pd.notna(wait) and wait is not None:
            header += f" | waiting {int(wait)}d"
            if int(wait) >= STALE_WAIT_DAYS:
                header += " STALE"
        with st.expander(header):
            cols = st.columns([4, 1])
            with cols[1]:
                if source == HOSPITAL and st.button("Done", key=f"done_{row['case_id']}"):
                    try:
                        record_review(
                            HOSPITAL_SOURCE,
                            str(row["case_id"]),
                            "queue",
                            "unclear",
                            "cleared from queue",
                        )
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Could not save review")
                        st.error(str(exc))
            render_evidence(row["case_obj"], watcher, reader, model)
            if source == HOSPITAL:
                v1, v2, v3 = st.columns(3)
                for label, decision, col in (
                    ("Harm", "harm", v1),
                    ("No harm", "no_harm", v2),
                    ("Unclear", "unclear", v3),
                ):
                    with col:
                        if st.button(label, key=f"{decision}_{row['case_id']}"):
                            try:
                                record_review(
                                    HOSPITAL_SOURCE,
                                    str(row["case_id"]),
                                    "queue",
                                    decision,
                                    None,
                                )
                                st.rerun()
                            except Exception as exc:
                                logger.exception("Could not save review")
                                st.error(str(exc))


def render_model_tab(model, scored: pd.DataFrame, source: str) -> None:
    report = model.report
    if report is None:
        st.warning("Model has not been trained.")
        return

    st.markdown(
        f"Weights are fit on **{report.n_train}** labeled cases. Every number below comes "
        f"from **{report.n_test}** held-out cases the model never saw."
    )
    if source in MIMIC_SOURCES:
        st.warning(
            "The label here is a proxy: ICD complication-of-care codes, the family behind the "
            "AHRQ Patient Safety Indicators. Administrative coding is known to under-capture "
            "harm, so absence of a code is weak evidence that nothing happened."
        )
    if source == SAFETYHOPS:
        st.warning(
            "The label here is a planted harm condition on the encounter (hemorrhage, "
            "oversedation, iatrogenic hypoglycemia, and similar). Chronic disease is not "
            "counted as harm. There are no free-text notes in this warehouse."
        )

    a, b, c, d = st.columns(4)
    a.metric("ROC AUC (held out)", f"{report.roc_auc:.3f}")
    b.metric(
        "PR AUC (held out)",
        f"{report.average_precision:.3f}",
        delta=f"{report.pr_lift:.2f}x baseline",
        help=(
            "Area under the precision-recall curve. Unlike ROC AUC, which always "
            f"nulls at 0.500, this one nulls at the {report.pr_baseline:.1%} label "
            "prevalence, so the multiple beside it is the part that carries meaning."
        ),
    )
    c.metric(f"Recall @ {report.threshold:.2f}", f"{report.recall:.2f}")
    d.metric(f"Precision @ {report.threshold:.2f}", f"{report.precision:.2f}")

    e, f, g, h = st.columns(4)
    e.metric(
        "Cross-validated ROC AUC", f"{report.cv_roc_auc_mean:.3f} +/- {report.cv_roc_auc_std:.3f}"
    )
    f.metric(
        "Cross-validated PR AUC",
        f"{report.cv_average_precision_mean:.3f} +/- {report.cv_average_precision_std:.3f}",
        help="The noisier of the two metrics when positives are scarce, so the spread matters.",
    )
    g.metric("Brier score", f"{report.brier:.3f}", help="Calibration error; lower is better.")
    h.metric(
        "Label prevalence",
        f"{report.positive_rate:.1%}",
        help="Also the PR AUC baseline: what a model that guessed at the base rate would score.",
    )

    st.caption(
        f"PR AUC is the metric to read here. This is a ranking problem under a fixed review "
        f"budget, and ROC AUC flatters a model when positives are scarce. At a "
        f"{report.pr_baseline:.1%} base rate, chance scores {report.pr_baseline:.3f} and this "
        f"model scores {report.average_precision:.3f}."
    )
    if report.positive_rate > 0.30:
        st.warning(
            f"This cohort is {report.positive_rate:.0%} positive, far denser than harm in a real "
            "hospital population. Both AUCs, and PR AUC especially, will read lower against a "
            "realistic base rate of a few percent."
        )
    if report.n_test < 150:
        st.caption(
            "With a held-out set this small, the single-split numbers are noisy. The "
            "cross-validated figures are the ones to trust."
        )

    tn, fp, fn, tp = report.confusion
    st.markdown("**Held-out confusion matrix**")
    st.dataframe(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["no label", "labeled"],
            columns=["not flagged", "flagged"],
        )
    )

    st.markdown("**Precision at review capacity**")
    capacity_rows = []
    for k in (10, 25, 50, 100, 200):
        if k > len(scored):
            break
        top = scored.head(k)["label"]
        capacity_rows.append(
            {
                "reviewing top": k,
                "precision": top.mean(),
                "caught": int(top.sum()),
                "of total": int(scored["label"].sum()),
            }
        )
    if capacity_rows:
        st.dataframe(pd.DataFrame(capacity_rows), hide_index=True)
        st.caption(
            f"Base rate is {scored['label'].mean():.1%}, so anything above that is lift over "
            "reviewing cases at random."
        )

    st.markdown("**What the model learned each signal is worth**")
    weight_table = pd.DataFrame(model.feature_table())
    value_column = "log_odds_weight" if model.model_type == "logistic" else "split_importance"
    st.dataframe(weight_table, width="stretch", hide_index=True)
    st.bar_chart(weight_table.set_index("feature")[value_column])

    if model.model_type == "logistic":
        st.caption(
            "Positive coefficients push a case up the queue, negative ones pull it down. The "
            f"intercept is {model.intercept:.2f}, the log-odds when no signal fires."
        )
    else:
        st.markdown("**Decision rules**")
        st.code(model.describe_rules(), language="text")

    st.markdown("**Choosing a threshold**")
    sweep = pd.DataFrame(report.threshold_sweep).set_index("threshold")
    st.line_chart(sweep[["precision", "recall", "f1"]])

    group_label = "admission type" if source in MIMIC_SOURCES else "scenario"
    st.markdown(f"**Score distribution by {group_label}**")
    by_group = (
        scored.groupby("group")
        .agg(cases=("score", "size"), mean_score=("score", "mean"), label_rate=("label", "mean"))
        .sort_values("mean_score", ascending=False)
    )
    st.dataframe(by_group, width="stretch")


def render_coverage_tab(coverage: pd.DataFrame, scored: pd.DataFrame, source: str) -> None:
    st.markdown(
        "Before trusting any score, check which triggers this dataset can actually support. "
        "A trigger that never fires is more important to know about than one that does."
    )

    dark = coverage[coverage["cases"] == 0]
    if len(dark):
        st.error(
            "Not derivable from this dataset: "
            + ", ".join(f"`{name}`" for name in dark["trigger"])
            + ". These contribute nothing and should not be read as reassurance."
        )

    st.dataframe(
        coverage.style.format(
            {"% of cohort": "{:.1%}", "label rate when fired": "{:.1%}", "lift": "{:.2f}x"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Lift is the label rate among cases where the trigger fired, divided by the "
        f"{scored['label'].mean():.1%} cohort base rate. Below 1.00x means the trigger is "
        "pointing the wrong way."
    )

    weak = coverage[(coverage["cases"] >= 5) & (coverage["lift"] < 1.0)]
    if len(weak):
        st.warning(
            "Firing but not informative here: "
            + ", ".join(f"`{name}`" for name in weak["trigger"])
        )

    st.bar_chart(coverage.set_index("trigger")["lift"])


def render_note_reader_tab(reader: Optional[ClinicalNoteReader], source: str) -> None:
    if reader is None:
        st.info(
            f"The active dataset has no clinical notes, so the note reader is not part of the "
            f"model right now. The MIMIC-IV demo excludes free text by design; notes ship as a "
            f"separate credentialed dataset. You can still exercise the reader below - it is "
            f"trained on its own labeled corpus, independent of the cohort."
        )
        reader = ClinicalNoteReader()

    st.markdown(
        "The reader classifies each sentence as unanticipated, known-risk, or neutral. It is a "
        "TF-IDF n-gram logistic regression trained on a hand-labeled sentence corpus, paired "
        "with a negation-aware pass so a ruled-out finding does not read as a real one."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Terms pushing toward 'not supposed to happen'**")
        st.dataframe(
            pd.DataFrame(reader.top_terms("unanticipated", 12), columns=["term", "weight"]),
            hide_index=True,
        )
    with right:
        st.markdown("**Terms pushing toward 'known, consented risk'**")
        st.dataframe(
            pd.DataFrame(reader.top_terms("expected_risk", 12), columns=["term", "weight"]),
            hide_index=True,
        )

    st.markdown("**Try it on your own text**")
    default = (
        "Postoperative bleeding required a return to theatre. The family had been told before "
        "surgery that this was a recognized risk of the operation."
    )
    text = st.text_area("Note text", value=default, height=110)
    if text.strip():
        probe = PatientCase(
            patient_id="probe",
            age=0,
            gender="U",
            events=[
                PatientEvent(
                    event_id="probe",
                    event_type="note",
                    timestamp="1970-01-01T00:00:00",
                    value="Ad hoc note",
                    details=text,
                )
            ],
        )
        for finding in reader.explain(probe):
            st.write(f"- _{finding.text}_")
            st.caption(f"{finding.label} ({finding.probability:.0%})")
        st.json({k: v for k, v in reader.extract_features(probe).items()})


def sidebar():
    with st.sidebar:
        st.header("Data source")
        have_mimic = mimic_available()
        have_hops = hops_available()
        cohorts = sql_cohorts()
        options = (
            [SYNTHETIC]
            + ([HOSPITAL] if sql_available() else [])
            + ([SAFETYHOPS] if have_hops else [])
            + ([MIMIC] if have_mimic else [])
            + list(cohorts)
        )
        source = st.radio("Cohort", options=options, key="source_radio")
        if not have_mimic:
            st.caption(
                f"Extract the MIMIC-IV demo to `{MIMIC_ROOT_DEFAULT}` to score real records."
            )
        if not have_hops:
            st.caption(
                "SafetyHops is the synthetic warehouse (SQL Server database `SafetyHops`). "
                "It is missing or unreachable on this instance."
            )
        if not cohorts:
            st.caption(
                "Run `python -m scripts.build_safetynet_db` to score canonical `core` cohorts "
                "out of the SafetyNet database."
            )

        num_cases, harm_ratio, hard_negative_ratio, seed = 600, 0.15, 0.35, 7
        if source == SYNTHETIC:
            num_cases = st.slider("Cases", 100, 2000, 600, step=100)
            harm_ratio = st.slider("Harm prevalence", 0.05, 0.40, 0.15, step=0.05)
            hard_negative_ratio = st.slider(
                "Hard negatives among non-harm cases",
                0.0,
                0.8,
                0.35,
                step=0.05,
                help="Cases with real red flags that documentation shows were expected.",
            )
            seed = int(st.number_input("Random seed", value=7, step=1))
        elif source == MIMIC:
            st.caption(
                "100 real de-identified patients, 275 admissions, labeled by ICD "
                "complication-of-care codes. No clinical notes in this release."
            )
        elif source == HOSPITAL:
            st.caption(
                "Upload one folder's CSVs (or a zip). SafetyNet guesses which file is stays, "
                "labs, meds, and transfers from names and column headers. Ranking starts "
                "only after a discharge time is present."
            )
            bundle = st.file_uploader(
                "Hospital folder or zip",
                type=["csv", "zip"],
                accept_multiple_files=True,
                key="hospital_bundle",
            )
            if st.button("Import folder", type="primary"):
                if not bundle:
                    st.error("Choose the CSVs from one folder, or one zip.")
                else:
                    try:
                        with st.spinner("Importing into SafetyNet SQL and ranking new discharges. This can take a minute."):
                            folder = Path(tempfile.mkdtemp(prefix="safetynet_extract_"))
                            unpacked = write_bundle(
                                folder, [(item.name, item.getvalue()) for item in bundle]
                            )
                            result = ingest_and_rank(unpacked)
                        st.session_state["hospital_ingest"] = result
                        st.session_state["hospital_ingest_error"] = None
                        st.cache_resource.clear()
                    except Exception as exc:
                        logger.exception("Hospital CSV import failed")
                        st.session_state["hospital_ingest_error"] = str(exc)
            last = st.session_state.get("hospital_ingest")
            err = st.session_state.get("hospital_ingest_error")
            if err:
                st.error(err)
            if last:
                st.success(
                    f"Imported {last['stays_upserted']} stays - {last['in_house']} still in house, "
                    f"{last['discharged']} discharged, {last['ranked']['scored']} newly ranked. "
                    "Discharged stays should appear in the review list."
                )
                if last.get("mapping"):
                    st.caption("Mapped files: " + str(last["mapping"].get("files", {})))
        elif source == SAFETYHOPS:
            st.caption(
                "Synthetic encounters in SQL Server database `SafetyHops`: labs, meds, "
                "procedures, and conditions. Harm labels come from planted harm conditions, "
                "not from the watcher features. No clinical notes."
            )
        else:
            st.caption(
                f"{cohorts[source]} cases read from `core.cases`, source tag "
                f"`{SQL_SOURCES[source]}`. Same objects the CSV path produces."
            )

        st.header("Scoring model")
        model_type = st.radio(
            "Family",
            options=["logistic", "tree"],
            format_func=lambda m: "Logistic regression" if m == "logistic" else "Decision tree",
            key="model_family",
        )

        st.header("Review list")
        mode_label = st.radio(
            "How to pick stays for review",
            ["A fixed number of stays", "A score cutoff"],
            key="queue_mode",
        )
        mode = "capacity" if mode_label == "A fixed number of stays" else "threshold"
        capacity = 20
        threshold = 0.40
        if mode == "capacity":
            capacity = int(
                st.number_input(
                    "How many stays to review",
                    min_value=5,
                    max_value=300,
                    value=20,
                    step=5,
                    help="A reviewer only has time for a short list. Highest scores go first.",
                )
            )
        else:
            threshold = st.slider("Minimum score to include", 0.05, 0.95, 0.40, step=0.05)

    return source, num_cases, harm_ratio, hard_negative_ratio, seed, model_type, (
        mode,
        capacity,
        threshold,
    )


def main() -> None:
    st.title("SafetyNet: Harm Event Triage")
    st.markdown(
        "Scans completed records for patterns suggesting a harm event went unreported, then "
        "ranks cases for human review. Retrospective quality assurance, not a diagnostic tool."
    )

    source, num_cases, harm_ratio, hard_negative_ratio, seed, model_type, queue = sidebar()
    last_import = st.session_state.get("hospital_ingest")
    if last_import and source == HOSPITAL:
        st.success(
            f"Last import stored {last_import['stays_upserted']} stays. "
            f"{last_import['discharged']} have a discharge time and can be ranked. "
            f"{last_import['in_house']} are still in house (not on the list yet)."
        )
    cases, watcher, reader = load_cases(
        source, num_cases, harm_ratio, hard_negative_ratio, seed
    )
    if source == HOSPITAL and not cases:
        st.info(
            "No discharged stays yet. Use **Hospital extract (upload CSVs)** in the sidebar "
            "and import stays.csv, labs.csv, meds.csv, and transfers.csv. "
            "Sample files are in `tests/fixtures/hospital/in_house`."
        )
        return
    if source == HOSPITAL:
        train_cases, train_watcher, _ = load_cases(SYNTHETIC, 200, 0.2, 0.3, 7)
        model = fit_model(
            train_cases, [train_watcher], "hospital_ranker", model_type, queue[2], len(train_cases)
        )
    else:
        extractors = [watcher] if reader is None else [watcher, reader]
        model = fit_model(cases, extractors, source, model_type, queue[2], len(cases))
    scored = score_cohort(cases, model, source, model_type, queue[2])
    coverage = coverage_frame(cases, watcher, source)

    if source == SAFETYHOPS:
        origin = "SQL Server database `SafetyHops`"
    elif source in SQL_SOURCES:
        origin = "the SafetyNet database"
    else:
        origin = "CSV"
    if source in MIMIC_SOURCES:
        st.success(
            f"Scoring **{len(cases)} real admissions** from the MIMIC-IV clinical database demo, "
            f"read from {origin}. Structured signals only, since this release carries no free text."
        )
    elif source == HOSPITAL:
        st.success(
            f"Scoring **{len(cases)} discharged hospital stays** stored in SafetyNet SQL. "
            "In-house stays are kept but not ranked until discharged is filled in."
        )
    elif source == SAFETYHOPS:
        st.success(
            f"Scoring **{len(cases)} synthetic encounters** from {origin}. "
            "Structured signals only; this warehouse has no free-text notes."
        )
    else:
        st.info(
            f"Scoring **{len(cases)} synthetic cases** from {origin}. Useful for exercising the "
            "note reader, which real records here cannot."
        )

    queue_tab, model_tab, coverage_tab, note_tab = st.tabs(
        ["Review list", "Model performance", "Data coverage", "Note reader"]
    )
    with queue_tab:
        render_review_queue(scored, watcher, reader, model, source, queue)
    with model_tab:
        render_model_tab(model, scored, source)
    with coverage_tab:
        render_coverage_tab(coverage, scored, source)
    with note_tab:
        render_note_reader_tab(reader, source)


if __name__ == "__main__":
    main()
