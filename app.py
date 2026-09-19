import pandas as pd
import streamlit as st

from src.data.generator import generate_dataset
from src.domain.models import PatientCase, PatientEvent
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher

st.set_page_config(page_title="SafetyNet", layout="wide")


@st.cache_resource(show_spinner="Generating cohort and fitting the scoring model...")
def build_engine(
    num_cases: int,
    harm_ratio: float,
    hard_negative_ratio: float,
    model_type: str,
    threshold: float,
    seed: int,
):
    dataset = generate_dataset(
        num_cases=num_cases,
        harm_ratio=harm_ratio,
        hard_negative_ratio=hard_negative_ratio,
        seed=seed,
    )
    watcher = StructuredDataWatcher()
    reader = ClinicalNoteReader()
    model = HarmScoringModel(
        extractors=[watcher, reader], model_type=model_type, threshold=threshold
    )
    model.train(dataset)
    return dataset, watcher, reader, model


def score_cohort(dataset, model) -> pd.DataFrame:
    rows = []
    for case in dataset:
        rows.append(
            {
                "patient_id": case.patient_id,
                "age": case.age,
                "gender": case.gender,
                "score": model.predict_score(case),
                "is_harm": case.is_harm_event,
                "scenario": case.scenario,
                "case_obj": case,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def render_evidence(case: PatientCase, watcher, reader, model) -> None:
    contributions = model.get_evidence(case)
    raising = [c for c in contributions if c.contribution > 0]
    lowering = [c for c in contributions if c.contribution < 0]

    left, right = st.columns(2)

    with left:
        st.markdown("**Signals raising the score**")
        if raising:
            for c in raising:
                st.write(f"- {c.description} — `+{c.contribution:.2f}` (value {c.value:g})")
        else:
            st.write("None.")

        st.markdown("**Signals lowering the score**")
        if lowering:
            for c in lowering:
                st.write(f"- {c.description} — `{c.contribution:.2f}` (value {c.value:g})")
        else:
            st.write("None.")

        breakdown = model.score_breakdown(case)
        if model.model_type == "logistic":
            st.caption(
                f"Baseline {breakdown['baseline_log_odds']:.2f} + contributions "
                f"{breakdown['sum_of_contributions']:+.2f} = logit {breakdown['logit']:.2f}, "
                f"which is a probability of {breakdown['probability']:.2f}."
            )
        else:
            st.caption(
                "Contributions are the change in predicted risk at each split along this case's "
                f"path through the tree, ending at {breakdown['probability']:.2f}."
            )

        structured = watcher.explain(case)
        if structured:
            st.markdown("**Structured triggers**")
            for note in structured:
                st.write(f"- {note}")

    with right:
        findings = reader.explain(case)
        st.markdown("**Note language**")
        if findings:
            badge = {
                "unanticipated": "not supposed to happen",
                "expected_risk": "known / consented risk",
                "neutral": "neutral",
            }
            for f in findings[:6]:
                tag = badge.get(f.label, f.label)
                st.write(f"- _{f.text}_")
                extras = []
                if f.error_terms:
                    extras.append("process failure wording: " + ", ".join(f.error_terms))
                if f.harm_terms:
                    extras.append("deterioration wording: " + ", ".join(f.harm_terms))
                if f.negated:
                    extras.append("negated")
                suffix = f" · {'; '.join(extras)}" if extras else ""
                st.caption(f"{tag} ({f.probability:.0%} confidence){suffix}")
        else:
            st.write("No harm-adjacent language detected.")

        st.markdown("**Timeline**")
        for event in case.sorted_events():
            st.caption(f"{event.event_type.upper()} · {event.value} — {event.details}")


def render_review_queue(scored: pd.DataFrame, watcher, reader, model, threshold: float) -> None:
    flagged = scored[scored["score"] >= threshold]

    a, b, c, d = st.columns(4)
    a.metric("Cases scanned", len(scored))
    b.metric("Flagged for review", len(flagged))
    if len(flagged):
        b.caption(f"{len(flagged) / len(scored):.0%} of the cohort")
    hit_rate = flagged["is_harm"].mean() if len(flagged) else 0.0
    c.metric("Flag hit rate", f"{hit_rate:.0%}")
    caught = flagged["is_harm"].sum()
    total_harm = scored["is_harm"].sum()
    d.metric("Harm cases caught", f"{caught}/{total_harm}")

    st.caption(
        "Hit rate and caught counts use the synthetic ground truth. In deployment these come "
        "from reviewer adjudication of the queue, which is also what retrains the model."
    )

    if not len(flagged):
        st.info("Nothing above the review threshold. Lower it in the sidebar to see more cases.")
        return

    for _, row in flagged.iterrows():
        header = (
            f"Patient {row['patient_id'][:8]} · age {row['age']}{row['gender']} · "
            f"score {row['score']:.2f} · ground truth harm={row['is_harm']}"
        )
        with st.expander(header):
            render_evidence(row["case_obj"], watcher, reader, model)


def render_model_tab(model, scored: pd.DataFrame) -> None:
    report = model.report
    if report is None:
        st.warning("Model has not been trained.")
        return

    st.markdown(
        f"Weights are fit on **{report.n_train}** labeled cases and all numbers below come from "
        f"**{report.n_test}** held-out cases the model never saw."
    )

    a, b, c, d = st.columns(4)
    a.metric("ROC AUC (held out)", f"{report.roc_auc:.3f}")
    b.metric("PR AUC (held out)", f"{report.average_precision:.3f}")
    c.metric(f"Recall @ {report.threshold:.2f}", f"{report.recall:.2f}")
    d.metric(f"Precision @ {report.threshold:.2f}", f"{report.precision:.2f}")

    e, f, g = st.columns(3)
    e.metric("Cross-validated ROC AUC", f"{report.cv_roc_auc_mean:.3f} ± {report.cv_roc_auc_std:.3f}")
    f.metric("Brier score", f"{report.brier:.3f}", help="Calibration error; lower is better.")
    g.metric("Harm prevalence", f"{report.positive_rate:.1%}")

    tn, fp, fn, tp = report.confusion
    st.markdown("**Held-out confusion matrix**")
    st.dataframe(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["actually no harm", "actually harm"],
            columns=["not flagged", "flagged"],
        ),
        use_container_width=False,
    )

    st.markdown("**What the model learned each signal is worth**")
    weight_table = pd.DataFrame(model.feature_table())
    st.dataframe(weight_table, use_container_width=True, hide_index=True)
    value_column = "log_odds_weight" if model.model_type == "logistic" else "split_importance"
    st.bar_chart(weight_table.set_index("feature")[value_column])

    if model.model_type == "logistic":
        st.caption(
            "Positive coefficients push a case up the queue, negative ones pull it down. "
            f"The intercept is {model.intercept:.2f}, the log-odds of harm when no signal fires."
        )
    else:
        st.markdown("**Decision rules**")
        st.code(model.describe_rules(), language="text")

    st.markdown("**Choosing the review threshold**")
    sweep = pd.DataFrame(report.threshold_sweep).set_index("threshold")
    st.line_chart(sweep[["precision", "recall", "f1"]])
    st.caption("Pick the threshold from the review capacity you actually have, not from a default.")

    st.markdown("**Score distribution by scenario**")
    by_scenario = (
        scored.groupby("scenario")
        .agg(cases=("score", "size"), mean_score=("score", "mean"), harm=("is_harm", "mean"))
        .sort_values("mean_score", ascending=False)
    )
    st.dataframe(by_scenario, use_container_width=True)
    st.caption(
        "Hard negatives such as planned_icu_admission and chemotherapy_cytopenia carry the same "
        "structured red flags as harm cases. Their scores show whether the model is reading framing "
        "or just pattern-matching on labs."
    )


def render_note_reader_tab(reader: ClinicalNoteReader) -> None:
    st.markdown(
        "The reader classifies each sentence as unanticipated, known-risk, or neutral. It is a "
        "TF-IDF n-gram logistic regression trained on a hand-labeled sentence corpus, paired with a "
        "negation-aware lexicon pass so a ruled-out finding does not read as a real one."
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
        "Postoperative bleeding required a return to theatre. The family had been told before surgery "
        "that this was a recognized risk of the operation."
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


def main() -> None:
    st.title("SafetyNet: Harm Event Triage")
    st.markdown(
        "Scans structured records and clinical notes for patterns that suggest a harm event was "
        "missed, then ranks cases for human review. Not a diagnostic tool."
    )

    with st.sidebar:
        st.header("Cohort")
        num_cases = st.slider("Cases", 100, 2000, 600, step=100)
        harm_ratio = st.slider("Harm prevalence", 0.05, 0.40, 0.15, step=0.05)
        hard_negative_ratio = st.slider(
            "Hard negatives among non-harm cases",
            0.0,
            0.8,
            0.35,
            step=0.05,
            help="Cases with real red flags that documentation shows were expected or planned.",
        )
        seed = st.number_input("Random seed", value=7, step=1)

        st.header("Scoring model")
        model_type = st.radio(
            "Family",
            options=["logistic", "tree"],
            format_func=lambda m: "Logistic regression" if m == "logistic" else "Decision tree",
        )
        threshold = st.slider("Review threshold", 0.05, 0.95, 0.40, step=0.05)

    dataset, watcher, reader, model = build_engine(
        num_cases=num_cases,
        harm_ratio=harm_ratio,
        hard_negative_ratio=hard_negative_ratio,
        model_type=model_type,
        threshold=threshold,
        seed=int(seed),
    )
    scored = score_cohort(dataset, model)

    queue_tab, model_tab, note_tab = st.tabs(
        ["Review queue", "Model performance", "Note reader"]
    )
    with queue_tab:
        render_review_queue(scored, watcher, reader, model, threshold)
    with model_tab:
        render_model_tab(model, scored)
    with note_tab:
        render_note_reader_tab(reader)


if __name__ == "__main__":
    main()
