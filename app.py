import streamlit as st
import pandas as pd
from src.data.generator import generate_dataset
from src.engine.watcher import StructuredDataWatcher
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.model import LogisticScoringModel

st.set_page_config(page_title="SafetyNet", layout="wide")

@st.cache_resource
def load_data_and_train_model():
    # 1. Generate Data
    dataset = generate_dataset(num_cases=500, harm_ratio=0.15)
    
    # 2. Setup Engine
    watchers = [StructuredDataWatcher(), ClinicalNoteReader()]
    model = LogisticScoringModel(extractors=watchers)
    
    # 3. Train Model
    model.train(dataset)
    
    return dataset, watchers, model

def main():
    st.title("SafetyNet: Harm Event Triage")
    st.markdown("Continuously scanning patient records to flag likely-missed harm events for human review.")
    
    dataset, watchers, model = load_data_and_train_model()
    
    # Score all cases
    scored_cases = []
    for case in dataset:
        score = model.predict_score(case)
        evidence = model.get_evidence(case)
        scored_cases.append({
            "patient_id": case.patient_id,
            "age": case.age,
            "gender": case.gender,
            "score": score,
            "evidence": evidence,
            "is_harm": case.is_harm_event,
            "case_obj": case
        })
        
    # Sort by score descending
    scored_cases.sort(key=lambda x: x["score"], reverse=True)
    
    # Display
    st.subheader("High Priority Reviews")
    st.markdown("These cases have been flagged by the model for secondary human review. This is not a diagnostic tool.")
    
    # Only show top cases (score > 0.4 for demo)
    top_cases = [c for c in scored_cases if c["score"] > 0.4]
    
    if not top_cases:
        st.info("No high-priority cases currently flagged.")
    
    for case in top_cases:
        with st.expander(f"Patient {case['patient_id'][:8]}... | Alert Score: {case['score']:.2f} | Ground Truth: {case['is_harm']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Evidence Driving Score:**")
                evidence_items = [(f, i) for f, i in case["evidence"].items() if i > 0]
                if evidence_items:
                    for feature, impact in sorted(evidence_items, key=lambda x: x[1], reverse=True):
                        st.write(f"- 🔴 **{feature}**: +{impact:.2f} impact")
                else:
                    st.write("No specific strong red flags found.")
                    
            with col2:
                st.markdown("**Recent Events:**")
                for event in case["case_obj"].events[-3:]:
                    st.write(f"- **{event.event_type.upper()}**: {event.value}")
                    st.caption(f"{event.details}")

if __name__ == "__main__":
    main()
