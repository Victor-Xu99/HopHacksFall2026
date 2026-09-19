"""Synthetic patient cases for the demo.

Three populations, and the second one is what makes the exercise honest:

  harm           - a real adverse event, documented as unanticipated
  hard negative  - the same structured red flags (low hemoglobin, ICU transfer,
                   readmission, transfusion) but documented as a known,
                   consented, or planned part of care
  routine        - unremarkable admissions

Without the hard negatives, any keyword matcher scores a perfect AUC and the
learned weights mean nothing. With them, the model has to learn that an antidote
matters a lot, that an ICU transfer only matters when it was unplanned, and that
expected-risk phrasing should pull a score *down*.

Note wording here is paraphrased away from the classifier's training corpus on
purpose, so the note reader is generalizing rather than recalling.
"""

import random
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from faker import Faker

from src.domain.models import PatientCase, PatientEvent

fake = Faker()

HARM_SCENARIOS = (
    "opioid_oversedation",
    "postop_hemorrhage",
    "contrast_aki",
    "anticoagulant_dosing_error",
    "missed_diagnosis_readmission",
    "hospital_acquired_infection",
    "insulin_hypoglycemia",
    "procedural_injury",
    # Real harm that nobody wrote up as harm. This is the case hospitals actually
    # miss, and the only way to catch it is the structured trigger.
    "undocumented_oversedation",
    "undocumented_reoperation",
    # Harm with neither a structured trigger nor revealing documentation. No
    # system catches these, and leaving them in keeps the metrics honest.
    "silent_delayed_escalation",
)

HARD_NEGATIVE_SCENARIOS = (
    "expected_postop_anemia",
    "planned_icu_admission",
    "chemotherapy_cytopenia",
    "consented_known_complication",
    "planned_staged_readmission",
    "disease_progression",
    "negated_workup",
    # Surprise wording with no patient harm behind it, and an error caught before
    # it reached the patient. Both should look tempting and score low.
    "incidental_finding",
    "intercepted_near_miss",
    # The classic trigger-tool false positives: an antidote used routinely, and an
    # antidote for a problem the patient arrived with. Both fire `antidote_given`.
    "routine_reversal_agent",
    "community_overdose_on_arrival",
)


def _event_factory(base_time: datetime) -> Callable[..., PatientEvent]:
    def make(event_type: str, hours: float, value: str, details: str) -> PatientEvent:
        return PatientEvent(
            event_id=fake.uuid4(),
            event_type=event_type,
            timestamp=(base_time + timedelta(hours=hours)).isoformat(),
            value=value,
            details=details,
        )

    return make


def _baseline_events(make: Callable[..., PatientEvent]) -> List[PatientEvent]:
    return [
        make("admission", 0, "Inpatient admission", "Admitted for scheduled operative management."),
        make("lab", 0.5, "Hemoglobin", f"{random.uniform(12.0, 14.5):.1f}"),
        make("lab", 0.5, "Creatinine", f"{random.uniform(0.7, 1.1):.2f}"),
        make("lab", 0.5, "WBC", f"{random.uniform(5.0, 9.5):.1f}"),
        make("procedure", 2, "Index procedure", "Elective procedure completed without intraoperative difficulty."),
        make("medication", 3, "Acetaminophen", "Scheduled analgesia per standing order."),
    ]


def _routine_tail(make: Callable[..., PatientEvent]) -> List[PatientEvent]:
    return [
        make("lab", 20, "Hemoglobin", f"{random.uniform(11.0, 13.0):.1f}"),
        make("lab", 20, "Creatinine", f"{random.uniform(0.7, 1.2):.2f}"),
        make(
            "note",
            26,
            "Progress Note",
            "Comfortable this morning and up to the chair with nursing. "
            "Incision looks clean. Laboratory values reviewed and stable.",
        ),
        make(
            "note",
            46,
            "Discharge Summary",
            "Course was uncomplicated. Tolerating a regular diet and pain is managed with oral medication. "
            "Follow-up arranged in clinic.",
        ),
    ]


def _harm_events(scenario: str, make: Callable[..., PatientEvent]) -> List[PatientEvent]:
    if scenario == "opioid_oversedation":
        return [
            make("medication", 8, "Hydromorphone PCA", "Patient-controlled analgesia initiated."),
            make("transfer", 11, "ICU", "Unplanned transfer after airway compromise on the floor."),
            make("medication", 10.5, "Naloxone", "Administered for unresponsiveness with a respiratory rate of six."),
            make(
                "note",
                12,
                "Rapid Response Note",
                "Rapid response called for somnolence and shallow breathing. Reversal agent required, "
                "which is not something we anticipated on this analgesic regimen. "
                "The degree of sedation went well past what was intended.",
            ),
        ]

    if scenario == "postop_hemorrhage":
        return [
            make("lab", 9, "Hemoglobin", f"{random.uniform(6.6, 7.8):.1f}"),
            make("medication", 10, "PRBC transfusion", "Two units packed red blood cells for ongoing blood loss."),
            make("procedure", 12, "Unplanned return to operating room", "Emergent takeback for hematoma evacuation and control of bleeding."),
            make(
                "note",
                13,
                "Operative Note Addendum",
                "Blood loss was far greater than this operation typically involves. "
                "Re-exploration was needed to address a bleeding point that we did not expect to encounter.",
            ),
        ]

    if scenario == "contrast_aki":
        return [
            make("procedure", 6, "CT angiography", "Contrast study obtained for abdominal pain."),
            make("lab", 30, "Creatinine", f"{random.uniform(2.3, 3.4):.2f}"),
            make("lab", 30, "Potassium", f"{random.uniform(5.6, 6.4):.1f}"),
            make(
                "note",
                32,
                "Nephrology Consult",
                "Kidney function fell off sharply after the contrast load. "
                "Looking back, the study could reasonably have been held until hydration was optimized. "
                "This decline is out of keeping with her baseline trajectory.",
            ),
        ]

    if scenario == "anticoagulant_dosing_error":
        return [
            make("medication", 7, "Warfarin", "Received two separate orders for the same evening dose."),
            make("lab", 26, "INR", f"{random.uniform(4.8, 7.2):.1f}"),
            make("medication", 27, "Phytonadione", "Vitamin K given for supratherapeutic anticoagulation."),
            make(
                "note",
                28,
                "Pharmacy Intervention Note",
                "Duplicate anticoagulant administration identified on reconciliation. "
                "The patient received twice the intended amount because of an order entry problem. "
                "Reversal was necessary.",
            ),
        ]

    if scenario == "missed_diagnosis_readmission":
        return [
            make("note", 40, "Discharge Summary", "Discharged home with instructions to return for any new symptoms."),
            make("admission", 200, "Readmission within 30 days", "Unplanned readmission eleven days after discharge."),
            make("lab", 201, "WBC", f"{random.uniform(17.5, 23.0):.1f}"),
            make("lab", 201, "Lactate", f"{random.uniform(4.2, 6.5):.1f}"),
            make(
                "note",
                202,
                "Admission History",
                "Presents with an intra-abdominal collection that was almost certainly developing at the time of discharge. "
                "The abnormal white count on the day of discharge was not acted upon. "
                "Earlier imaging would likely have caught this.",
            ),
        ]

    if scenario == "hospital_acquired_infection":
        return [
            make("procedure", 4, "Central venous catheter placement", "Internal jugular line placed for access."),
            make("lab", 60, "WBC", f"{random.uniform(18.0, 24.0):.1f}"),
            make("lab", 60, "Lactate", f"{random.uniform(3.8, 5.5):.1f}"),
            make("transfer", 62, "ICU", "Unplanned transfer for septic shock requiring vasopressors."),
            make(
                "note",
                63,
                "Critical Care Note",
                "Bloodstream infection attributable to the indwelling line. "
                "Septic deterioration of this severity was not part of the anticipated hospital course. "
                "Line care documentation is incomplete for several shifts.",
            ),
        ]

    if scenario == "insulin_hypoglycemia":
        return [
            make("medication", 8, "Insulin glargine", "Long-acting insulin administered while patient remained NPO."),
            make("lab", 14, "Glucose", f"{random.uniform(30.0, 48.0):.0f}"),
            make("medication", 14.5, "Dextrose 50", "Given emergently for profound hypoglycemia."),
            make(
                "note",
                15,
                "Nursing Event Note",
                "Found diaphoretic and difficult to arouse with a critically low glucose. "
                "The basal insulin should have been held while she was fasting. "
                "Rescue dextrose was required.",
            ),
        ]

    if scenario == "procedural_injury":
        return [
            make("procedure", 5, "Thoracentesis", "Ultrasound-guided drainage of pleural effusion."),
            make("procedure", 8, "Unplanned return to operating room", "Unscheduled chest tube placement in the OR for iatrogenic pneumothorax."),
            make("transfer", 9, "ICU", "Unplanned transfer for hypoxemia after the procedure."),
            make(
                "note",
                10,
                "Procedure Complication Note",
                "Lung was entered during needle placement, producing a pneumothorax. "
                "This injury was unintended and was recognized only on the post-procedure film. "
                "Oxygenation worsened before the tube was placed.",
            ),
        ]

    if scenario == "undocumented_oversedation":
        # Harm occurred, documentation is silent about it.
        return [
            make("medication", 8, "Hydromorphone", "Intravenous opioid given for breakthrough pain."),
            make("medication", 11, "Naloxone", "Administered once."),
            make("lab", 12, "Potassium", f"{random.uniform(3.6, 4.4):.1f}"),
            make(
                "note",
                14,
                "Progress Note",
                "Mental status back to baseline. Vital signs within normal limits. Continue current plan and mobilize today.",
            ),
        ]

    if scenario == "undocumented_reoperation":
        return [
            make("lab", 10, "Hemoglobin", f"{random.uniform(6.9, 8.0):.1f}"),
            make("procedure", 12, "Unplanned return to operating room", "Returned to theatre for wound exploration."),
            make(
                "note",
                16,
                "Progress Note",
                "Postoperative day one following the second procedure. Dressing dry. Plan for discharge when mobilizing independently.",
            ),
        ]

    if scenario == "silent_delayed_escalation":
        return [
            make("lab", 26, "WBC", f"{random.uniform(12.5, 14.5):.1f}"),
            make(
                "note",
                30,
                "Progress Note",
                "Slightly warm this morning. Encouraged incentive spirometry. Will continue to observe.",
            ),
        ]

    return []


def _hard_negative_events(scenario: str, make: Callable[..., PatientEvent]) -> List[PatientEvent]:
    if scenario == "expected_postop_anemia":
        return [
            make("lab", 12, "Hemoglobin", f"{random.uniform(7.9, 8.8):.1f}"),
            make("medication", 13, "PRBC transfusion", "One unit given per the anemia protocol."),
            make(
                "note",
                14,
                "Progress Note",
                "Hemoglobin sits where we would predict after an operation of this size, and she was counseled "
                "beforehand that a transfusion might be needed. Vital signs are stable and there is no sign of active bleeding.",
            ),
        ]

    if scenario == "planned_icu_admission":
        return [
            make("transfer", 4, "ICU", "Planned postoperative admission arranged before the case began."),
            make("lab", 10, "Lactate", f"{random.uniform(2.6, 3.4):.1f}"),
            make(
                "note",
                11,
                "Critical Care Note",
                "Overnight critical care monitoring was booked in advance for this complex resection, as planned "
                "preoperatively. The mildly raised lactate is the anticipated consequence of a case of this length "
                "and is already trending down.",
            ),
        ]

    if scenario == "chemotherapy_cytopenia":
        return [
            make("medication", 4, "Carboplatin", "Cycle three administered on schedule."),
            make("lab", 72, "Platelets", f"{random.uniform(42.0, 88.0):.0f}"),
            make("lab", 72, "WBC", f"{random.uniform(1.2, 2.8):.1f}"),
            make(
                "note",
                74,
                "Oncology Progress Note",
                "Counts have fallen in the pattern we see at this point in the cycle, an anticipated effect of the "
                "regimen that was reviewed with him before treatment started. No fever and no bleeding. "
                "Supportive care continues per protocol.",
            ),
        ]

    if scenario == "consented_known_complication":
        return [
            make("lab", 20, "Bilirubin", f"{random.uniform(2.4, 4.2):.1f}"),
            make("lab", 20, "ALT", f"{random.uniform(150.0, 320.0):.0f}"),
            make(
                "note",
                22,
                "Surgical Progress Note",
                "A transient rise in liver enzymes sits inside the published complication rate for this reconstruction "
                "and was among the risks reviewed at consent. Technique and prophylaxis were appropriate throughout. "
                "Values are anticipated to normalize.",
            ),
        ]

    if scenario == "planned_staged_readmission":
        return [
            make("note", 40, "Discharge Summary", "Discharged home ahead of the second stage of the reconstruction."),
            make("admission", 220, "Readmission within 30 days", "Scheduled readmission for the planned second stage."),
            make("lab", 221, "Hemoglobin", f"{random.uniform(9.6, 11.0):.1f}"),
            make(
                "note",
                222,
                "Admission Note",
                "Returns as arranged for the staged operation that was described to the family before the first procedure. "
                "He has done well at home in the interval.",
            ),
        ]

    if scenario == "disease_progression":
        return [
            make("lab", 30, "Creatinine", f"{random.uniform(1.6, 2.2):.2f}"),
            make("transfer", 34, "ICU", "Planned transfer for closer monitoring of end-stage heart failure."),
            make(
                "note",
                36,
                "Palliative Care Note",
                "Worsening breathlessness and renal function reflect the natural course of her advanced cardiomyopathy "
                "rather than anything that happened here. Goals of care were revisited with the family, who understood "
                "this trajectory was likely.",
            ),
        ]

    if scenario == "negated_workup":
        return [
            make("lab", 18, "Hemoglobin", f"{random.uniform(9.2, 10.4):.1f}"),
            make(
                "note",
                19,
                "Progress Note",
                "No evidence of active hemorrhage on repeat imaging and an intra-abdominal collection was ruled out. "
                "Denies chest pain or shortness of breath. No acute distress and no deterioration overnight.",
            ),
        ]

    if scenario == "incidental_finding":
        return [
            make("procedure", 6, "CT abdomen", "Imaging obtained to confirm drain position."),
            make(
                "note",
                8,
                "Radiology Correlation Note",
                "An unexpected finding of a small renal cyst was noted on the scan. This was not the reason for imaging "
                "and carries no clinical significance. Outpatient follow-up imaging in one year is sufficient.",
            ),
        ]

    if scenario == "intercepted_near_miss":
        return [
            make("lab", 16, "Potassium", f"{random.uniform(5.6, 6.0):.1f}"),
            make(
                "note",
                18,
                "Pharmacy Note",
                "An order for the wrong dose was entered but pharmacy intercepted it during verification and it never "
                "reached the patient. The correct dose was substituted. No change in the patient's condition.",
            ),
        ]

    if scenario == "routine_reversal_agent":
        return [
            make("medication", 2.5, "Sugammadex", "Given at the conclusion of the case per routine practice."),
            make("lab", 12, "Hemoglobin", f"{random.uniform(10.2, 11.4):.1f}"),
            make(
                "note",
                13,
                "Anesthesia Note",
                "Neuromuscular blockade reversed at the end of the case as we do routinely. Extubated in theatre "
                "without difficulty and recovery was smooth.",
            ),
        ]

    if scenario == "community_overdose_on_arrival":
        return [
            make("medication", 0.25, "Naloxone", "Given in the emergency department on arrival."),
            make("lab", 1, "Lactate", f"{random.uniform(2.6, 3.6):.1f}"),
            make(
                "note",
                1.5,
                "Emergency Department Note",
                "Brought in after an out-of-hospital ingestion and reversed on arrival, before any treatment here began. "
                "The event preceded this admission entirely. Observed overnight and back to baseline.",
            ),
        ]

    return []


def generate_case(
    is_harm: bool,
    hard_negative: bool = False,
    scenario: Optional[str] = None,
) -> PatientCase:
    base_time = datetime.now() - timedelta(days=random.randint(1, 30))
    make = _event_factory(base_time)
    events = _baseline_events(make)

    if scenario is None:
        if is_harm:
            scenario = random.choice(HARM_SCENARIOS)
        elif hard_negative:
            scenario = random.choice(HARD_NEGATIVE_SCENARIOS)
        else:
            scenario = "routine"

    # An explicit scenario picks the events; `is_harm` remains the caller's to set,
    # so a scenario can be paired with either label when probing the model.
    if scenario in HARM_SCENARIOS:
        events.extend(_harm_events(scenario, make))
    elif scenario in HARD_NEGATIVE_SCENARIOS:
        events.extend(_hard_negative_events(scenario, make))
    elif scenario == "routine":
        events.extend(_routine_tail(make))
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    return PatientCase(
        patient_id=fake.uuid4(),
        age=random.randint(18, 90),
        gender=random.choice(["M", "F"]),
        events=events,
        is_harm_event=is_harm,
        scenario=scenario,
    )


def generate_dataset(
    num_cases: int,
    harm_ratio: float = 0.2,
    hard_negative_ratio: float = 0.35,
    seed: Optional[int] = None,
) -> List[PatientCase]:
    """`hard_negative_ratio` is the share of the *non-harm* cases that carry red flags."""
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    cases: List[PatientCase] = []
    for _ in range(num_cases):
        if random.random() < harm_ratio:
            cases.append(generate_case(is_harm=True))
        else:
            cases.append(
                generate_case(is_harm=False, hard_negative=random.random() < hard_negative_ratio)
            )
    return cases


def generate_labeled_sample(num_cases: int = 120, **kwargs) -> List[PatientCase]:
    """Stand-in for a reviewer-adjudicated training sample."""
    cases = generate_dataset(num_cases, **kwargs)
    for case in cases:
        case.label_source = "hand_labeled"
    return cases
