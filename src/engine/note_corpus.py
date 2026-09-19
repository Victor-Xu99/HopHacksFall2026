"""Hand-labeled sentences used to train the note-language classifier.

Three classes, and the distinction between the first two is the whole point:

  unanticipated  - "this wasn't supposed to happen" framing
  expected_risk  - "this is a known, consented, anticipated risk" framing
  neutral        - routine documentation

The same clinical event can appear in both of the first two classes (a bleed, a
reintubation, an AKI). What separates them is how the clinician frames causation
and expectation, so the sentences below deliberately pair similar events with
opposite framing. Phrasing here is kept distinct from the synthetic generator's
templates so the model is not just memorizing the data generator.
"""

from typing import List, Tuple

UNANTICIPATED: List[str] = [
    "This was an unexpected complication that we did not anticipate given her stable preoperative status.",
    "The degree of bleeding was out of proportion to what this procedure normally involves.",
    "In retrospect the abnormal creatinine trend on day two should have been acted on sooner.",
    "Patient received a duplicate dose of anticoagulant due to an order entry issue.",
    "The decline was not consistent with the expected postoperative course.",
    "Unfortunately the drain was inadvertently dislodged during repositioning.",
    "There was a delay in recognizing the developing compartment syndrome.",
    "The hypoglycemic episode appears related to insulin dosing rather than her underlying illness.",
    "Failure to obtain the follow-up imaging contributed to the late diagnosis.",
    "This deterioration was not predicted by any of her preoperative risk factors.",
    "The wrong concentration of the infusion was hung and ran for approximately two hours.",
    "Oversedation requiring reversal was not an anticipated outcome of this regimen.",
    "An avoidable pressure injury developed during the prolonged ICU stay.",
    "The injury to the adjacent structure was unintended and recognized only postoperatively.",
    "Her respiratory arrest was sudden and without a clear precipitant we had considered.",
    "We had not expected the graft to fail this early in the postoperative period.",
    "The allergy was documented in the chart but the medication was given regardless.",
    "This adverse event was preventable with earlier escalation of care.",
    "Symptoms worsened despite treatment in a way that surprised the team.",
    "Retained surgical material was identified on the postoperative film.",
    "The line was placed in the wrong vessel and required removal and revision.",
    "Something clearly went wrong with the handoff between the two services.",
    "Acute kidney injury developed after a contrast study that probably should have been deferred.",
    "The fall occurred while the patient was ambulating without the ordered assistance.",
    "This was an iatrogenic pneumothorax following the central line attempt.",
    "Her course took an abrupt turn for the worse that none of us anticipated.",
    "The second operation was needed to address a problem created during the first.",
    "Sepsis developed from a source that should have been controlled at the index procedure.",
    "The dosing error was identified only after the patient became obtunded.",
    "He returned emergently with a complication that was not part of the expected recovery.",
]

EXPECTED_RISK: List[str] = [
    "Postoperative bleeding is a known risk of this procedure and was discussed during consent.",
    "The patient was counseled preoperatively that reoperation is sometimes necessary.",
    "Her anemia is within the range expected after a case of this magnitude.",
    "Mild renal dysfunction is an anticipated effect of the nephrotoxic regimen she requires.",
    "Nausea following anesthesia is an expected side effect and resolved with antiemetics.",
    "This complication, while unfortunate, falls within the published complication rate for the operation.",
    "Transfusion requirement was anticipated given her preoperative hemoglobin.",
    "The consent form documents infection as one of the known risks.",
    "Pain at the surgical site is typical for postoperative day one.",
    "Her prolonged intubation is consistent with the expected course for this degree of injury.",
    "The drop in platelets is an expected effect of the chemotherapy cycle.",
    "Transient confusion after this procedure is common in patients of her age.",
    "Escalation to the intensive care unit was planned in advance given the complexity of the resection.",
    "Progression of her underlying malignancy, not a complication of care, explains the decline.",
    "The wound drainage is within normal limits for this stage of healing.",
    "Risks including bleeding, infection, and the possible need for further surgery were reviewed with the family.",
    "Hypotension during induction is a recognized and anticipated response to the agents used.",
    "This is the natural history of the disease rather than an adverse event.",
    "Her readmission was scheduled as part of the planned staged reconstruction.",
    "The elevated white count reflects her known inflammatory condition.",
    "Deterioration is consistent with the expected trajectory of advanced heart failure.",
    "As anticipated in a patient on therapeutic anticoagulation, minor bruising is present.",
    "The team discussed that this outcome was possible despite optimal management.",
    "Postoperative ileus is expected after bowel manipulation and is resolving.",
    "This known complication occurred despite appropriate technique and prophylaxis.",
    "Her low blood pressure responded to fluids as expected during the dialysis session.",
    "The recurrence was anticipated given the margins reported at the initial resection.",
    "Fever in the first forty-eight hours is a common and expected postoperative finding.",
    "Per protocol she was monitored for this anticipated reaction and treated promptly.",
    "The family understood before surgery that a second stage would likely be required.",
]

# Half of these describe serious clinical events with no framing at all. They are
# the hardest and most important negatives: naming a rapid response or a
# transfusion is not, by itself, a claim that anything went wrong, and the
# classifier has to stay agnostic until framing appears.
NEUTRAL: List[str] = [
    "Rapid response was called for hypoxia and the patient was placed on high-flow oxygen.",
    "Transferred to the intensive care unit for closer monitoring.",
    "Hemoglobin was 7.2 and two units of packed cells were given.",
    "Returned to the operating room this afternoon.",
    "Creatinine is 2.4 today, up from 1.1 on admission.",
    "Naloxone was administered and mental status improved.",
    "The patient was intubated for airway protection.",
    "Blood cultures drawn and broad-spectrum antibiotics started.",
    "Readmitted eight days after the previous discharge.",
    "Glucose measured 44 and dextrose was given.",
    "Vital signs stable overnight, afebrile, tolerating diet.",
    "Dressing changed, incision clean and dry without erythema.",
    "Patient ambulating in the hallway with physical therapy twice today.",
    "Laboratory studies reviewed and within normal limits.",
    "Discussed discharge planning with case management.",
    "Continue current medications and follow up in clinic in two weeks.",
    "No acute distress, lungs clear to auscultation bilaterally.",
    "Foley removed and patient voiding spontaneously.",
    "Pain controlled on oral analgesics.",
    "Family updated at bedside, all questions answered.",
    "Plan is to repeat the metabolic panel in the morning.",
    "Physical exam unchanged from yesterday.",
    "Recovering well with steady progress toward baseline function.",
    "Home health services arranged prior to discharge.",
    "Denies chest pain, shortness of breath, or dizziness.",
    "Appetite improving and intake adequate without supplementation.",
    "Wound edges well approximated, staples intact.",
    "Telemetry reviewed, sinus rhythm without ectopy.",
    "Ready for transfer to the medical floor from the recovery unit.",
    "Reviewed imaging with radiology, no new findings.",
]


def labeled_sentences() -> Tuple[List[str], List[str]]:
    """Return (texts, labels) for the seed corpus."""
    texts = UNANTICIPATED + EXPECTED_RISK + NEUTRAL
    labels = (
        ["unanticipated"] * len(UNANTICIPATED)
        + ["expected_risk"] * len(EXPECTED_RISK)
        + ["neutral"] * len(NEUTRAL)
    )
    return texts, labels
