"""Minimal-contrast sentence pairs: same clinical event, opposite framing.

Every pair names one clinical event twice and changes only whether the note
presents it as something that went wrong or something that was foreseen. A
reader that has genuinely learned framing flips its answer across the pair. A
reader that has memorized which clinical events tend to be harm cannot.

These sentences are held out from everything. They are not in the training
corpus of `note_corpus`, and they are not drawn from the synthetic generator's
templates, so both the trained classifier and the keyword lexicon meet them for
the first time here.

Roughly half the pairs deliberately avoid the obvious cue vocabulary
("anticipated", "consented", "error") and carry their framing through ordinary
idiom instead -- "par for the course", "had no business", "went in with no tray
in front of her". That is on purpose. A keyword list should be expected to
struggle with those, and if it does not struggle at all then the cue list was
written too close to the test.
"""

from typing import List, Tuple

# (clinical event, unanticipated framing, expected-risk framing)
CONTRAST_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    (
        "pneumothorax after central line",
        "The lung was dropped during the subclavian attempt.",
        "Pneumothorax is a recognized hazard of subclavian access and was on the consent.",
    ),
    (
        "postoperative fever",
        "Her fever on day five traces back to the collection we failed to drain.",
        "Low-grade fever on day one is par for the course after a case like this.",
    ),
    (
        "acute kidney injury",
        "Creatinine climbed because she got contrast she never should have received.",
        "A bump in creatinine is part and parcel of the regimen she is on.",
    ),
    (
        "reintubation",
        "She had to go back on the ventilator after being extubated far too early.",
        "Reintubation was factored into the plan given her baseline lung disease.",
    ),
    (
        "surgical bleeding",
        "The ooze from the site was well beyond anything this operation should produce.",
        "Some oozing sits comfortably inside the usual range for this repair.",
    ),
    (
        "hyponatremia",
        "Sodium fell because the maintenance fluid order was never adjusted.",
        "A mild sodium drift is routine while she is on this diuretic.",
    ),
    (
        "inpatient fall",
        "He went down in the bathroom after the bed alarm was left switched off.",
        "Fall risk was reviewed with the family given his advanced dementia.",
    ),
    (
        "wound dehiscence",
        "The closure gave way, which has no business happening at day three.",
        "Given her steroid dependence, wound breakdown was flagged as likely from the outset.",
    ),
    (
        "ventricular arrhythmia",
        "She threw a run of VT that caught the whole team off guard.",
        "Transient ectopy after cardiac surgery is the norm rather than the exception.",
    ),
    (
        "venous thromboembolism",
        "A clot formed while prophylaxis sat unordered for four days.",
        "Thromboembolism remains a hazard we accept in this population despite prophylaxis.",
    ),
    (
        "pressure injury",
        "A sacral sore appeared because turns simply were not being done.",
        "Skin breakdown was foreseeable during a stay of this length and was discussed.",
    ),
    (
        "aspiration",
        "She aspirated when feeds were started before the swallow study came back.",
        "Aspiration is a standing hazard with her neurological deficit.",
    ),
    (
        "intraoperative hypotension",
        "Pressure bottomed out after a dose far too high for her weight.",
        "A dip in pressure on induction is entirely ordinary with these agents.",
    ),
    (
        "catheter infection",
        "The line got infected after it sat in considerably longer than it should have.",
        "Catheter infection is one of the hazards covered in the consent discussion.",
    ),
    (
        "delirium",
        "He became confused on a benzodiazepine he should never have been started on.",
        "Confusion in a man his age after this operation is commonplace.",
    ),
    (
        "anemia",
        "Her count dropped from bleeding nobody picked up until the morning round.",
        "Her count is sitting about where we would predict after this procedure.",
    ),
    (
        "hyperkalemia",
        "Potassium rose after the supplement was mistakenly continued.",
        "A rise in potassium tracks with her declining kidney function.",
    ),
    (
        "readmission",
        "He bounced back within a week with a problem that was already brewing at discharge.",
        "He came back as arranged for the next stage of the reconstruction.",
    ),
    (
        "seizure",
        "She seized when her antiepileptic was held without anyone noticing.",
        "Breakthrough seizures happen in her syndrome even on full therapy.",
    ),
    (
        "respiratory depression",
        "Breathing slowed to the point of needing reversal after the doses were stacked.",
        "Reversal at the conclusion of the case is our standing practice.",
    ),
    (
        "extravasation",
        "The vesicant leaked into the arm because the line was never checked.",
        "Extravasation is a documented hazard of this agent.",
    ),
    (
        "graft failure",
        "The graft clotted much sooner than it had any business doing.",
        "Early graft loss falls inside the failure rate quoted to him.",
    ),
    (
        "nerve injury",
        "The nerve was caught in the retractor during the approach.",
        "Numbness in that distribution is described in the consent paperwork.",
    ),
    (
        "clostridioides difficile colitis",
        "She developed colitis on antibiotics that were continued well past need.",
        "Colitis is a hazard we accept when treating an infection this severe.",
    ),
    (
        "hypoglycemia",
        "Sugar crashed after insulin went in with no tray in front of her.",
        "Low sugars come with the tight control her team has deliberately chosen.",
    ),
    (
        "postoperative ileus",
        "The gut stayed asleep far longer than anything in this case would explain.",
        "Slow return of bowel function is standard after handling the bowel.",
    ),
    (
        "transfusion reaction",
        "She reacted to a unit that was never properly cross-checked.",
        "Febrile reactions are a known accompaniment of transfusion.",
    ),
    (
        "stroke",
        "He stroked after anticoagulation was stopped without any plan to restart.",
        "Stroke risk was quantified for him before he agreed to the operation.",
    ),
    (
        "retained foreign body",
        "A sponge turned up on the film after we had already closed.",
        "The packing was left deliberately and documented for removal on Thursday.",
    ),
    (
        "death",
        "He died of something we had every opportunity to catch.",
        "Death from progressive disease was foreseen and hospice was involved early.",
    ),
)


def contrast_sentences() -> Tuple[List[str], List[str]]:
    """Flatten the pairs into (texts, labels) for ordinary scoring."""
    texts: List[str] = []
    labels: List[str] = []
    for _, unanticipated, expected in CONTRAST_PAIRS:
        texts.extend((unanticipated, expected))
        labels.extend(("unanticipated", "expected_risk"))
    return texts, labels
