"""Structured-data red flags: labs, meds, ICU transfers, OR returns, readmissions.

Every signal here is a classic trigger-tool item: something that is cheap to
detect in structured data and that correlates with harm often enough to be worth
a human look, but never on its own proof of harm. The scoring layer decides how
much each one counts.
"""

from typing import Dict, List, Optional, Tuple

from src.domain.models import PatientCase, PatientEvent
from src.engine.interfaces import FeatureExtractor

# Drugs given essentially only to reverse an overshoot of another drug. Their
# presence implies something went further than intended.
SPECIFIC_ANTIDOTES: Dict[str, str] = {
    "naloxone": "opioid oversedation",
    "flumazenil": "benzodiazepine oversedation",
    "protamine": "heparin over-anticoagulation",
    "idarucizumab": "dabigatran reversal",
    "andexanet": "factor Xa inhibitor reversal",
    "digoxin immune fab": "digoxin toxicity",
    "dantrolene": "malignant hyperthermia",
}

# Agents that reverse an overshoot *or* treat something routine. In the MIMIC-IV
# demo these outnumber the specific antidotes roughly 200 to 1: calcium gluconate
# arrives as "sliding scale (Critical Care-Ionized calcium)" electrolyte
# replacement, acetylcysteine as a mucolytic, dextrose on a hypoglycemia protocol.
# Lumping them in with naloxone is how a trigger tool drowns its reviewers, so
# they get their own lower-weighted feature and the model decides what they cost.
CONTEXT_DEPENDENT_REVERSAL: Dict[str, str] = {
    "phytonadione": "warfarin reversal, or routine vitamin K repletion",
    "vitamin k": "warfarin reversal, or routine vitamin K repletion",
    "glucagon": "hypoglycemia rescue, or a motility agent for imaging",
    "dextrose 50": "hypoglycemia rescue, or protocol glucose replacement",
    "d50": "hypoglycemia rescue, or protocol glucose replacement",
    "acetylcysteine": "acetaminophen toxicity, or a mucolytic",
    "calcium gluconate": "hyperkalemia rescue, or calcium repletion",
    "sodium polystyrene": "hyperkalemia rescue, or routine potassium binding",
    "sugammadex": "residual blockade, but used routinely at the end of anesthesia",
}

ANTIDOTES: Dict[str, str] = {**SPECIFIC_ANTIDOTES, **CONTEXT_DEPENDENT_REVERSAL}

# Real unit names spell this several ways: "Medical Intensive Care Unit (MICU)",
# "Coronary Care Unit (CCU)", "Neuro Stepdown".
ESCALATION_UNIT_TOKENS = (
    "icu",
    "intensive",
    "ccu",
    "coronary care",
    "step-down",
    "stepdown",
    "neuro intermediate",
)

# Coded procedures that are, by their own wording, a return to theatre. Real ICD
# titles never say "unplanned", so keyword heuristics tuned on prose miss them:
# "Reopening of recent laparotomy site" and "Control Bleeding in Abdominal Wall,
# Open Approach" both appear in the MIMIC demo.
REOPERATION_PHRASES = (
    "reopening of",
    "reclosure",
    "control bleeding",
    "control of hemorrhage",
    "re-exploration",
    "reexploration",
    "evacuation of hematoma",
)

BLOOD_PRODUCT_PHRASES = (
    "transfus",
    "prbc",
    "packed red",
    "packed rbc",
    "whole blood",
    "fresh frozen plasma",
    "cryoprecipitate",
    "platelets",
)

# lab name -> (direction, abnormal threshold, critical threshold)
LabRule = Tuple[str, float, float]
LAB_RULES: Dict[str, LabRule] = {
    "hemoglobin": ("low", 9.0, 7.0),
    "hgb": ("low", 9.0, 7.0),
    "platelets": ("low", 100.0, 50.0),
    "glucose": ("low", 70.0, 50.0),
    "creatinine": ("high", 1.5, 2.5),
    "potassium": ("high", 5.5, 6.5),
    "lactate": ("high", 2.5, 4.0),
    "inr": ("high", 3.0, 4.5),
    "troponin": ("high", 0.04, 0.50),
    "wbc": ("high", 12.0, 18.0),
    "bilirubin": ("high", 2.0, 5.0),
    "alt": ("high", 120.0, 400.0),
}

TREATMENT_EVENT_TYPES = ("procedure", "medication")

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "antidote_given": "A specific antidote was administered",
    "antidote_after_treatment_start": "Antidote given after our treatment began, not on arrival",
    "context_dependent_reversal_agent": "A reversal agent that also has routine uses was given",
    "unplanned_icu_transfer": "Unplanned transfer to a higher level of care",
    "repeat_icu_escalation": "Escalated to intensive care more than once in one stay",
    "unplanned_return_to_or": "Unplanned return to the operating room",
    "readmission_30d": "Readmission within 30 days of discharge",
    "rapid_response_called": "Rapid response or code team activated",
    "transfusion_given": "Blood product transfused",
    "abnormal_post_treatment_lab": "Abnormal lab drawn after treatment began",
    "critical_post_treatment_lab": "Critically abnormal lab drawn after treatment began",
    "hemoglobin_drop": "Hemoglobin fell 2 g/dL or more between draws",
    "creatinine_doubled": "Creatinine at least doubled from baseline (AKI pattern)",
    "abnormal_lab_types_count": "How many distinct lab types were abnormal after treatment",
}


class StructuredDataWatcher(FeatureExtractor):
    """Monitors structured data (labs, meds, transfers, encounters) for red flags.

    `anchor` sets the line between what the patient arrived with and what happened
    under our care. "procedure" anchors on the index procedure, which suits an
    operative episode. "admission" anchors on arrival, which suits a whole
    admission where procedure times may be coded to the day rather than the minute.
    """

    def __init__(self, anchor: str = "procedure"):
        if anchor not in ("procedure", "admission"):
            raise ValueError(f"anchor must be 'procedure' or 'admission', got {anchor!r}")
        self.anchor = anchor

    def feature_descriptions(self) -> Dict[str, str]:
        return dict(FEATURE_DESCRIPTIONS)

    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        features = {name: 0.0 for name in FEATURE_DESCRIPTIONS}
        events = case.sorted_events()
        treatment_start = self._treatment_start(events)
        escalations = 0

        for event in events:
            text = f"{event.value} {event.details}".lower()

            if event.event_type == "medication":
                specific = self._matched_antidote(event.value, SPECIFIC_ANTIDOTES)
                if specific is not None:
                    features["antidote_given"] = 1.0
                    # An antidote given before we treated the patient reverses
                    # something they arrived with, not something we caused.
                    if treatment_start is not None and event.timestamp > treatment_start:
                        features["antidote_after_treatment_start"] = 1.0
                elif self._matched_antidote(event.value, CONTEXT_DEPENDENT_REVERSAL) is not None:
                    features["context_dependent_reversal_agent"] = 1.0
                if any(phrase in text for phrase in BLOOD_PRODUCT_PHRASES):
                    features["transfusion_given"] = 1.0

            elif event.event_type == "transfer":
                escalation = any(k in event.value.lower() for k in ESCALATION_UNIT_TOKENS)
                # A transfer planned before the fact is not a red flag; an
                # unplanned one usually means the patient deteriorated.
                if escalation and "planned" not in text.replace("unplanned", ""):
                    features["unplanned_icu_transfer"] = 1.0
                    escalations += 1
                if "rapid response" in text or "code blue" in text:
                    features["rapid_response_called"] = 1.0

            elif event.event_type == "procedure":
                if self._is_unplanned_or_return(event):
                    features["unplanned_return_to_or"] = 1.0
                if any(phrase in text for phrase in BLOOD_PRODUCT_PHRASES):
                    features["transfusion_given"] = 1.0

            elif event.event_type == "admission":
                if "readmission" in text:
                    features["readmission_30d"] = 1.0

            elif event.event_type == "note":
                if "rapid response" in text or "code blue" in text:
                    features["rapid_response_called"] = 1.0

        features["repeat_icu_escalation"] = 1.0 if escalations >= 2 else 0.0

        abnormal, critical = self._post_treatment_lab_flags(events, treatment_start)
        # Distinct lab types rather than raw draws. A real ICU admission racks up
        # hundreds of abnormal results, and a count that scales with length of stay
        # would swamp every other signal.
        features["abnormal_lab_types_count"] = float(len({lab for lab, _, _ in abnormal}))
        features["abnormal_post_treatment_lab"] = 1.0 if abnormal else 0.0
        features["critical_post_treatment_lab"] = 1.0 if critical else 0.0

        trends = self._lab_trends(events)
        features["hemoglobin_drop"] = 1.0 if trends["hemoglobin_drop"] else 0.0
        features["creatinine_doubled"] = 1.0 if trends["creatinine_doubled"] else 0.0

        return features

    def explain(self, case: PatientCase) -> List[str]:
        """Reviewer-facing bullets naming the specific triggering values."""
        notes: List[str] = []
        events = case.sorted_events()
        treatment_start = self._treatment_start(events)

        for event in events:
            if event.event_type == "medication":
                antidote = self._matched_antidote(event.value)
                if antidote is not None:
                    drug, reason = antidote
                    timing = (
                        "after treatment began"
                        if treatment_start is not None and event.timestamp > treatment_start
                        else "on arrival, before our treatment"
                    )
                    tier = "Antidote" if drug in SPECIFIC_ANTIDOTES else "Reversal agent"
                    notes.append(f"{tier} **{drug.title()}** given {timing} ({reason})")
            elif event.event_type == "transfer" and "icu" in event.value.lower():
                notes.append(f"Transfer to {event.value}: {event.details}")
            elif event.event_type == "procedure" and self._is_unplanned_or_return(event):
                notes.append(f"Unplanned OR return: {event.value} — {event.details}")
            elif event.event_type == "admission" and "readmission" in f"{event.value} {event.details}".lower():
                notes.append(f"{event.value}: {event.details}")

        abnormal, critical = self._post_treatment_lab_flags(events, treatment_start)
        for lab, value, severity in abnormal:
            marker = "critical" if (lab, value, severity) in critical else "abnormal"
            notes.append(f"Post-treatment {lab} = {value:g} ({marker})")

        trends = self._lab_trends(events)
        if trends["hemoglobin_drop"]:
            notes.append(f"Hemoglobin dropped {trends['hemoglobin_drop']:.1f} g/dL between draws")
        if trends["creatinine_doubled"]:
            notes.append(f"Creatinine rose {trends['creatinine_doubled']:.1f}x from baseline")

        return notes

    @staticmethod
    def _matched_antidote(
        drug: str, catalog: Optional[Dict[str, str]] = None
    ) -> Optional[Tuple[str, str]]:
        name = drug.lower()
        for antidote, reason in (catalog or ANTIDOTES).items():
            if antidote in name:
                return antidote, reason
        return None

    @staticmethod
    def _is_unplanned_or_return(event: PatientEvent) -> bool:
        text = f"{event.value} {event.details}".lower()
        # A coded reoperation states what it is; prose has to say it was unplanned.
        if any(phrase in text for phrase in REOPERATION_PHRASES):
            return True
        in_or = any(k in text for k in ("operating room", " or ", "reoperation", "re-operation", "surgery", "washout"))
        unplanned = any(k in text for k in ("unplanned", "emergent", "return to", "takeback", "take-back", "unscheduled"))
        return in_or and unplanned

    def _treatment_start(self, events: List[PatientEvent]) -> Optional[str]:
        """When our care began, which is the line between baseline and consequence.

        Anything before it the patient arrived with.
        """
        if self.anchor == "admission":
            for event in events:
                if event.event_type == "admission":
                    return event.timestamp
        for event in events:
            if event.event_type == "procedure":
                return event.timestamp
        for event in events:
            if event.event_type in TREATMENT_EVENT_TYPES:
                return event.timestamp
        return None

    @staticmethod
    def _post_treatment_lab_flags(
        events: List[PatientEvent], treatment_start: Optional[str]
    ) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]:
        abnormal: List[Tuple[str, float, str]] = []
        critical: List[Tuple[str, float, str]] = []

        for event in events:
            if event.event_type != "lab":
                continue
            if treatment_start is not None and event.timestamp <= treatment_start:
                continue  # baseline draw, not a post-treatment result

            value = event.numeric_details
            rule = LAB_RULES.get(event.value.strip().lower())
            if value is None or rule is None:
                continue

            direction, abnormal_cut, critical_cut = rule
            is_abnormal = value < abnormal_cut if direction == "low" else value > abnormal_cut
            is_critical = value < critical_cut if direction == "low" else value > critical_cut
            if is_abnormal:
                record = (event.value, value, "critical" if is_critical else "abnormal")
                abnormal.append(record)
                if is_critical:
                    critical.append(record)

        return abnormal, critical

    @staticmethod
    def _lab_trends(events: List[PatientEvent]) -> Dict[str, float]:
        """Deltas matter more than single values for bleeding and kidney injury."""
        series: Dict[str, List[float]] = {}
        for event in events:
            if event.event_type != "lab":
                continue
            value = event.numeric_details
            if value is None:
                continue
            series.setdefault(event.value.strip().lower(), []).append(value)

        hgb = series.get("hemoglobin") or series.get("hgb") or []
        drop = max((hgb[0] - v for v in hgb[1:]), default=0.0)

        creat = series.get("creatinine") or []
        ratio = max((v / creat[0] for v in creat[1:]), default=0.0) if creat and creat[0] > 0 else 0.0

        return {
            "hemoglobin_drop": drop if drop >= 2.0 else 0.0,
            "creatinine_doubled": ratio if ratio >= 2.0 else 0.0,
        }
