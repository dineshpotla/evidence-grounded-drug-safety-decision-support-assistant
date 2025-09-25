from __future__ import annotations

from collections.abc import Iterable

from .types import Intent, SafetyRequest

REQUIRED_SLOTS: dict[Intent, set[str]] = {
    Intent.INTERACTION: {"drug_a", "drug_b"},
    Intent.PREGNANCY: {"drug", "trimester"},
    Intent.RENAL: {"drug", "kidney_status"},
    Intent.PATIENT_SPECIFIC: {
        "drug",
        "age_group",
        "pregnancy_status",
        "kidney_status",
        "liver_status",
    },
    Intent.GENERAL: {"drug"},
}

INTERACTION_KEYWORDS = {
    "interaction",
    "interact",
    "combine",
    "together",
    "with",
    "co-administer",
    "coadminister",
}
PREGNANCY_KEYWORDS = {"pregnancy", "pregnant", "trimester", "fetal", "lactation", "breastfeeding"}
RENAL_KEYWORDS = {"ckd", "kidney", "renal", "dialysis", "egfr", "creatinine"}


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in keywords)


def classify_intent(request: SafetyRequest) -> Intent:
    text = request.question.lower()

    if request.drug_a and request.drug_b:
        return Intent.INTERACTION
    if _contains_any(text, INTERACTION_KEYWORDS):
        return Intent.INTERACTION

    if request.pregnancy_status and request.pregnancy_status.lower() in {"pregnant", "yes"}:
        return Intent.PREGNANCY
    if _contains_any(text, PREGNANCY_KEYWORDS):
        return Intent.PREGNANCY

    if request.kidney_status and request.kidney_status.lower() != "none":
        return Intent.RENAL
    if _contains_any(text, RENAL_KEYWORDS):
        return Intent.RENAL

    if request.age_group or request.current_meds or request.liver_status:
        return Intent.PATIENT_SPECIFIC

    return Intent.GENERAL


def missing_slots(request: SafetyRequest, intent: Intent) -> list[str]:
    missing: list[str] = []
    required = REQUIRED_SLOTS.get(intent, set())

    for slot in required:
        value = getattr(request, slot, None)
        if value is None:
            missing.append(slot)
            continue

        if isinstance(value, str) and not value.strip():
            missing.append(slot)
        elif isinstance(value, list) and len(value) == 0:
            missing.append(slot)

    return sorted(missing)
