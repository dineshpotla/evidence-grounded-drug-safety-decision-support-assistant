from __future__ import annotations

from .intents import missing_slots
from .types import Intent, SafetyRequest

SAFE_SLOT_QUESTIONS: dict[str, str] = {
    "drug": "Which drug are you asking about?",
    "drug_a": "What is the first drug?",
    "drug_b": "What is the second drug?",
    "trimester": "If pregnant, which trimester (1, 2, or 3)?",
    "age_group": "Which age group applies (pediatric, adult, over 65)?",
    "pregnancy_status": "Is the patient pregnant (yes/no/unknown)?",
    "kidney_status": "Do they have kidney disease (none/CKD/dialysis)?",
    "liver_status": "Do they have liver disease (none/mild/moderate/severe)?",
}


def generate_follow_up_questions(
    request: SafetyRequest,
    intent: Intent,
    max_questions: int = 3,
) -> list[str]:
    slots = missing_slots(request, intent)
    if not slots:
        return []

    questions = [SAFE_SLOT_QUESTIONS[slot] for slot in slots if slot in SAFE_SLOT_QUESTIONS]
    return questions[:max_questions]
