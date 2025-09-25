from drug_safety_assistant.intents import classify_intent
from drug_safety_assistant.question_engine import generate_follow_up_questions
from drug_safety_assistant.types import Intent, SafetyRequest


def test_interaction_missing_slots_generates_safe_questions() -> None:
    request = SafetyRequest(question="Any interaction risk?")
    intent = classify_intent(request)
    assert intent == Intent.INTERACTION

    questions = generate_follow_up_questions(request=request, intent=intent)
    assert len(questions) <= 3
    assert "first drug" in questions[0].lower() or "second drug" in questions[0].lower()


def test_general_with_drug_no_followup() -> None:
    request = SafetyRequest(question="Is ibuprofen safe?", drug="ibuprofen")
    intent = classify_intent(request)
    questions = generate_follow_up_questions(request=request, intent=intent)
    assert questions == []
