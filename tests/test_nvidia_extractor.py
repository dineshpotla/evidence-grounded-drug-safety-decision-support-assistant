from drug_safety_assistant.llm.nvidia_extractor import NvidiaIntentExtractor
from drug_safety_assistant.types import Intent, SafetyRequest


def test_heuristic_extract_interaction_entities() -> None:
    extractor = NvidiaIntentExtractor(api_key="")
    request = SafetyRequest(question="Does warfarin interact with amiodarone in elderly patients?")

    entities = extractor.extract(request)

    assert entities.intent_hint == Intent.INTERACTION
    lowered = [item.lower() for item in entities.drug_mentions]
    assert "warfarin" in lowered
    assert "amiodarone" in lowered
    assert entities.age_group == "over 65"


def test_enrich_request_populates_missing_slots() -> None:
    extractor = NvidiaIntentExtractor(api_key="")
    request = SafetyRequest(question="Is metformin safe in CKD?")

    entities = extractor.extract(request)
    enriched = extractor.enrich_request(request, entities)

    assert enriched.drug is not None
    assert enriched.kidney_status == "CKD"
