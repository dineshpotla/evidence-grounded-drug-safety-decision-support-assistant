from pathlib import Path

from drug_safety_assistant.types import SafetyRequest
from drug_safety_assistant.utils.drug_resolver import DrugNameResolver


def test_resolve_misspelled_drug_names(tmp_path: Path) -> None:
    dictionary = tmp_path / "drug_dict.txt"
    dictionary.write_text("dapsone\nbenzoyl peroxide\n", encoding="utf-8")

    resolver = DrugNameResolver(dictionary_path=str(dictionary), threshold=0.84)
    request = SafetyRequest(
        question="Can benzole peroxide be used with daposone 7.5% gel?",
        drug_a="benzole peroxide topical",
        drug_b="daposone 7.5% gel",
    )

    resolved = resolver.resolve_request(request)

    assert resolved.drug_a == "benzoyl peroxide"
    assert resolved.drug_b == "dapsone"
