from drug_safety_assistant.utils.clinical_text import contains_drug_term, normalize_drug_name


def test_normalize_drug_name_removes_form_and_strength() -> None:
    assert normalize_drug_name("Dapsone 7.5% Gel") == "dapsone"
    assert normalize_drug_name("benzoyl peroxide topical") == "benzoyl peroxide"


def test_contains_drug_term_handles_typo_fuzzily() -> None:
    text = "Topical dapsone gel can discolor when combined with benzoyl peroxide"
    assert contains_drug_term(text, "daposone")
