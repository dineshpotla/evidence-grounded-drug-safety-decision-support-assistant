from drug_safety_assistant.retrieval.aggregator import filter_retrieved_evidence
from drug_safety_assistant.types import Intent, RetrievedEvidence, SafetyRequest


def test_filter_retrieved_evidence_drops_unrelated_publication() -> None:
    evidence = [
        RetrievedEvidence(
            source="pubmed",
            citation_id="PMID:irrelevant",
            title="Metformin and contrast medium guidance",
            snippet="Recommendations for contrast induced kidney injury and metformin.",
            metadata={},
            strength_score=1,
        ),
        RetrievedEvidence(
            source="openfda",
            citation_id="OPENFDA:dapsone",
            title="Dapsone Label",
            snippet=(
                "Topical dapsone and benzoyl peroxide can cause temporary "
                "yellow/orange discoloration."
            ),
            metadata={},
            strength_score=2,
        ),
    ]

    request = SafetyRequest(
        question="Can benzoyl peroxide be used with dapsone gel?",
        drug_a="benzoyl peroxide",
        drug_b="dapsone",
    )

    filtered = filter_retrieved_evidence(
        evidence=evidence,
        request=request,
        intent=Intent.INTERACTION,
    )

    citation_ids = [item.citation_id for item in filtered]
    assert "OPENFDA:dapsone" in citation_ids
    assert "PMID:irrelevant" not in citation_ids
