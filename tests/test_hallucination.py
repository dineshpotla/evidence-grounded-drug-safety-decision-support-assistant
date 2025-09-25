from drug_safety_assistant.hallucination import run_hallucination_guard
from drug_safety_assistant.types import EvidencePack, GeneratedClaim, RetrievedEvidence


def test_claim_removed_when_citation_missing() -> None:
    pack = EvidencePack(items=[])
    claims = [GeneratedClaim(text="Unsupported claim", citation_ids=[])]

    result = run_hallucination_guard(claims=claims, pack=pack)
    assert len(result.validated_claims) == 0
    assert len(result.removed_claims) == 1


def test_numeric_validation_blocks_wrong_prr() -> None:
    pack = EvidencePack(
        items=[
            RetrievedEvidence(
                source="faers",
                citation_id="FAERS:X",
                title="FAERS",
                snippet="Signal",
                metadata={"prr": 1.5},
                strength_score=2,
            )
        ]
    )

    claims = [
        GeneratedClaim(
            text="Post-marketing signal shows PRR=2.4",
            citation_ids=["FAERS:X"],
        )
    ]

    result = run_hallucination_guard(claims=claims, pack=pack)
    assert len(result.validated_claims) == 0
    assert len(result.removed_claims) == 1
