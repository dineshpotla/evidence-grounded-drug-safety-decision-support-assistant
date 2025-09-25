from drug_safety_assistant.risk_scoring import compute_risk_score
from drug_safety_assistant.types import EvidencePack, RetrievedEvidence, RiskLevel, SafetyRequest


def test_high_risk_from_label_and_faers() -> None:
    pack = EvidencePack(
        items=[
            RetrievedEvidence(
                source="openfda",
                citation_id="OPENFDA:1",
                title="Label",
                snippet="Boxed warning about life-threatening toxicity",
                metadata={"label_severity": 3},
                strength_score=3,
            ),
            RetrievedEvidence(
                source="faers",
                citation_id="FAERS:1",
                title="Signal",
                snippet="PRR=2.4",
                metadata={"prr": 2.4},
                strength_score=3,
            ),
        ]
    )
    request = SafetyRequest(
        question="Is this safe?",
        drug="example",
        age_group="over 65",
        kidney_status="CKD",
    )

    risk = compute_risk_score(request=request, pack=pack)
    assert risk.risk_level == RiskLevel.HIGH
    assert risk.weighted_score >= 2.0
