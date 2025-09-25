from __future__ import annotations

import re

from .types import (
    EvidencePack,
    GeneratedClaim,
    HallucinationGuardResult,
    HallucinationIssue,
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "drug",
    "patient",
    "risk",
    "is",
    "are",
    "to",
    "of",
    "in",
}


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _check_claim_entailment(claim: GeneratedClaim, evidence_text: str) -> bool:
    claim_tokens = _keywords(claim.text)
    evidence_tokens = _keywords(evidence_text)
    overlap = claim_tokens & evidence_tokens
    return len(overlap) >= 2


def run_hallucination_guard(
    claims: list[GeneratedClaim], pack: EvidencePack
) -> HallucinationGuardResult:
    result = HallucinationGuardResult()
    evidence_by_id = {item.citation_id: item for item in pack.items}

    for claim in claims:
        if not claim.citation_ids:
            result.removed_claims.append(claim)
            result.issues.append(
                HallucinationIssue(
                    issue_type="missing_citation",
                    detail=f"Claim has no citation: {claim.text}",
                )
            )
            continue

        unknown = [cid for cid in claim.citation_ids if cid not in evidence_by_id]
        if unknown:
            result.removed_claims.append(claim)
            result.issues.append(
                HallucinationIssue(
                    issue_type="invalid_citation",
                    detail=f"Claim cites unknown IDs {unknown}: {claim.text}",
                )
            )
            continue

        evidence_text = " ".join(evidence_by_id[cid].snippet for cid in claim.citation_ids)
        if not _check_claim_entailment(claim, evidence_text):
            result.removed_claims.append(claim)
            result.issues.append(
                HallucinationIssue(
                    issue_type="consistency_failure",
                    detail=f"Claim weakly supported by cited evidence: {claim.text}",
                )
            )
            continue

        if not _validate_numeric_claims(claim, evidence_by_id):
            result.removed_claims.append(claim)
            result.issues.append(
                HallucinationIssue(
                    issue_type="numeric_mismatch",
                    detail=f"Numeric value mismatch for claim: {claim.text}",
                )
            )
            continue

        result.validated_claims.append(claim)

    return result


def _validate_numeric_claims(claim: GeneratedClaim, evidence_by_id: dict[str, object]) -> bool:
    # Checks statements like "PRR=1.87" against FAERS metadata in cited evidence.
    match = re.search(r"PRR\s*[=:]\s*([0-9]*\.?[0-9]+)", claim.text, flags=re.IGNORECASE)
    if not match:
        return True

    claimed = float(match.group(1))
    for citation_id in claim.citation_ids:
        evidence = evidence_by_id.get(citation_id)
        if evidence is None:
            continue
        metadata = getattr(evidence, "metadata", {})
        if "prr" in metadata:
            actual = float(metadata["prr"])
            return abs(actual - claimed) <= 0.1

    return False
