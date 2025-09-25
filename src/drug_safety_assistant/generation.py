from __future__ import annotations

from .types import (
    EvidenceCitation,
    EvidencePack,
    GeneratedClaim,
    Intent,
    RiskBreakdown,
    StructuredResponse,
)


def generate_claims(pack: EvidencePack) -> list[GeneratedClaim]:
    claims: list[GeneratedClaim] = []

    openfda_items = pack.by_source("openfda")
    if openfda_items:
        item = openfda_items[0]
        claims.append(
            GeneratedClaim(
                text=f"Regulatory labeling notes key safety concerns: {item.snippet[:180]}",
                citation_ids=[item.citation_id],
            )
        )

    pubmed_items = sorted(pack.by_source("pubmed"), key=lambda x: x.strength_score, reverse=True)
    if pubmed_items:
        item = pubmed_items[0]
        claims.append(
            GeneratedClaim(
                text=(
                    "Clinical literature reports safety findings relevant to this "
                    f"question: {item.snippet[:180]}"
                ),
                citation_ids=[item.citation_id],
            )
        )

    faers_items = pack.by_source("faers")
    if faers_items:
        item = faers_items[0]
        prr = item.metadata.get("prr")
        if prr is not None:
            claims.append(
                GeneratedClaim(
                    text=(
                        f"Post-marketing signal detected with PRR={float(prr):.2f}: "
                        f"{item.snippet[:180]}"
                    ),
                    citation_ids=[item.citation_id],
                )
            )
        else:
            claims.append(
                GeneratedClaim(
                    text=f"Post-marketing signal summary: {item.snippet[:180]}",
                    citation_ids=[item.citation_id],
                )
            )

    return claims


def _default_monitoring_recommendations(intent: Intent, risk: RiskBreakdown) -> list[str]:
    recs: list[str] = []

    if risk.risk_level.value == "High":
        recs.append(
            "Increase monitoring intensity for serious adverse effects and clinical deterioration."
        )
    elif risk.risk_level.value == "Moderate":
        recs.append("Use targeted monitoring for adverse effects relevant to this risk profile.")
    else:
        recs.append("Routine safety monitoring appears reasonable based on current evidence.")

    if intent == Intent.INTERACTION:
        recs.append(
            "Reconcile all concomitant medications and monitor for interaction-specific symptoms."
        )
    if intent == Intent.PREGNANCY:
        recs.append("Use obstetric-focused monitoring and reassess as trimester status changes.")
    if intent == Intent.RENAL:
        recs.append(
            "Follow renal function trends and monitor for accumulation-related adverse events."
        )

    recs.append("Escalate to a clinician if new serious symptoms emerge.")
    return recs


def compose_structured_response(
    *,
    intent: Intent,
    risk: RiskBreakdown,
    validated_claims: list[GeneratedClaim],
    pack: EvidencePack,
    guard_supported_ratio: float,
    follow_up_questions: list[str] | None = None,
) -> StructuredResponse:
    follow_ups = follow_up_questions or []

    if validated_claims:
        summary_text = " ".join(claim.text for claim in validated_claims)
    else:
        summary_text = (
            "Retrieved evidence is limited or weakly supportive for definitive safety conclusions."
        )

    used_ids = [cid for claim in validated_claims for cid in claim.citation_ids]
    if not used_ids:
        used_ids = [item.citation_id for item in pack.items[:5]]

    citations: list[EvidenceCitation] = []
    seen: set[str] = set()
    for evidence in pack.items:
        if evidence.citation_id in used_ids and evidence.citation_id not in seen:
            citations.append(
                EvidenceCitation(
                    citation_id=evidence.citation_id,
                    source=evidence.source,
                    title=evidence.title,
                    details=evidence.snippet[:220],
                )
            )
            seen.add(evidence.citation_id)

    uncertainty_parts = []
    if not pack.by_source("openfda"):
        uncertainty_parts.append("No OpenFDA label content retrieved")
    if not pack.by_source("pubmed"):
        uncertainty_parts.append("no PubMed evidence retrieved")
    if not pack.by_source("faers"):
        uncertainty_parts.append("no FAERS signal data retrieved")
    if guard_supported_ratio < 1:
        uncertainty_parts.append("one or more claims failed support checks")

    uncertainty = (
        "Uncertainty due to " + ", ".join(uncertainty_parts) + "."
        if uncertainty_parts
        else "Evidence support is internally consistent across retrieved sources."
    )

    return StructuredResponse(
        intent=intent,
        follow_up_questions=follow_ups,
        safety_summary=summary_text,
        risk_level=risk.risk_level,
        risk_score=risk.weighted_score,
        evidence_sources=citations,
        monitoring_recommendations=_default_monitoring_recommendations(intent, risk),
        uncertainty_statement=uncertainty,
        guard_supported_ratio=guard_supported_ratio,
    )
