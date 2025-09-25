from __future__ import annotations

from ..config import settings
from ..evidence import build_evidence_pack, evidence_pack_to_context
from ..generation import compose_structured_response
from ..hallucination import run_hallucination_guard
from ..intents import classify_intent
from ..llm.claude import ClaudeAgent
from ..llm.nvidia_extractor import NvidiaIntentExtractor
from ..prompts import render_prompt
from ..question_engine import generate_follow_up_questions
from ..retrieval.agentic import AgenticRetriever
from ..risk_scoring import compute_risk_score
from ..types import EvidenceChunk, Intent, RiskLevel, SafetyRequest, StructuredResponse
from ..utils.drug_resolver import DrugNameResolver


class DrugSafetyAssistant:
    def __init__(
        self,
        retriever: AgenticRetriever | None = None,
        extractor: NvidiaIntentExtractor | None = None,
        claude_agent: ClaudeAgent | None = None,
    ) -> None:
        self.retriever = retriever or AgenticRetriever()
        self.extractor = extractor or NvidiaIntentExtractor()
        self.claude_agent = claude_agent or ClaudeAgent()
        self.drug_resolver = DrugNameResolver()

    def assess(self, request: SafetyRequest) -> StructuredResponse:
        entities = self.extractor.extract(request)
        enriched_request = self.extractor.enrich_request(request, entities)
        enriched_request = self.drug_resolver.resolve_request(enriched_request)

        intent = entities.intent_hint or classify_intent(enriched_request)
        follow_ups = generate_follow_up_questions(request=enriched_request, intent=intent)

        if follow_ups:
            return StructuredResponse(
                intent=intent,
                follow_up_questions=follow_ups,
                safety_summary=(
                    "Additional clinical context is required before producing a "
                    "grounded safety answer."
                ),
                risk_level=RiskLevel.LOW,
                risk_score=0.0,
                evidence_sources=[],
                monitoring_recommendations=[
                    "Provide the missing context to continue evidence-grounded risk assessment."
                ],
                uncertainty_statement=(
                    "No retrieval performed yet because required clinical slots are missing."
                ),
                guard_supported_ratio=1.0,
            )

        evidence_items = self.retriever.retrieve(request=enriched_request, intent=intent)
        pack = build_evidence_pack(evidence_items)

        patient_context = self._patient_context_text(enriched_request)
        evidence_text = evidence_pack_to_context(pack)
        prompt = render_prompt(
            intent=intent,
            request=enriched_request,
            evidence_pack_text=evidence_text,
            prompt_version=settings.prompt_version,
            patient_context_text=patient_context,
        )

        claim_chunks = [
            EvidenceChunk(
                chunk_id=f"{item.citation_id}::claim",
                source=item.source,
                parent_citation_id=item.citation_id,
                text=item.snippet,
                metadata=item.metadata,
            )
            for item in pack.items
        ]
        claims = self.claude_agent.generate_claims(question=prompt, chunks=claim_chunks)
        guard = run_hallucination_guard(claims=claims, pack=pack)

        risk = compute_risk_score(request=enriched_request, pack=pack)

        return compose_structured_response(
            intent=intent,
            risk=risk,
            validated_claims=guard.validated_claims,
            pack=pack,
            guard_supported_ratio=guard.supported_ratio,
            follow_up_questions=[],
        )

    def _patient_context_text(self, request: SafetyRequest) -> str:
        parts = []
        if request.age_group:
            parts.append(f"age_group={request.age_group}")
        if request.pregnancy_status:
            parts.append(f"pregnancy_status={request.pregnancy_status}")
        if request.trimester:
            parts.append(f"trimester={request.trimester}")
        if request.kidney_status:
            parts.append(f"kidney_status={request.kidney_status}")
        if request.liver_status:
            parts.append(f"liver_status={request.liver_status}")
        if request.current_meds:
            parts.append(f"current_meds={','.join(request.current_meds)}")
        return "; ".join(parts) if parts else "no additional context"


def infer_intent_only(request: SafetyRequest) -> Intent:
    extractor = NvidiaIntentExtractor()
    entities = extractor.extract(request)
    if entities.intent_hint:
        return entities.intent_hint

    enriched_request = extractor.enrich_request(request, entities)
    return classify_intent(enriched_request)
