from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    INTERACTION = "interaction"
    PREGNANCY = "pregnancy"
    RENAL = "renal"
    PATIENT_SPECIFIC = "patient_specific"
    GENERAL = "general"


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class SafetyRequest(BaseModel):
    question: str = Field(min_length=3)
    drug: Optional[str] = None
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    age_group: Optional[str] = None
    pregnancy_status: Optional[str] = None
    trimester: Optional[str] = None
    kidney_status: Optional[str] = None
    liver_status: Optional[str] = None
    current_meds: List[str] = Field(default_factory=list)


class ExtractedEntities(BaseModel):
    intent_hint: Optional[Intent] = None
    drug_mentions: List[str] = Field(default_factory=list)
    age_group: Optional[str] = None
    pregnancy_status: Optional[str] = None
    trimester: Optional[str] = None
    kidney_status: Optional[str] = None
    liver_status: Optional[str] = None


class EvidenceChunk(BaseModel):
    chunk_id: str
    source: str
    parent_citation_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    source: str
    citation_id: str
    title: str
    snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    strength_score: int = 0


class EvidencePack(BaseModel):
    items: List[RetrievedEvidence] = Field(default_factory=list)

    def by_source(self, source: str) -> List[RetrievedEvidence]:
        return [item for item in self.items if item.source == source]

    def citation_ids(self) -> set[str]:
        return {item.citation_id for item in self.items}


class GeneratedClaim(BaseModel):
    text: str
    citation_ids: List[str] = Field(default_factory=list)


class HallucinationIssue(BaseModel):
    issue_type: str
    detail: str


class HallucinationGuardResult(BaseModel):
    validated_claims: List[GeneratedClaim] = Field(default_factory=list)
    removed_claims: List[GeneratedClaim] = Field(default_factory=list)
    issues: List[HallucinationIssue] = Field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        total = len(self.validated_claims) + len(self.removed_claims)
        if total == 0:
            return 1.0
        return len(self.validated_claims) / total


class EvidenceCitation(BaseModel):
    citation_id: str
    source: str
    title: str
    details: str


class RiskBreakdown(BaseModel):
    label_severity: int
    literature_strength: int
    faers_signal: int
    patient_modifiers: int
    weighted_score: float
    risk_level: RiskLevel
    explanation: str


class StructuredResponse(BaseModel):
    intent: Intent
    follow_up_questions: List[str] = Field(default_factory=list)
    safety_summary: str
    risk_level: RiskLevel
    risk_score: float
    evidence_sources: List[EvidenceCitation] = Field(default_factory=list)
    monitoring_recommendations: List[str] = Field(default_factory=list)
    uncertainty_statement: str
    guard_supported_ratio: float = 1.0

    def to_markdown(self) -> str:
        lines = [
            "### Safety Summary",
            self.safety_summary,
            "",
            "### Risk Level",
            f"{self.risk_level.value} ({self.risk_score:.2f})",
            "",
            "### Evidence Sources",
        ]

        if self.evidence_sources:
            for citation in self.evidence_sources:
                lines.append(
                    f"- [{citation.citation_id}] {citation.title} "
                    f"({citation.source}): {citation.details}"
                )
        else:
            lines.append("- No evidence citations available.")

        lines.extend(["", "### Monitoring Recommendations"])
        if self.monitoring_recommendations:
            lines.extend([f"- {item}" for item in self.monitoring_recommendations])
        else:
            lines.append("- No specific monitoring recommendation generated.")

        lines.extend(
            [
                "",
                "### Uncertainty Statement",
                self.uncertainty_statement,
                "",
                f"Support ratio after hallucination checks: {self.guard_supported_ratio:.2%}",
            ]
        )

        if self.follow_up_questions:
            lines.extend(["", "### Follow-up Questions"])
            lines.extend([f"- {q}" for q in self.follow_up_questions])

        return "\n".join(lines)
