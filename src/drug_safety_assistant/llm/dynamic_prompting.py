from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..types import EvidenceChunk


@dataclass(frozen=True)
class DynamicPromptContext:
    focus: str
    source_priority: list[str]
    max_claims: int
    evidence_budget: int
    query_terms: list[str]


def build_prompt_context(
    question: str,
    *,
    chunks: list[EvidenceChunk] | None = None,
) -> DynamicPromptContext:
    lowered = question.lower()
    query_terms = _important_terms(question)
    evidence_budget = len(chunks or [])

    if _is_interaction_query(lowered):
        return DynamicPromptContext(
            focus="interaction",
            source_priority=["openfda", "faers", "pubmed"],
            max_claims=4,
            evidence_budget=evidence_budget,
            query_terms=query_terms,
        )

    if _is_pregnancy_query(lowered):
        return DynamicPromptContext(
            focus="pregnancy",
            source_priority=["openfda", "pubmed", "faers"],
            max_claims=3,
            evidence_budget=evidence_budget,
            query_terms=query_terms,
        )

    if _is_renal_query(lowered):
        return DynamicPromptContext(
            focus="renal",
            source_priority=["openfda", "pubmed", "faers"],
            max_claims=3,
            evidence_budget=evidence_budget,
            query_terms=query_terms,
        )

    return DynamicPromptContext(
        focus="general",
        source_priority=["openfda", "pubmed", "faers"],
        max_claims=3,
        evidence_budget=evidence_budget,
        query_terms=query_terms,
    )


def extraction_directives(question: str) -> str:
    lowered = question.lower()
    lines = [
        "- Always output strict JSON matching the schema.",
        "- Extract canonical drug mentions without dosage/form if possible.",
    ]

    if _is_interaction_query(lowered):
        lines.append(
            "- Interaction focus: capture at least two distinct drug mentions when present."
        )
    if _is_pregnancy_query(lowered):
        lines.append(
            "- Pregnancy focus: infer pregnancy_status and trimester if evidence exists in text."
        )
    if _is_renal_query(lowered):
        lines.append("- Renal focus: infer kidney_status (none/CKD/dialysis) when suggested.")
    if "elderly" in lowered or "over 65" in lowered or "geriatric" in lowered:
        lines.append("- Capture age_group as over 65 when the question implies elderly context.")

    return "\n".join(lines)


def rerank_policy_text(question: str, chunks: list[EvidenceChunk], top_k: int) -> str:
    ctx = build_prompt_context(question, chunks=chunks)
    focus_terms = ", ".join(ctx.query_terms[:12]) if ctx.query_terms else "none"
    source_priority = " > ".join(ctx.source_priority)
    return (
        f"Dynamic policy:\n"
        f"- focus={ctx.focus}\n"
        f"- prioritize_sources={source_priority}\n"
        f"- query_terms={focus_terms}\n"
        f"- candidate_chunks={ctx.evidence_budget}\n"
        f"- return_top_k={top_k}\n"
        "- Prefer chunks that explicitly mention queried drugs and clinically relevant modifiers."
    )


def claim_policy_text(question: str, chunks: list[EvidenceChunk]) -> str:
    ctx = build_prompt_context(question, chunks=chunks)
    focus_terms = ", ".join(ctx.query_terms[:12]) if ctx.query_terms else "none"
    source_priority = " > ".join(ctx.source_priority)
    return (
        f"Dynamic policy:\n"
        f"- focus={ctx.focus}\n"
        f"- source_priority={source_priority}\n"
        f"- max_claims={ctx.max_claims}\n"
        f"- focus_terms={focus_terms}\n"
        "- Prefer concrete safety signals over generic statements.\n"
        "- Do not infer unsupported interactions or outcomes.\n"
        "- If evidence is weak, produce fewer claims and keep uncertainty explicit."
    )


def judge_policy_text(question: str, response: dict[str, Any]) -> str:
    ctx = build_prompt_context(question, chunks=[])
    citation_count = len(response.get("evidence_sources", []) or [])
    followup_count = len(response.get("follow_up_questions", []) or [])
    return (
        f"Dynamic rubric:\n"
        f"- question_focus={ctx.focus}\n"
        f"- expected_source_priority={' > '.join(ctx.source_priority)}\n"
        f"- observed_citation_count={citation_count}\n"
        f"- observed_followup_count={followup_count}\n"
        "- Penalize unsupported safety claims and irrelevant citations.\n"
        "- Reward clear uncertainty communication when evidence is thin."
    )


def _is_interaction_query(lowered_question: str) -> bool:
    tokens = ["interact", "interaction", "with", "together", "coadminister", "combine"]
    return any(token in lowered_question for token in tokens)


def _is_pregnancy_query(lowered_question: str) -> bool:
    tokens = ["pregnan", "trimester", "fetal", "foetal", "lactation", "breastfeeding"]
    return any(token in lowered_question for token in tokens)


def _is_renal_query(lowered_question: str) -> bool:
    tokens = ["renal", "kidney", "ckd", "dialysis", "egfr"]
    return any(token in lowered_question for token in tokens)


def _important_terms(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    stop = {
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
        "can",
        "use",
        "used",
        "safe",
        "safely",
    }
    dedup: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in stop or word in seen:
            continue
        dedup.append(word)
        seen.add(word)
    return dedup[:20]
