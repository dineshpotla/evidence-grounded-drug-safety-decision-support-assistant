from drug_safety_assistant.llm.dynamic_prompting import (
    build_prompt_context,
    claim_policy_text,
    extraction_directives,
    judge_policy_text,
    rerank_policy_text,
)
from drug_safety_assistant.types import EvidenceChunk


def test_build_prompt_context_interaction_focus() -> None:
    ctx = build_prompt_context("Does warfarin interact with amiodarone?")
    assert ctx.focus == "interaction"
    assert ctx.source_priority[0] == "openfda"
    assert "warfarin" in ctx.query_terms


def test_build_prompt_context_pregnancy_focus() -> None:
    ctx = build_prompt_context("Is valproate safe in first trimester pregnancy?")
    assert ctx.focus == "pregnancy"
    assert ctx.max_claims == 3


def test_extraction_directives_include_context_specific_rules() -> None:
    directives = extraction_directives(
        "Can metformin be used in CKD and pregnancy in elderly patients?"
    )
    assert "Pregnancy focus" in directives
    assert "Renal focus" in directives
    assert "over 65" in directives


def test_rerank_and_claim_policies_include_dynamic_fields() -> None:
    chunks = [
        EvidenceChunk(
            chunk_id="a",
            source="openfda",
            parent_citation_id="OPENFDA:1",
            text="Interaction warning",
            metadata={},
        )
    ]
    rerank = rerank_policy_text("Does drug A interact with drug B?", chunks, 3)
    claims = claim_policy_text("Does drug A interact with drug B?", chunks)

    assert "focus=interaction" in rerank
    assert "return_top_k=3" in rerank
    assert "max_claims" in claims


def test_judge_policy_summarizes_observed_response_shape() -> None:
    policy = judge_policy_text(
        "Is metformin safe in CKD?",
        {
            "evidence_sources": [{"citation_id": "PMID:1"}, {"citation_id": "OPENFDA:1"}],
            "follow_up_questions": [],
        },
    )
    assert "question_focus=renal" in policy
    assert "observed_citation_count=2" in policy
