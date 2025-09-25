from drug_safety_assistant.llm.judge import (
    LLMAnswerJudge,
    LLMJudgeResult,
    _parse_json_object,
    _result_from_dict,
)


def test_llm_judge_fallback_scores_are_stable() -> None:
    judge = LLMAnswerJudge(enabled=False, api_key="")
    response = {
        "safety_summary": "Evidence suggests interaction risk requiring monitoring.",
        "monitoring_recommendations": ["Monitor adverse effects closely."],
        "evidence_sources": [{"citation_id": "OPENFDA:1"}, {"citation_id": "PMID:1"}],
        "guard_supported_ratio": 0.8,
    }

    result = judge.evaluate(question="test", response=response)

    assert result.supported_claims_score == 0.8
    assert result.citation_quality_score == 1.0
    assert result.clinical_helpfulness_score == 1.0
    assert result.conciseness_score == 1.0
    assert result.overall_score == 0.92
    assert result.hallucination_detected is True


def test_result_from_dict_clamps_scores() -> None:
    parsed = _parse_json_object(
        'notes {"supported_claims_score": 1.4, "citation_quality_score": -1, '
        '"clinical_helpfulness_score": 0.7, "conciseness_score": 0.4, '
        '"overall_score": 5, "hallucination_detected": 1, "notes": "ok"}'
    )

    assert parsed is not None

    result = _result_from_dict(parsed)
    assert result.supported_claims_score == 1.0
    assert result.citation_quality_score == 0.0
    assert result.clinical_helpfulness_score == 0.7
    assert result.conciseness_score == 0.4
    assert result.overall_score == 1.0
    assert result.hallucination_detected is True
    assert result.notes == "ok"


def test_llm_judge_uses_model_result_when_enabled(monkeypatch) -> None:
    judge = LLMAnswerJudge(enabled=True, api_key="fake-key")
    expected = LLMJudgeResult(
        supported_claims_score=0.9,
        citation_quality_score=0.8,
        clinical_helpfulness_score=0.85,
        conciseness_score=0.95,
        overall_score=0.88,
        hallucination_detected=False,
        notes="model verdict",
    )

    monkeypatch.setattr(judge, "_evaluate_with_nvidia", lambda question, response: expected)

    result = judge.evaluate(question="q", response={})
    assert result == expected
