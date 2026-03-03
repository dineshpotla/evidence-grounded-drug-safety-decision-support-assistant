from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from ..config import settings
from .dynamic_prompting import judge_policy_text


@dataclass(frozen=True)
class LLMJudgeResult:
    supported_claims_score: float
    citation_quality_score: float
    clinical_helpfulness_score: float
    conciseness_score: float
    overall_score: float
    hallucination_detected: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_claims_score": self.supported_claims_score,
            "citation_quality_score": self.citation_quality_score,
            "clinical_helpfulness_score": self.clinical_helpfulness_score,
            "conciseness_score": self.conciseness_score,
            "overall_score": self.overall_score,
            "hallucination_detected": self.hallucination_detected,
            "notes": self.notes,
        }


class LLMAnswerJudge:
    """LLM-as-judge for tool outputs with deterministic fallback."""

    def __init__(
        self,
        enabled: bool = False,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.model = model or settings.nvidia_judge_model
        self.base_url = settings.nvidia_base_url.rstrip("/")
        self.session = requests.Session()

    def evaluate(self, question: str, response: dict[str, Any]) -> LLMJudgeResult:
        if self.enabled and self.api_key:
            judged = self._evaluate_with_nvidia(question=question, response=response)
            if judged is not None:
                return judged
        return self._heuristic_fallback(response)

    def _evaluate_with_nvidia(
        self,
        question: str,
        response: dict[str, Any],
    ) -> LLMJudgeResult | None:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 260,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator for drug safety assistant outputs. "
                        "Return only valid JSON and score conservatively."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(question=question, response=response),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            api_response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=settings.llm_judge_timeout_seconds,
            )
            api_response.raise_for_status()
            data = api_response.json()
        except (requests.RequestException, ValueError):
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        text = str(message.get("content", "")).strip()
        parsed = _parse_json_object(text)
        if parsed is None:
            return None

        return _result_from_dict(parsed)

    def _build_prompt(self, question: str, response: dict[str, Any]) -> str:
        compact = {
            "risk_level": response.get("risk_level"),
            "risk_score": response.get("risk_score"),
            "safety_summary": response.get("safety_summary"),
            "monitoring_recommendations": response.get("monitoring_recommendations", []),
            "uncertainty_statement": response.get("uncertainty_statement"),
            "guard_supported_ratio": response.get("guard_supported_ratio", 1.0),
            "evidence_sources": response.get("evidence_sources", []),
            "follow_up_questions": response.get("follow_up_questions", []),
        }

        return (
            "Evaluate this assistant answer for clinical decision-support quality.\n"
            "Score each field from 0.0 to 1.0.\n"
            "Output JSON only with schema:\n"
            "{"
            '"supported_claims_score": float,'
            '"citation_quality_score": float,'
            '"clinical_helpfulness_score": float,'
            '"conciseness_score": float,'
            '"overall_score": float,'
            '"hallucination_detected": bool,'
            '"notes": string'
            "}\n"
            "Guidance:\n"
            "- Lower supported_claims_score if answer seems weakly grounded.\n"
            "- Lower citation_quality_score if citations are missing or irrelevant.\n"
            "- Lower clinical_helpfulness_score if monitoring advice is weak.\n"
            "- Lower conciseness_score if verbose, repetitive, or unclear.\n"
            "- Set hallucination_detected=true if likely unsupported claims appear.\n"
            f"{judge_policy_text(question=question, response=response)}\n"
            f"Question: {question}\n"
            f"Answer JSON: {json.dumps(compact)}"
        )

    def _heuristic_fallback(self, response: dict[str, Any]) -> LLMJudgeResult:
        summary = str(response.get("safety_summary", "")).strip()
        monitoring = response.get("monitoring_recommendations", [])
        evidence = response.get("evidence_sources", [])

        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        summary_words = len(summary.split())
        guard_ratio = _clamp_0_1(response.get("guard_supported_ratio", 1.0))

        citation_quality = 1.0 if evidence_count >= 2 else (0.6 if evidence_count == 1 else 0.2)
        helpfulness = 1.0 if summary and monitoring else (0.5 if summary else 0.1)
        conciseness = 1.0 if 0 < summary_words <= 120 else 0.6 if summary_words <= 170 else 0.3
        hallucination = guard_ratio < 0.95

        overall = (
            0.4 * guard_ratio
            + 0.25 * citation_quality
            + 0.2 * helpfulness
            + 0.15 * conciseness
        )
        notes = (
            "fallback heuristic"
            if summary
            else "fallback heuristic: empty summary reduces helpfulness"
        )

        return LLMJudgeResult(
            supported_claims_score=round(guard_ratio, 4),
            citation_quality_score=round(citation_quality, 4),
            clinical_helpfulness_score=round(helpfulness, 4),
            conciseness_score=round(conciseness, 4),
            overall_score=round(_clamp_0_1(overall), 4),
            hallucination_detected=hallucination,
            notes=notes,
        )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _result_from_dict(values: dict[str, Any]) -> LLMJudgeResult:
    return LLMJudgeResult(
        supported_claims_score=round(_clamp_0_1(values.get("supported_claims_score", 0.0)), 4),
        citation_quality_score=round(_clamp_0_1(values.get("citation_quality_score", 0.0)), 4),
        clinical_helpfulness_score=round(
            _clamp_0_1(values.get("clinical_helpfulness_score", 0.0)), 4
        ),
        conciseness_score=round(_clamp_0_1(values.get("conciseness_score", 0.0)), 4),
        overall_score=round(_clamp_0_1(values.get("overall_score", 0.0)), 4),
        hallucination_detected=bool(values.get("hallucination_detected", False)),
        notes=str(values.get("notes", "")).strip()[:240],
    )


def _clamp_0_1(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number
