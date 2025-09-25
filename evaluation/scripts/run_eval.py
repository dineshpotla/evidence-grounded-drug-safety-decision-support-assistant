from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from drug_safety_assistant.config import settings
from drug_safety_assistant.llm.judge import LLMAnswerJudge
from drug_safety_assistant.pipeline.orchestrator import DrugSafetyAssistant
from drug_safety_assistant.types import SafetyRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = PROJECT_ROOT / "evaluation" / "benchmarks" / "sample_benchmark.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def required_prefix_hits(required: list[str], evidence_ids: list[str]) -> tuple[int, int]:
    if not required:
        return (0, 0)

    hits = 0
    for prefix in required:
        if any(eid.startswith(prefix) for eid in evidence_ids):
            hits += 1

    return hits, len(required)


def _request_payload(case: dict) -> dict:
    slots = case.get("slots", {})
    return {
        "question": case["question"],
        "drug": slots.get("drug"),
        "drug_a": slots.get("drug_a"),
        "drug_b": slots.get("drug_b"),
        "age_group": slots.get("age_group"),
        "pregnancy_status": slots.get("pregnancy_status"),
        "trimester": slots.get("trimester"),
        "kidney_status": slots.get("kidney_status"),
        "liver_status": slots.get("liver_status"),
        "current_meds": slots.get("current_meds", []),
    }


def _request_cache_key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)


def _assess_payload(payload: dict) -> dict:
    assistant = DrugSafetyAssistant()
    request = SafetyRequest(**payload)
    return assistant.assess(request).model_dump()


def _judge_payload(question: str, response: dict, use_llm_judge: bool) -> dict:
    judge = LLMAnswerJudge(enabled=use_llm_judge)
    return judge.evaluate(question=question, response=response).to_dict()


def _normalize_risk_level(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))

    text = str(value)
    if text.startswith("RiskLevel."):
        raw = text.split(".", 1)[1]
        lowered = raw.lower()
        if lowered == "low":
            return "Low"
        if lowered == "moderate":
            return "Moderate"
        if lowered == "high":
            return "High"
    return text


def evaluate(
    cases: list[dict],
    workers: int = 6,
    use_llm_judge: bool = False,
    judge_workers: int = 4,
) -> dict:
    total = len(cases)

    # Deduplicate requests first so large synthetic benchmarks can be executed quickly.
    unique_payloads: dict[str, dict] = {}
    for case in cases:
        payload = _request_payload(case)
        key = _request_cache_key(payload)
        if key not in unique_payloads:
            unique_payloads[key] = payload

    response_cache: dict[str, dict] = {}
    unique_total = len(unique_payloads)
    print(f"Unique request signatures: {unique_total}/{total}")

    if unique_total == 0:
        return {"metrics": {}, "cases": []}

    max_workers = max(1, min(workers, unique_total))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_assess_payload, payload): key
            for key, payload in unique_payloads.items()
        }

        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            response_cache[key] = future.result()
            completed += 1
            if completed % 5 == 0 or completed == unique_total:
                print(f"Assessed unique requests: {completed}/{unique_total}")

    judge_cache: dict[str, dict] = {}
    max_judge_workers = max(1, min(judge_workers, unique_total))
    with ThreadPoolExecutor(max_workers=max_judge_workers) as executor:
        futures = {
            executor.submit(
                _judge_payload,
                unique_payloads[key]["question"],
                response_cache[key],
                use_llm_judge,
            ): key
            for key in unique_payloads
        }

        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            judge_cache[key] = future.result()
            completed += 1
            if completed % 5 == 0 or completed == unique_total:
                print(f"Judged unique answers: {completed}/{unique_total}")

    risk_agreements = 0
    hallucination_flags = 0
    helpful_count = 0
    concise_count = 0
    required_hits = 0
    required_total = 0
    support_sum = 0.0
    citation_coverage_hits = 0
    followup_count = 0
    evidence_count_sum = 0
    judge_supported_sum = 0.0
    judge_citation_sum = 0.0
    judge_helpfulness_sum = 0.0
    judge_conciseness_sum = 0.0
    judge_overall_sum = 0.0
    judge_hallucination_flags = 0

    per_case: list[dict] = []

    for index, case in enumerate(cases, start=1):
        payload = _request_payload(case)
        key = _request_cache_key(payload)
        response_dict = response_cache[key]

        evidence_ids = [
            citation["citation_id"] for citation in response_dict.get("evidence_sources", [])
        ]
        risk_level = _normalize_risk_level(response_dict.get("risk_level", "Low"))
        risk_score = float(response_dict.get("risk_score", 0.0))
        supported_ratio = float(response_dict.get("guard_supported_ratio", 1.0))
        summary = str(response_dict.get("safety_summary", ""))
        monitoring = list(response_dict.get("monitoring_recommendations", []))
        followups = list(response_dict.get("follow_up_questions", []))
        judge_result = judge_cache.get(key, {})

        if risk_level == case["expected_risk_level"]:
            risk_agreements += 1

        if supported_ratio < 1.0:
            hallucination_flags += 1

        support_sum += supported_ratio

        if summary and monitoring:
            helpful_count += 1

        if len(summary.split()) <= 120:
            concise_count += 1

        if followups:
            followup_count += 1

        evidence_count_sum += len(evidence_ids)
        if evidence_ids:
            citation_coverage_hits += 1

        judge_supported_sum += float(judge_result.get("supported_claims_score", 0.0))
        judge_citation_sum += float(judge_result.get("citation_quality_score", 0.0))
        judge_helpfulness_sum += float(judge_result.get("clinical_helpfulness_score", 0.0))
        judge_conciseness_sum += float(judge_result.get("conciseness_score", 0.0))
        judge_overall_sum += float(judge_result.get("overall_score", 0.0))
        if bool(judge_result.get("hallucination_detected", False)):
            judge_hallucination_flags += 1

        hit, denom = required_prefix_hits(case.get("required_citations", []), evidence_ids)
        required_hits += hit
        required_total += denom

        per_case.append(
            {
                "id": case["id"],
                "question": case["question"],
                "risk_pred": risk_level,
                "risk_expected": case["expected_risk_level"],
                "risk_score": risk_score,
                "guard_supported_ratio": supported_ratio,
                "citation_ids": evidence_ids,
                "safety_summary": summary,
                "monitoring_recommendations": monitoring,
                "uncertainty_statement": str(
                    response_dict.get("uncertainty_statement", "")
                ),
                "follow_up_questions": followups,
                "judge": judge_result,
            }
        )

        if index % 500 == 0 or index == total:
            print(f"Scored cases: {index}/{total}")

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_cases": total,
        "unique_requests": unique_total,
        "retrieval_recall_at_k": round(required_hits / required_total, 4)
        if required_total
        else 0.0,
        "risk_agreement": round(risk_agreements / total, 4) if total else 0.0,
        "hallucination_rate": round(hallucination_flags / total, 4) if total else 0.0,
        "citation_precision": round(citation_coverage_hits / total, 4) if total else 0.0,
        "supported_claim_rate": round(support_sum / total, 4) if total else 0.0,
        "helpfulness_score": round(helpful_count / total, 4) if total else 0.0,
        "conciseness_score": round(concise_count / total, 4) if total else 0.0,
        "followup_rate": round(followup_count / total, 4) if total else 0.0,
        "avg_evidence_count": round(evidence_count_sum / total, 4) if total else 0.0,
        "llm_judge_enabled": use_llm_judge,
        "judge_supported_claims_score": round(judge_supported_sum / total, 4) if total else 0.0,
        "judge_citation_quality_score": round(judge_citation_sum / total, 4) if total else 0.0,
        "judge_clinical_helpfulness_score": round(judge_helpfulness_sum / total, 4)
        if total
        else 0.0,
        "judge_conciseness_score": round(judge_conciseness_sum / total, 4) if total else 0.0,
        "judge_overall_score": round(judge_overall_sum / total, 4) if total else 0.0,
        "judge_hallucination_rate": round(judge_hallucination_flags / total, 4) if total else 0.0,
    }

    return {"metrics": metrics, "cases": per_case}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drug safety assistant evaluation")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to timestamped file in evaluation/results",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of cases evaluated (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Thread workers for unique request assessments",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=4,
        help="Thread workers for LLM-as-judge scoring",
    )
    parser.add_argument(
        "--llm-judge",
        dest="llm_judge",
        action="store_true",
        help="Enable LLM-as-judge using NVIDIA model (falls back to heuristic if unavailable)",
    )
    parser.add_argument(
        "--no-llm-judge",
        dest="llm_judge",
        action="store_false",
        help="Disable LLM-as-judge and use heuristic judge only",
    )
    parser.set_defaults(llm_judge=settings.enable_llm_judge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.benchmark)

    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    results = evaluate(
        cases,
        workers=args.workers,
        use_llm_judge=args.llm_judge,
        judge_workers=args.judge_workers,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        output_file = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_file = args.output

    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["metrics"], indent=2))
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
