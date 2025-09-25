from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "nvidia_extract_benchmark.json"

DEFAULT_MODELS = [
    "mistralai/mistral-nemotron",
    "moonshotai/kimi-k2-instruct-0905",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
]

CASES = [
    {
        "question": "Does warfarin interact with amiodarone in an elderly patient?",
        "expected": {
            "intent": "interaction",
            "drug_mentions": ["warfarin", "amiodarone"],
        },
    },
    {
        "question": "Is valproate safe in first trimester pregnancy?",
        "expected": {
            "intent": "pregnancy",
            "drug_mentions": ["valproate"],
        },
    },
    {
        "question": "Is metformin safe in CKD stage 3?",
        "expected": {
            "intent": "renal",
            "drug_mentions": ["metformin"],
        },
    },
    {
        "question": "Any serious side effects of isotretinoin?",
        "expected": {
            "intent": "general",
            "drug_mentions": ["isotretinoin"],
        },
    },
    {
        "question": "Can digoxin and clarithromycin be taken together?",
        "expected": {
            "intent": "interaction",
            "drug_mentions": ["digoxin", "clarithromycin"],
        },
    },
]

PROMPT_TEMPLATE = (
    "Extract entities from this clinical drug safety question and respond with JSON only. "
    'JSON schema: {{"intent":"interaction|pregnancy|renal|patient_specific|general", '
    '"drug_mentions":[],"age_group":null,"pregnancy_status":null,'
    '"trimester":null,"kidney_status":null,"liver_status":null}}. '
    "No markdown. No prose.\n"
    "Question: {question}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA models for extraction quality")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model IDs to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout seconds",
    )
    return parser.parse_args()


def call_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    timeout: int,
) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical information extraction model.",
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(question=question),
            },
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}

    choices = data.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}

    content = str(choices[0].get("message", {}).get("content", ""))
    json_blob = extract_json_block(content)
    if not json_blob:
        return {
            "valid_json": False,
            "raw": content,
        }

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return {
            "valid_json": False,
            "raw": content,
        }

    return {
        "valid_json": isinstance(parsed, dict),
        "parsed": parsed if isinstance(parsed, dict) else None,
        "raw": content,
    }


def extract_json_block(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def evaluate_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
) -> dict:
    total = len(CASES)
    valid_json_count = 0
    intent_hits = 0
    drug_hits = 0
    failures = 0
    case_outputs = []

    for case in CASES:
        question = case["question"]
        expected = case["expected"]
        output = call_model(
            api_key=api_key,
            base_url=base_url,
            model=model,
            question=question,
            timeout=timeout,
        )

        if "error" in output:
            failures += 1
            case_outputs.append({
                "question": question,
                "error": output["error"],
            })
            continue

        if output.get("valid_json"):
            valid_json_count += 1
            parsed = output.get("parsed") or {}

            if parsed.get("intent") == expected.get("intent"):
                intent_hits += 1

            got_drugs = {
                str(item).lower().strip()
                for item in parsed.get("drug_mentions", [])
                if str(item).strip()
            }
            expected_drugs = {item.lower() for item in expected.get("drug_mentions", [])}
            if expected_drugs.issubset(got_drugs):
                drug_hits += 1

            case_outputs.append({
                "question": question,
                "parsed": parsed,
            })
        else:
            case_outputs.append({
                "question": question,
                "raw": output.get("raw", "")[:400],
            })

    return {
        "model": model,
        "total_cases": total,
        "failures": failures,
        "valid_json_rate": round(valid_json_count / total, 4),
        "intent_accuracy": round(intent_hits / total, 4),
        "drug_mention_recall": round(drug_hits / total, 4),
        "details": case_outputs,
    }


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("NVIDIA_API_KEY", "")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required in .env or environment")

    results = []
    for model in args.models:
        results.append(
            evaluate_model(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=args.timeout,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
