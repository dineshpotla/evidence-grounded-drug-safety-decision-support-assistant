from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "benchmarks" / "synthetic_3000.jsonl"

INTERACTION_HIGH = [
    ("warfarin", "amiodarone"),
    ("digoxin", "clarithromycin"),
    ("simvastatin", "clarithromycin"),
    ("linezolid", "sertraline"),
    ("clopidogrel", "omeprazole"),
]
INTERACTION_MODERATE = [
    ("metformin", "cimetidine"),
    ("lisinopril", "ibuprofen"),
    ("furosemide", "ibuprofen"),
    ("losartan", "spironolactone"),
]
PREGNANCY_HIGH = ["valproate", "isotretinoin", "warfarin"]
RENAL_MODERATE = ["metformin", "gabapentin", "allopurinol"]
GENERAL_MODERATE = ["isotretinoin", "amiodarone", "clozapine", "fluoroquinolone"]
GENERAL_LOW = ["loratadine", "acetaminophen", "saline spray"]


def build_interaction_case(idx: int, high: bool) -> dict:
    if high:
        drug_a, drug_b = random.choice(INTERACTION_HIGH)
        expected = "High"
    else:
        drug_a, drug_b = random.choice(INTERACTION_MODERATE)
        expected = "Moderate"

    age_group = random.choice(["adult", "over 65"])
    question = f"Does {drug_a} interact with {drug_b} in an {age_group} patient?"

    return {
        "id": f"syn_{idx:05d}",
        "question": question,
        "intent": "interaction",
        "expected_risk_level": expected,
        "slots": {
            "drug": None,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "age_group": age_group,
            "pregnancy_status": "no",
            "trimester": None,
            "kidney_status": "none",
            "liver_status": "none",
            "current_meds": [],
        },
        "required_citations": ["OPENFDA", "PMID", "FAERS"],
    }


def build_pregnancy_case(idx: int) -> dict:
    drug = random.choice(PREGNANCY_HIGH)
    trimester = random.choice(["1", "2", "3"])
    question = f"Is {drug} safe in trimester {trimester} pregnancy?"

    return {
        "id": f"syn_{idx:05d}",
        "question": question,
        "intent": "pregnancy",
        "expected_risk_level": "High",
        "slots": {
            "drug": drug,
            "drug_a": None,
            "drug_b": None,
            "age_group": "adult",
            "pregnancy_status": "yes",
            "trimester": trimester,
            "kidney_status": "none",
            "liver_status": "none",
            "current_meds": [],
        },
        "required_citations": ["OPENFDA", "PMID"],
    }


def build_renal_case(idx: int) -> dict:
    drug = random.choice(RENAL_MODERATE)
    kidney_status = random.choice(["CKD", "dialysis"])
    question = f"How safe is {drug} for a patient with {kidney_status}?"

    return {
        "id": f"syn_{idx:05d}",
        "question": question,
        "intent": "renal",
        "expected_risk_level": "Moderate",
        "slots": {
            "drug": drug,
            "drug_a": None,
            "drug_b": None,
            "age_group": random.choice(["adult", "over 65"]),
            "pregnancy_status": "no",
            "trimester": None,
            "kidney_status": kidney_status,
            "liver_status": "none",
            "current_meds": [],
        },
        "required_citations": ["OPENFDA", "FAERS"],
    }


def build_general_case(idx: int, moderate: bool) -> dict:
    if moderate:
        drug = random.choice(GENERAL_MODERATE)
        expected = "Moderate"
    else:
        drug = random.choice(GENERAL_LOW)
        expected = "Low"

    question = f"Any serious safety concerns with {drug}?"

    return {
        "id": f"syn_{idx:05d}",
        "question": question,
        "intent": "general",
        "expected_risk_level": expected,
        "slots": {
            "drug": drug,
            "drug_a": None,
            "drug_b": None,
            "age_group": "adult",
            "pregnancy_status": "no",
            "trimester": None,
            "kidney_status": "none",
            "liver_status": "none",
            "current_meds": [],
        },
        "required_citations": ["OPENFDA", "PMID"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark JSONL")
    parser.add_argument("--num-queries", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    makers = [
        lambda idx: build_interaction_case(idx, high=True),
        lambda idx: build_interaction_case(idx, high=False),
        build_pregnancy_case,
        build_renal_case,
        lambda idx: build_general_case(idx, moderate=True),
        lambda idx: build_general_case(idx, moderate=False),
    ]

    rows = []
    for idx in range(1, args.num_queries + 1):
        maker = random.choice(makers)
        rows.append(maker(idx))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
