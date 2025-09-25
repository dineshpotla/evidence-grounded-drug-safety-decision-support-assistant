from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_safety_assistant.retrieval.faers import FAERSRetriever
from drug_safety_assistant.retrieval.openfda import OpenFDARetriever
from drug_safety_assistant.retrieval.pubmed import PubMedRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus JSONL from a drug list")
    parser.add_argument(
        "--drugs",
        type=Path,
        required=True,
        help="Text file with one drug name per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output corpus JSONL path",
    )
    parser.add_argument(
        "--pubmed-per-drug",
        type=int,
        default=25,
        help="Number of PubMed entries per drug",
    )
    parser.add_argument(
        "--label-per-drug",
        type=int,
        default=12,
        help="Number of OpenFDA label entries per drug",
    )
    return parser.parse_args()


def load_drugs(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            values.append(token)
    return values


def main() -> None:
    args = parse_args()
    drugs = load_drugs(args.drugs)

    openfda = OpenFDARetriever()
    pubmed = PubMedRetriever()
    faers = FAERSRetriever()

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for index, drug in enumerate(drugs, start=1):
        label_items = openfda.search_labels(drug_name=drug, limit=args.label_per_drug)
        pubmed_items = pubmed.search(
            query=f"{drug} safety adverse reactions contraindications",
            max_results=args.pubmed_per_drug,
            start_year=2010,
        )
        faers_items = faers.fetch_signal(drug_name=drug)

        for item in [*label_items, *pubmed_items, *faers_items]:
            key = (item.source, item.citation_id, item.snippet)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "source": item.source,
                    "citation_id": item.citation_id,
                    "title": item.title,
                    "text": item.snippet,
                    "metadata": item.metadata,
                    "strength_score": item.strength_score,
                }
            )

        if index % 25 == 0 or index == len(drugs):
            print(f"Processed drugs: {index}/{len(drugs)} | rows={len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote corpus rows: {len(rows)} to {args.output}")


if __name__ == "__main__":
    main()
