from __future__ import annotations

import json
from pathlib import Path

from drug_safety_assistant.retrieval.persistent_index import (
    PersistentCorpusIndexBuilder,
    PersistentHybridRetriever,
)


def test_build_and_search_persistent_index(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "source": "openfda",
            "citation_id": "OPENFDA:WARFARIN",
            "title": "Warfarin Label",
            "text": "Boxed warning mentions serious bleeding risk and interaction with amiodarone.",
            "metadata": {"label_severity": 3},
            "strength_score": 3,
        },
        {
            "source": "pubmed",
            "citation_id": "PMID:123",
            "title": "Warfarin Interaction Study",
            "text": (
                "Observational evidence reports elevated INR when warfarin is "
                "combined with amiodarone."
            ),
            "metadata": {"year": 2019},
            "strength_score": 1,
        },
        {
            "source": "faers",
            "citation_id": "FAERS:WARFARIN:BLEEDING",
            "title": "FAERS Warfarin",
            "text": (
                "FAERS signal indicates bleeding reports linked to warfarin and "
                "interaction terms."
            ),
            "metadata": {"prr": 1.8},
            "strength_score": 2,
        },
    ]

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    index_dir = tmp_path / "index"
    builder = PersistentCorpusIndexBuilder(index_dir=index_dir)
    manifest = builder.build_from_jsonl(corpus_path=corpus_path, batch_size=2)

    assert manifest["chunk_count"] >= 3
    assert PersistentHybridRetriever.is_ready(index_dir)

    retriever = PersistentHybridRetriever(index_dir=index_dir)
    results = retriever.search("warfarin amiodarone interaction", top_k=3)

    assert results
    citation_ids = {item.parent_citation_id for item in results}
    assert "OPENFDA:WARFARIN" in citation_ids
