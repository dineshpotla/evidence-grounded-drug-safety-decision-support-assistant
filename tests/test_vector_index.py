import numpy as np

from drug_safety_assistant.retrieval.vector_index import VectorIndex
from drug_safety_assistant.types import EvidenceChunk


def test_vector_index_search_returns_best_match() -> None:
    chunks = [
        EvidenceChunk(
            chunk_id="c1", source="pubmed", parent_citation_id="P1", text="heart failure"
        ),
        EvidenceChunk(
            chunk_id="c2", source="pubmed", parent_citation_id="P2", text="kidney disease"
        ),
        EvidenceChunk(
            chunk_id="c3", source="pubmed", parent_citation_id="P3", text="drug interaction"
        ),
    ]
    vectors = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    index = VectorIndex(use_faiss=False)
    index.add(chunks=chunks, embeddings=vectors)
    top = index.search(query_vector=query, top_k=2)

    assert top[0].chunk_id == "c2"
