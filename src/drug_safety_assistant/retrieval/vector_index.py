from __future__ import annotations

import numpy as np

from ..config import settings
from ..types import EvidenceChunk

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency path
    faiss = None


class VectorIndex:
    def __init__(self, use_faiss: bool | None = None) -> None:
        self.use_faiss = settings.use_faiss if use_faiss is None else use_faiss
        self._chunks: list[EvidenceChunk] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None

    def add(self, chunks: list[EvidenceChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            self._chunks = []
            self._matrix = np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0))
            self._faiss_index = None
            return

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk and embedding counts do not match")

        self._chunks = chunks
        self._matrix = embeddings.astype(np.float32)

        if self.use_faiss and faiss is not None:
            dim = int(self._matrix.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(self._matrix)
            self._faiss_index = index
        else:
            self._faiss_index = None

    def search(self, query_vector: np.ndarray, top_k: int) -> list[EvidenceChunk]:
        if not self._chunks:
            return []

        top_k = max(1, min(top_k, len(self._chunks)))
        query = query_vector.astype(np.float32).reshape(1, -1)

        if self._faiss_index is not None:
            _, idx = self._faiss_index.search(query, top_k)
            return [self._chunks[i] for i in idx[0].tolist() if i >= 0]

        assert self._matrix is not None
        scores = np.dot(self._matrix, query_vector.astype(np.float32))
        order = np.argsort(scores)[::-1][:top_k]
        return [self._chunks[int(i)] for i in order]
