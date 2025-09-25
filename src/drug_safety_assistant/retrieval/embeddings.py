from __future__ import annotations

import hashlib
import re

import numpy as np

from ..config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency path
    SentenceTransformer = None


class MedCPTEmbedder:
    """MedCPT encoder wrapper with deterministic hash fallback."""

    def __init__(
        self,
        query_model: str | None = None,
        doc_model: str | None = None,
        dim: int = 768,
    ) -> None:
        self.query_model_name = query_model or settings.medcpt_query_model
        self.doc_model_name = doc_model or settings.medcpt_doc_model
        self.fallback_dim = dim
        self._query_encoder = None
        self._doc_encoder = None

    @property
    def dim(self) -> int:
        if self._query_encoder is not None:
            return int(self._query_encoder.get_sentence_embedding_dimension())
        return self.fallback_dim

    def embed_query(self, text: str) -> np.ndarray:
        encoder = self._load_query_encoder()
        if encoder is None:
            return self._fallback_embedding(text)

        vector = np.array(encoder.encode([text], normalize_embeddings=True), dtype=np.float32)[0]
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        encoder = self._load_doc_encoder()
        if encoder is None:
            vectors = [self._fallback_embedding(text) for text in texts]
            return np.vstack(vectors).astype(np.float32)

        return np.array(
            encoder.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )

    def _load_query_encoder(self):
        if self._query_encoder is not None:
            return self._query_encoder

        if not settings.enable_medcpt_models:
            return None

        if SentenceTransformer is None:
            return None

        try:
            self._query_encoder = SentenceTransformer(self.query_model_name)
        except Exception:  # pragma: no cover - runtime dependency failures
            self._query_encoder = None

        return self._query_encoder

    def _load_doc_encoder(self):
        if self._doc_encoder is not None:
            return self._doc_encoder

        if not settings.enable_medcpt_models:
            return None

        if SentenceTransformer is None:
            return None

        try:
            self._doc_encoder = SentenceTransformer(self.doc_model_name)
        except Exception:  # pragma: no cover - runtime dependency failures
            self._doc_encoder = None

        return self._doc_encoder

    def _fallback_embedding(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        vector = np.zeros(self.fallback_dim, dtype=np.float32)

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.fallback_dim
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
