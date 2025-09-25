from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import settings
from ..types import EvidenceChunk, RetrievedEvidence
from .corpus import EvidenceCorpusBuilder
from .embeddings import MedCPTEmbedder

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{3,}")


@dataclass(frozen=True)
class IndexPaths:
    index_dir: Path
    manifest: Path
    chunks_jsonl: Path
    vectors_npy: Path
    sqlite_db: Path


class PersistentCorpusIndexBuilder:
    """Builds a production-style persistent corpus index.

    Artifacts:
    - `chunks.jsonl`: chunk metadata and text
    - `vectors.npy`: dense embeddings aligned to chunk IDs
    - `chunks.db`: SQLite + FTS5 lexical index
    - `manifest.json`: artifact metadata
    """

    def __init__(
        self,
        index_dir: str | Path,
        embedder: MedCPTEmbedder | None = None,
        corpus_builder: EvidenceCorpusBuilder | None = None,
    ) -> None:
        self.paths = IndexPaths(
            index_dir=Path(index_dir),
            manifest=Path(index_dir) / "manifest.json",
            chunks_jsonl=Path(index_dir) / "chunks.jsonl",
            vectors_npy=Path(index_dir) / "vectors.npy",
            sqlite_db=Path(index_dir) / "chunks.db",
        )
        self.embedder = embedder or MedCPTEmbedder()
        self.corpus_builder = corpus_builder or EvidenceCorpusBuilder()

    def build_from_jsonl(self, corpus_path: str | Path, batch_size: int = 128) -> dict:
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        self.paths.index_dir.mkdir(parents=True, exist_ok=True)

        chunk_count = self._write_chunk_file(corpus_file)
        if chunk_count == 0:
            raise ValueError("No chunks were generated from corpus")

        self._embed_chunk_file(chunk_count=chunk_count, batch_size=batch_size)
        self._build_sqlite_fts(chunk_count=chunk_count)

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "vector_dim": int(self.embedder.dim),
            "batch_size": int(batch_size),
            "index_dir": str(self.paths.index_dir),
            "corpus_path": str(corpus_file),
            "use_medcpt_models": settings.enable_medcpt_models,
        }
        self.paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _write_chunk_file(self, corpus_file: Path) -> int:
        chunk_count = 0

        with (
            corpus_file.open("r", encoding="utf-8") as src,
            self.paths.chunks_jsonl.open("w", encoding="utf-8") as out,
        ):
            for line in src:
                row_text = line.strip()
                if not row_text:
                    continue

                row = json.loads(row_text)
                evidence = RetrievedEvidence(
                    source=str(row.get("source", "unknown")),
                    citation_id=str(row.get("citation_id", f"doc-{chunk_count}")),
                    title=str(row.get("title", "Untitled")),
                    snippet=str(row.get("text", row.get("snippet", ""))),
                    metadata=dict(row.get("metadata", {})),
                    strength_score=int(row.get("strength_score", 0)),
                )

                chunks = self.corpus_builder.build_chunks([evidence])
                for chunk in chunks:
                    record = {
                        "id": chunk_count,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "parent_citation_id": chunk.parent_citation_id,
                        "title": evidence.title,
                        "text": chunk.text,
                        "strength_score": evidence.strength_score,
                        "metadata": chunk.metadata,
                    }
                    out.write(json.dumps(record) + "\n")
                    chunk_count += 1

        return chunk_count

    def _embed_chunk_file(self, chunk_count: int, batch_size: int) -> None:
        vectors = np.lib.format.open_memmap(
            self.paths.vectors_npy,
            mode="w+",
            dtype=np.float32,
            shape=(chunk_count, self.embedder.dim),
        )

        buffer_texts: list[str] = []
        buffer_ids: list[int] = []

        with self.paths.chunks_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                buffer_ids.append(int(row["id"]))
                buffer_texts.append(str(row["text"]))

                if len(buffer_texts) >= batch_size:
                    self._flush_embedding_batch(vectors, buffer_ids, buffer_texts)
                    buffer_ids.clear()
                    buffer_texts.clear()

        if buffer_texts:
            self._flush_embedding_batch(vectors, buffer_ids, buffer_texts)

        vectors.flush()

    def _flush_embedding_batch(
        self,
        vectors: np.ndarray,
        chunk_ids: list[int],
        texts: list[str],
    ) -> None:
        embedded = self.embedder.embed_documents(texts)
        for idx, chunk_id in enumerate(chunk_ids):
            vectors[chunk_id, :] = embedded[idx]

    def _build_sqlite_fts(self, chunk_count: int) -> None:
        if self.paths.sqlite_db.exists():
            self.paths.sqlite_db.unlink()

        conn = sqlite3.connect(self.paths.sqlite_db)
        try:
            conn.execute(
                """
                CREATE TABLE chunks (
                  id INTEGER PRIMARY KEY,
                  chunk_id TEXT NOT NULL,
                  source TEXT NOT NULL,
                  parent_citation_id TEXT NOT NULL,
                  title TEXT NOT NULL,
                  text TEXT NOT NULL,
                  strength_score INTEGER NOT NULL,
                  metadata_json TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                  title,
                  text,
                  source,
                  content='chunks',
                  content_rowid='id'
                )
                """
            )

            with self.paths.chunks_jsonl.open("r", encoding="utf-8") as handle:
                rows = []
                for line in handle:
                    row = json.loads(line)
                    rows.append(
                        (
                            int(row["id"]),
                            str(row["chunk_id"]),
                            str(row["source"]),
                            str(row["parent_citation_id"]),
                            str(row["title"]),
                            str(row["text"]),
                            int(row.get("strength_score", 0)),
                            json.dumps(row.get("metadata", {})),
                        )
                    )

                conn.executemany(
                    """
                    INSERT INTO chunks (
                      id,
                      chunk_id,
                      source,
                      parent_citation_id,
                      title,
                      text,
                      strength_score,
                      metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.execute("CREATE INDEX idx_chunks_source ON chunks(source)")
            conn.execute("CREATE INDEX idx_chunks_parent ON chunks(parent_citation_id)")
            conn.commit()
        finally:
            conn.close()

        if chunk_count <= 0:
            raise ValueError("Index build produced no chunks")


class PersistentHybridRetriever:
    """Hybrid lexical+dense retrieval over a prebuilt persistent index."""

    def __init__(
        self,
        index_dir: str | Path,
        embedder: MedCPTEmbedder | None = None,
        lexical_k: int | None = None,
        dense_weight: float | None = None,
    ) -> None:
        self.paths = IndexPaths(
            index_dir=Path(index_dir),
            manifest=Path(index_dir) / "manifest.json",
            chunks_jsonl=Path(index_dir) / "chunks.jsonl",
            vectors_npy=Path(index_dir) / "vectors.npy",
            sqlite_db=Path(index_dir) / "chunks.db",
        )
        self.embedder = embedder or MedCPTEmbedder()
        self.lexical_k = lexical_k if lexical_k is not None else settings.lexical_candidate_k
        self.dense_weight = (
            dense_weight if dense_weight is not None else settings.hybrid_dense_weight
        )

        self._manifest = self._load_manifest()
        self._vectors = np.load(self.paths.vectors_npy, mmap_mode="r")

    @classmethod
    def is_ready(cls, index_dir: str | Path) -> bool:
        base = Path(index_dir)
        required = [base / "manifest.json", base / "vectors.npy", base / "chunks.db"]
        return all(path.exists() for path in required)

    def search(self, query_text: str, top_k: int) -> list[EvidenceChunk]:
        lexical = self._lexical_candidates(query_text, self.lexical_k)
        if not lexical:
            return []

        candidate_ids = [row["id"] for row in lexical]
        lexical_scores = np.array([row["lexical_score"] for row in lexical], dtype=np.float32)
        dense_scores = self._dense_scores(query_text=query_text, chunk_ids=candidate_ids)

        lexical_norm = _normalize_scores(lexical_scores)
        dense_norm = _normalize_scores(dense_scores)

        combined = (self.dense_weight * dense_norm) + ((1.0 - self.dense_weight) * lexical_norm)
        order = np.argsort(combined)[::-1][: max(1, min(top_k, len(candidate_ids)))]

        selected_ids = [candidate_ids[int(i)] for i in order]
        rows = self._fetch_chunk_rows(selected_ids)

        by_id = {row["id"]: row for row in rows}
        ranked_chunks: list[EvidenceChunk] = []
        for rank, idx in enumerate(selected_ids, start=1):
            row = by_id.get(idx)
            if row is None:
                continue

            metadata = row["metadata"]
            metadata["rank"] = rank
            metadata["title"] = row["title"]
            metadata["strength_score"] = row["strength_score"]

            ranked_chunks.append(
                EvidenceChunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    parent_citation_id=row["parent_citation_id"],
                    text=row["text"],
                    metadata=metadata,
                )
            )

        return ranked_chunks

    def _load_manifest(self) -> dict:
        if not self.paths.manifest.exists():
            raise FileNotFoundError(f"Missing index manifest: {self.paths.manifest}")
        return json.loads(self.paths.manifest.read_text(encoding="utf-8"))

    def _lexical_candidates(self, query_text: str, top_k: int) -> list[dict]:
        terms = [token.lower() for token in TOKEN_PATTERN.findall(query_text)]
        if not terms:
            return []

        match_query = " OR ".join(sorted(set(terms)))

        conn = sqlite3.connect(self.paths.sqlite_db)
        conn.row_factory = sqlite3.Row
        try:
            try:
                rows = conn.execute(
                    """
                    SELECT rowid AS id, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (match_query, int(top_k)),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for environments where FTS query parser rejects input.
                rows = conn.execute(
                    """
                    SELECT id, 0.0 AS rank
                    FROM chunks
                    WHERE lower(text) LIKE ?
                    LIMIT ?
                    """,
                    (f"%{terms[0]}%", int(top_k)),
                ).fetchall()
        finally:
            conn.close()

        output: list[dict] = []
        for row in rows:
            raw_rank = float(row["rank"]) if "rank" in row.keys() else 0.0
            lexical_score = -raw_rank
            output.append({"id": int(row["id"]), "lexical_score": lexical_score})

        return output

    def _dense_scores(self, query_text: str, chunk_ids: list[int]) -> np.ndarray:
        query_vector = self.embedder.embed_query(query_text)
        candidate_vectors = self._vectors[np.array(chunk_ids, dtype=np.int64)]
        return np.dot(candidate_vectors, query_vector)

    def _fetch_chunk_rows(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        conn = sqlite3.connect(self.paths.sqlite_db)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"""
                SELECT
                  id,
                  chunk_id,
                  source,
                  parent_citation_id,
                  title,
                  text,
                  strength_score,
                  metadata_json
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                tuple(ids),
            ).fetchall()
        finally:
            conn.close()

        output: list[dict] = []
        for row in rows:
            metadata = {}
            raw_metadata = row["metadata_json"]
            if raw_metadata:
                try:
                    metadata = json.loads(str(raw_metadata))
                except json.JSONDecodeError:
                    metadata = {}

            output.append(
                {
                    "id": int(row["id"]),
                    "chunk_id": str(row["chunk_id"]),
                    "source": str(row["source"]),
                    "parent_citation_id": str(row["parent_citation_id"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                    "strength_score": int(row["strength_score"]),
                    "metadata": metadata,
                }
            )

        return output


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores

    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    if abs(max_val - min_val) < 1e-8:
        return np.ones_like(scores, dtype=np.float32)

    return ((scores - min_val) / (max_val - min_val)).astype(np.float32)
