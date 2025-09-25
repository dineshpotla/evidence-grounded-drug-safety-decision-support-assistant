from __future__ import annotations

from ..config import settings
from ..llm.claude import ClaudeAgent
from ..types import Intent, RetrievedEvidence, SafetyRequest
from ..utils.clinical_text import contains_drug_term, unique_normalized_terms
from .aggregator import MultiSourceRetriever
from .corpus import EvidenceCorpusBuilder
from .embeddings import MedCPTEmbedder
from .persistent_index import PersistentHybridRetriever
from .vector_index import VectorIndex


class AgenticRetriever:
    """Multi-stage retrieval: source retrieval -> vector recall -> Claude reranking."""

    def __init__(
        self,
        source_retriever: MultiSourceRetriever | None = None,
        embedder: MedCPTEmbedder | None = None,
        corpus_builder: EvidenceCorpusBuilder | None = None,
        claude_agent: ClaudeAgent | None = None,
    ) -> None:
        self.source_retriever = source_retriever or MultiSourceRetriever()
        self.embedder = embedder or MedCPTEmbedder()
        self.corpus_builder = corpus_builder or EvidenceCorpusBuilder()
        self.claude_agent = claude_agent or ClaudeAgent()
        self.persistent = self._build_persistent_retriever()

    def retrieve(self, request: SafetyRequest, intent: Intent) -> list[RetrievedEvidence]:
        required_terms = self._required_terms(request)

        if self.persistent is not None:
            query_text = self._query_text(request=request)
            chunks = self.persistent.search(query_text=query_text, top_k=settings.vector_top_k)
            if chunks:
                reranked = self.claude_agent.rerank_chunks(
                    question=query_text,
                    chunks=chunks,
                    top_k=settings.rerank_top_k,
                )
                persistent_evidence = self._persistent_chunks_to_evidence(reranked)
                if self._coverage_is_sufficient(
                    persistent_evidence,
                    required_terms=required_terms,
                    intent=intent,
                ):
                    return persistent_evidence

        raw_evidence = self.source_retriever.retrieve(request=request, intent=intent)
        if not raw_evidence:
            return []

        chunks = self.corpus_builder.build_chunks(raw_evidence)
        if not chunks:
            return raw_evidence

        query_text = self._query_text(request=request)
        doc_vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        query_vector = self.embedder.embed_query(query_text)

        index = VectorIndex()
        index.add(chunks=chunks, embeddings=doc_vectors)
        initial = index.search(query_vector=query_vector, top_k=settings.vector_top_k)

        reranked = self.claude_agent.rerank_chunks(
            question=query_text,
            chunks=initial,
            top_k=settings.rerank_top_k,
        )

        return self._chunks_to_evidence(reranked, raw_evidence)

    def _query_text(self, request: SafetyRequest) -> str:
        parts = [request.question]
        if request.drug:
            parts.append(f"drug={request.drug}")
        if request.drug_a:
            parts.append(f"drug_a={request.drug_a}")
        if request.drug_b:
            parts.append(f"drug_b={request.drug_b}")
        if request.age_group:
            parts.append(f"age_group={request.age_group}")
        if request.pregnancy_status:
            parts.append(f"pregnancy_status={request.pregnancy_status}")
        if request.trimester:
            parts.append(f"trimester={request.trimester}")
        if request.kidney_status:
            parts.append(f"kidney_status={request.kidney_status}")
        if request.liver_status:
            parts.append(f"liver_status={request.liver_status}")
        return " | ".join(parts)

    def _chunks_to_evidence(
        self,
        chunks,
        raw_evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        parent_map = {item.citation_id: item for item in raw_evidence}
        output: list[RetrievedEvidence] = []
        seen: set[tuple[str, str]] = set()

        for rank, chunk in enumerate(chunks, start=1):
            parent = parent_map.get(chunk.parent_citation_id)
            if parent is None:
                continue

            dedupe_key = (chunk.parent_citation_id, chunk.text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            metadata = dict(parent.metadata)
            metadata.update(chunk.metadata)
            metadata["rank"] = rank

            output.append(
                RetrievedEvidence(
                    source=parent.source,
                    citation_id=parent.citation_id,
                    title=parent.title,
                    snippet=chunk.text,
                    metadata=metadata,
                    strength_score=parent.strength_score,
                )
            )

        # Preserve at least one item per source when reranker output is sparse.
        if output:
            sources = {item.source for item in output}
            for parent in raw_evidence:
                if parent.source not in sources:
                    output.append(parent)
                    sources.add(parent.source)

        return output or raw_evidence

    def _required_terms(self, request: SafetyRequest) -> list[str]:
        return unique_normalized_terms([request.drug, request.drug_a, request.drug_b])

    def _coverage_is_sufficient(
        self,
        evidence: list[RetrievedEvidence],
        *,
        required_terms: list[str],
        intent: Intent,
    ) -> bool:
        if not evidence:
            return False
        if not required_terms:
            return len(evidence) >= 2

        term_hits = {term: 0 for term in required_terms}
        for item in evidence:
            combined_text = f"{item.title} {item.citation_id} {item.snippet}".lower()
            for term in required_terms:
                if contains_drug_term(combined_text, term):
                    term_hits[term] += 1

        if intent == Intent.INTERACTION and len(required_terms) >= 2:
            return all(count > 0 for count in term_hits.values())

        primary = required_terms[0]
        return term_hits.get(primary, 0) > 0

    def _persistent_chunks_to_evidence(self, chunks: list) -> list[RetrievedEvidence]:
        output: list[RetrievedEvidence] = []
        seen: set[tuple[str, str]] = set()

        for chunk in chunks:
            key = (chunk.parent_citation_id, chunk.text)
            if key in seen:
                continue
            seen.add(key)

            metadata = dict(chunk.metadata)
            title = str(metadata.pop("title", chunk.parent_citation_id))
            strength = int(metadata.pop("strength_score", 0))

            output.append(
                RetrievedEvidence(
                    source=chunk.source,
                    citation_id=chunk.parent_citation_id,
                    title=title,
                    snippet=chunk.text,
                    metadata=metadata,
                    strength_score=strength,
                )
            )

        return output

    def _build_persistent_retriever(self) -> PersistentHybridRetriever | None:
        if not settings.use_persistent_index:
            return None
        if not PersistentHybridRetriever.is_ready(settings.persistent_index_dir):
            return None
        return PersistentHybridRetriever(
            index_dir=settings.persistent_index_dir,
            embedder=self.embedder,
            lexical_k=settings.lexical_candidate_k,
            dense_weight=settings.hybrid_dense_weight,
        )
