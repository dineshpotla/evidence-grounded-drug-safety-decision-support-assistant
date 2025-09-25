from __future__ import annotations

from ..types import EvidenceChunk, RetrievedEvidence


class EvidenceCorpusBuilder:
    def __init__(self, max_words: int = 90, overlap_words: int = 20) -> None:
        self.max_words = max_words
        self.overlap_words = overlap_words

    def build_chunks(self, evidence_items: list[RetrievedEvidence]) -> list[EvidenceChunk]:
        output: list[EvidenceChunk] = []

        for item in evidence_items:
            chunks = self._chunk_text(item.snippet)
            for idx, chunk_text in enumerate(chunks):
                output.append(
                    EvidenceChunk(
                        chunk_id=f"{item.citation_id}::chunk:{idx}",
                        source=item.source,
                        parent_citation_id=item.citation_id,
                        text=chunk_text,
                        metadata={
                            "title": item.title,
                            "strength_score": item.strength_score,
                            "parent_metadata": item.metadata,
                        },
                    )
                )

        return output

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        if len(words) <= self.max_words:
            return [text]

        output: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.max_words, len(words))
            output.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(0, end - self.overlap_words)

        return output
