from __future__ import annotations

import json
import re

import requests

from ..config import settings
from ..types import EvidenceChunk, GeneratedClaim

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency path
    anthropic = None


class ClaudeAgent:
    """LLM agent wrapper with Anthropic or NVIDIA NIM backends plus fallback mode."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        nvidia_api_key: str | None = None,
        nvidia_model: str | None = None,
    ) -> None:
        self.anthropic_api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.anthropic_model = model or settings.anthropic_model
        self.nvidia_api_key = (
            nvidia_api_key if nvidia_api_key is not None else settings.nvidia_api_key
        )
        self.nvidia_model = nvidia_model or settings.nvidia_model
        self.nvidia_base_url = settings.nvidia_base_url.rstrip("/")
        self.session = requests.Session()
        self.client = (
            anthropic.Anthropic(api_key=self.anthropic_api_key)
            if anthropic is not None and self.anthropic_api_key
            else None
        )

    def rerank_chunks(
        self,
        question: str,
        chunks: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        if not chunks:
            return []

        if self.client is None and not self.nvidia_api_key:
            return self._fallback_rerank(question, chunks, top_k)

        prompt = self._build_rerank_prompt(question=question, chunks=chunks, top_k=top_k)
        text = self._message(prompt)
        ranked_ids = self._parse_ranked_ids(text)

        if not ranked_ids:
            return self._fallback_rerank(question, chunks, top_k)

        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ordered = [by_id[item] for item in ranked_ids if item in by_id]
        if len(ordered) < top_k:
            fallback_tail = [chunk for chunk in chunks if chunk.chunk_id not in ranked_ids]
            ordered.extend(fallback_tail)
        return ordered[:top_k]

    def generate_claims(
        self,
        question: str,
        chunks: list[EvidenceChunk],
    ) -> list[GeneratedClaim]:
        if not chunks:
            return []

        if self.client is None and not self.nvidia_api_key:
            return self._fallback_claims(chunks)

        prompt = self._build_claim_prompt(question=question, chunks=chunks)
        text = self._message(prompt)
        claims = self._parse_claims(text)
        return claims or self._fallback_claims(chunks)

    def _message(self, prompt: str) -> str:
        if self.client is not None:
            return self._message_anthropic(prompt)
        if self.nvidia_api_key:
            return self._message_nvidia(prompt)
        return ""

    def _message_anthropic(self, prompt: str) -> str:
        if self.client is None:
            return ""

        message = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=900,
            temperature=0.0,
            system=(
                "You are a drug safety RAG agent. Follow instructions exactly and avoid "
                "unsupported claims."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        parts = [item.text for item in message.content if getattr(item, "type", "") == "text"]
        return "\n".join(parts)

    def _message_nvidia(self, prompt: str) -> str:
        if not self.nvidia_api_key:
            return ""

        url = f"{self.nvidia_base_url}/chat/completions"
        payload = {
            "model": self.nvidia_model,
            "temperature": 0.0,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a drug safety RAG agent. Follow instructions exactly and "
                        "avoid unsupported claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=settings.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return ""

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

    def _build_rerank_prompt(
        self,
        question: str,
        chunks: list[EvidenceChunk],
        top_k: int,
    ) -> str:
        serialized = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "citation": chunk.parent_citation_id,
                "text": chunk.text[:500],
            }
            for chunk in chunks
        ]

        return (
            "Rerank evidence chunks for the drug safety question and return only JSON.\n"
            'JSON schema: {"ranked_chunk_ids": ["..."]}.\n'
            f"Select exactly {top_k} chunk IDs in descending relevance.\n"
            f"Question: {question}\n"
            f"Chunks: {json.dumps(serialized)}"
        )

    def _build_claim_prompt(self, question: str, chunks: list[EvidenceChunk]) -> str:
        serialized = [
            {
                "chunk_id": chunk.chunk_id,
                "citation_id": chunk.parent_citation_id,
                "text": chunk.text[:450],
            }
            for chunk in chunks
        ]

        return (
            "Generate concise drug safety claims grounded only in evidence chunks.\n"
            "Return only JSON with this schema: "
            '{"claims":[{"text":"...","citation_ids":["..."]}]}.\n'
            "Rules: include citation_ids for every claim, no dosing instructions.\n"
            f"Question: {question}\n"
            f"Evidence: {json.dumps(serialized)}"
        )

    def _parse_ranked_ids(self, text: str) -> list[str]:
        parsed = _parse_json_object(text)
        if not parsed:
            return []
        values = parsed.get("ranked_chunk_ids", [])
        if not isinstance(values, list):
            return []
        return [str(item) for item in values if str(item).strip()]

    def _parse_claims(self, text: str) -> list[GeneratedClaim]:
        parsed = _parse_json_object(text)
        if not parsed:
            return []

        raw_claims = parsed.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        output: list[GeneratedClaim] = []
        for row in raw_claims:
            if not isinstance(row, dict):
                continue
            claim_text = str(row.get("text", "")).strip()
            ids = row.get("citation_ids", [])
            if not claim_text or not isinstance(ids, list):
                continue
            citation_ids = [str(item).strip() for item in ids if str(item).strip()]
            if not citation_ids:
                continue
            output.append(GeneratedClaim(text=claim_text, citation_ids=citation_ids))

        return output

    def _fallback_rerank(
        self,
        question: str,
        chunks: list[EvidenceChunk],
        top_k: int,
    ) -> list[EvidenceChunk]:
        query_terms = _tokenize(question)

        scored: list[tuple[float, EvidenceChunk]] = []
        for chunk in chunks:
            terms = _tokenize(chunk.text)
            overlap = len(query_terms & terms)
            source_bonus = 0.2 if chunk.source == "openfda" else 0.0
            scored.append((overlap + source_bonus, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _fallback_claims(self, chunks: list[EvidenceChunk]) -> list[GeneratedClaim]:
        grouped: dict[str, list[EvidenceChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source, []).append(chunk)

        claims: list[GeneratedClaim] = []
        for source, group in grouped.items():
            top = group[0]
            prefix = {
                "openfda": "Regulatory label evidence indicates",
                "pubmed": "Clinical literature indicates",
                "faers": "Post-marketing surveillance indicates",
            }.get(source, "Retrieved evidence indicates")
            claims.append(
                GeneratedClaim(
                    text=f"{prefix}: {top.text[:220]}",
                    citation_ids=[top.parent_citation_id],
                )
            )

        return claims[:4]


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "drug",
        "patient",
        "risk",
        "is",
        "are",
        "to",
        "of",
        "in",
    }
    return {word for word in words if len(word) > 2 and word not in stop}
