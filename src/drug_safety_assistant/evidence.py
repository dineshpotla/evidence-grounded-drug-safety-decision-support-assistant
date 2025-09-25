from __future__ import annotations

from collections import OrderedDict

from .types import EvidencePack, RetrievedEvidence


def build_evidence_pack(items: list[RetrievedEvidence]) -> EvidencePack:
    dedup: "OrderedDict[str, RetrievedEvidence]" = OrderedDict()
    for item in items:
        key = f"{item.source}:{item.citation_id}"
        if key not in dedup:
            dedup[key] = item
    return EvidencePack(items=list(dedup.values()))


def evidence_pack_to_context(pack: EvidencePack, max_chars: int = 5000) -> str:
    lines: list[str] = []
    for item in pack.items:
        lines.append(f"[{item.citation_id}] ({item.source}) {item.title}: {item.snippet}")

    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[: max_chars - 3] + "..."
    return context
