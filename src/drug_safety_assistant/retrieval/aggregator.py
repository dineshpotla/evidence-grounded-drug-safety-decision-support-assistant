from __future__ import annotations

from ..types import Intent, RetrievedEvidence, SafetyRequest
from ..utils.clinical_text import contains_drug_term, normalize_drug_name, unique_normalized_terms
from .faers import FAERSRetriever
from .openfda import OpenFDARetriever
from .pubmed import PubMedRetriever


class MultiSourceRetriever:
    def __init__(
        self,
        openfda: OpenFDARetriever | None = None,
        pubmed: PubMedRetriever | None = None,
        faers: FAERSRetriever | None = None,
    ) -> None:
        self.openfda = openfda or OpenFDARetriever()
        self.pubmed = pubmed or PubMedRetriever()
        self.faers = faers or FAERSRetriever()

    def retrieve(self, request: SafetyRequest, intent: Intent) -> list[RetrievedEvidence]:
        evidence: list[RetrievedEvidence] = []
        drug_a = normalize_drug_name(request.drug_a)
        drug_b = normalize_drug_name(request.drug_b)
        primary_drug = normalize_drug_name(request.drug or request.drug_a or request.drug_b)

        if intent == Intent.INTERACTION and drug_a and drug_b:
            evidence.extend(self.openfda.search_labels(drug_a, limit=3))
            evidence.extend(self.openfda.search_labels(drug_b, limit=3))
            evidence.extend(
                self.pubmed.search(
                    query=(
                        f'("{drug_a}" AND "{drug_b}") AND '
                        "(interaction OR concomitant OR combination OR oxidation)"
                    ),
                    max_results=10,
                )
            )
            evidence.extend(self.faers.fetch_signal(drug_a))
            evidence.extend(self.faers.fetch_signal(drug_b))
            return filter_retrieved_evidence(
                evidence=evidence,
                request=request,
                intent=intent,
            )

        if primary_drug:
            evidence.extend(self.openfda.search_labels(primary_drug, limit=3))
            evidence.extend(
                self.pubmed.search(
                    query=f'"{primary_drug}" AND (safety OR adverse OR contraindication)',
                    max_results=10,
                )
            )
            evidence.extend(self.faers.fetch_signal(primary_drug))

        return filter_retrieved_evidence(
            evidence=evidence,
            request=request,
            intent=intent,
        )


def filter_retrieved_evidence(
    *,
    evidence: list[RetrievedEvidence],
    request: SafetyRequest,
    intent: Intent,
) -> list[RetrievedEvidence]:
    if not evidence:
        return []

    terms = unique_normalized_terms(
        [request.drug, request.drug_a, request.drug_b] + list(request.current_meds)
    )
    if not terms:
        return evidence

    filtered: list[RetrievedEvidence] = []
    term_coverage: dict[str, int] = {term: 0 for term in terms}

    for item in evidence:
        search_text = f"{item.title} {item.citation_id} {item.snippet}".lower()
        matched = [term for term in terms if contains_drug_term(search_text, term)]
        if not matched:
            continue

        metadata = dict(item.metadata)
        metadata["matched_terms"] = matched
        filtered.append(item.model_copy(update={"metadata": metadata}))
        for term in matched:
            term_coverage[term] += 1

    if not filtered:
        return []

    if intent == Intent.INTERACTION and len(terms) >= 2:
        missing_terms = [term for term, count in term_coverage.items() if count == 0]
        if missing_terms:
            supplemental = _supplement_missing_terms(
                evidence=evidence,
                existing=filtered,
                missing_terms=missing_terms,
            )
            filtered.extend(supplemental)

    # Rank by relevance and source signal so unrelated publications don't dominate.
    filtered.sort(
        key=lambda item: (
            len(item.metadata.get("matched_terms", [])),
            1 if item.source == "openfda" else 0,
            1 if item.source == "faers" else 0,
            item.strength_score,
        ),
        reverse=True,
    )

    dedup: list[RetrievedEvidence] = []
    seen: set[tuple[str, str]] = set()
    for item in filtered:
        key = (item.source, item.citation_id)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)

    return dedup


def _supplement_missing_terms(
    *,
    evidence: list[RetrievedEvidence],
    existing: list[RetrievedEvidence],
    missing_terms: list[str],
) -> list[RetrievedEvidence]:
    existing_keys = {(item.source, item.citation_id) for item in existing}
    output: list[RetrievedEvidence] = []

    for term in missing_terms:
        for item in evidence:
            if (item.source, item.citation_id) in existing_keys:
                continue

            search_text = f"{item.title} {item.citation_id} {item.snippet}".lower()
            if not contains_drug_term(search_text, term):
                continue

            metadata = dict(item.metadata)
            matched = set(metadata.get("matched_terms", []))
            matched.add(term)
            metadata["matched_terms"] = sorted(matched)

            patched = item.model_copy(update={"metadata": metadata})
            output.append(patched)
            existing_keys.add((item.source, item.citation_id))
            break

    return output
