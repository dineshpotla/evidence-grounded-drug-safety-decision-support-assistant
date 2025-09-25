from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from ..config import settings
from ..types import SafetyRequest
from .clinical_text import normalize_drug_name


class DrugNameResolver:
    def __init__(self, dictionary_path: str | None = None, threshold: float = 0.86) -> None:
        self.threshold = threshold
        self.dictionary_path = Path(dictionary_path or settings.drug_dictionary_path)
        self.terms = self._load_terms()

    def resolve_request(self, request: SafetyRequest) -> SafetyRequest:
        updated = request.model_copy(deep=True)
        if updated.drug:
            updated.drug = self.resolve(updated.drug)
        if updated.drug_a:
            updated.drug_a = self.resolve(updated.drug_a)
        if updated.drug_b:
            updated.drug_b = self.resolve(updated.drug_b)
        updated.current_meds = [self.resolve(item) for item in updated.current_meds]
        return updated

    def resolve(self, value: str) -> str:
        normalized = normalize_drug_name(value)
        if not normalized:
            return value.strip()

        if not self.terms:
            return normalized

        if normalized in self.terms:
            return normalized

        candidate, score = self._best_match(normalized)
        if candidate and score >= self.threshold:
            return candidate

        return normalized

    def _load_terms(self) -> list[str]:
        if not self.dictionary_path.exists():
            return []

        terms: list[str] = []
        seen: set[str] = set()
        for line in self.dictionary_path.read_text(encoding="utf-8").splitlines():
            token = normalize_drug_name(line)
            if not token or token in seen:
                continue
            seen.add(token)
            terms.append(token)

        return terms

    def _best_match(self, target: str) -> tuple[str | None, float]:
        best_term = None
        best_score = 0.0

        for term in self.terms:
            score = SequenceMatcher(a=target, b=term).ratio()
            if score > best_score:
                best_score = score
                best_term = term

        return best_term, best_score
