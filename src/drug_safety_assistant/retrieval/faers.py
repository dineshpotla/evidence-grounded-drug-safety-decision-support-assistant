from __future__ import annotations

from collections import Counter
from typing import Any

import requests

from ..config import settings
from ..types import RetrievedEvidence


class FAERSRetriever:
    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self.base_url = (base_url or settings.openfda_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout_seconds
        self.session = requests.Session()

    def fetch_signal(
        self, drug_name: str, event_term: str | None = None
    ) -> list[RetrievedEvidence]:
        if not drug_name.strip():
            return []

        normalized_drug = drug_name.upper()
        total_drug = self._total_reports(f'patient.drug.medicinalproduct:"{normalized_drug}"')
        total_all = self._total_reports(None)

        if total_drug <= 0 or total_all <= 0:
            return []

        sample_events = self._sample_events(normalized_drug, limit=100)
        if not sample_events:
            return []

        serious_pct = self._serious_outcome_pct(sample_events)
        reaction_counts = self._reaction_counts(sample_events)

        if event_term:
            reaction = event_term.upper()
        else:
            reaction = reaction_counts[0][0] if reaction_counts else "UNKNOWN"

        a = self._total_reports(
            (
                f'patient.drug.medicinalproduct:"{normalized_drug}"+AND+'
                f'patient.reaction.reactionmeddrapt:"{reaction}"'
            )
        )
        reaction_total = self._total_reports(f'patient.reaction.reactionmeddrapt:"{reaction}"')

        prr = self._compute_prr(
            a=a,
            total_drug=total_drug,
            reaction_total=reaction_total,
            total_all=total_all,
        )
        signal_score, bucket = self._signal_bucket(prr=prr, case_count=a)

        top_reactions_str = ", ".join(f"{name}:{count}" for name, count in reaction_counts[:5])
        snippet = (
            f"FAERS reports for {normalized_drug}: total={total_drug}, "
            f"serious_outcomes={serious_pct:.1f}%, "
            f"top_reactions={top_reactions_str}. "
            f"Signal term={reaction}, PRR={prr:.2f}, bucket={bucket}."
        )

        evidence = RetrievedEvidence(
            source="faers",
            citation_id=f"FAERS:{normalized_drug}:{reaction}",
            title=f"FAERS signal summary for {normalized_drug}",
            snippet=snippet,
            metadata={
                "drug": normalized_drug,
                "signal_term": reaction,
                "total_reports": total_drug,
                "serious_pct": serious_pct,
                "prr": prr,
                "case_count": a,
                "signal_bucket": bucket,
            },
            strength_score=signal_score,
        )
        return [evidence]

    def _sample_events(self, drug_name: str, limit: int = 100) -> list[dict[str, Any]]:
        url = f"{self.base_url}/drug/event.json"
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": max(1, min(limit, 100)),
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return []

        return payload.get("results", [])

    def _reaction_counts(self, events: list[dict[str, Any]]) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for event in events:
            patient = event.get("patient", {})
            reactions = patient.get("reaction", [])
            for reaction in reactions:
                term = reaction.get("reactionmeddrapt")
                if term:
                    counter[str(term).upper()] += 1
        return counter.most_common()

    def _serious_outcome_pct(self, events: list[dict[str, Any]]) -> float:
        if not events:
            return 0.0

        serious = sum(1 for event in events if str(event.get("serious", "0")) == "1")
        return (serious / len(events)) * 100.0

    def _total_reports(self, search: str | None) -> int:
        url = f"{self.base_url}/drug/event.json"
        params = {"limit": 1}
        if search:
            params["search"] = search

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return 0

        return int(payload.get("meta", {}).get("results", {}).get("total", 0))

    def _compute_prr(self, a: int, total_drug: int, reaction_total: int, total_all: int) -> float:
        b = max(total_drug - a, 0)
        c = max(reaction_total - a, 0)
        d = max(total_all - (a + b + c), 0)

        if (a + b) == 0 or (c + d) == 0 or c == 0:
            return 0.0

        left = a / (a + b)
        right = c / (c + d)
        if right == 0:
            return 0.0
        return left / right

    def _signal_bucket(self, prr: float, case_count: int) -> tuple[int, str]:
        if prr >= 2.0 and case_count >= 3:
            return 3, "High"
        if prr >= 1.5 and case_count >= 2:
            return 2, "Moderate"
        if prr > 0:
            return 1, "Low"
        return 0, "Low"
