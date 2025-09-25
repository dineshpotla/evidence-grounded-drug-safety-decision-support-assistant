from __future__ import annotations

from typing import Any

import requests

from ..config import settings
from ..types import RetrievedEvidence


class OpenFDARetriever:
    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self.base_url = (base_url or settings.openfda_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout_seconds
        self.session = requests.Session()

    def search_labels(self, drug_name: str, limit: int = 3) -> list[RetrievedEvidence]:
        if not drug_name.strip():
            return []

        query = (
            f'openfda.generic_name:"{drug_name}"+OR+'
            f'openfda.brand_name:"{drug_name}"+OR+'
            f'openfda.substance_name:"{drug_name}"'
        )

        url = f"{self.base_url}/drug/label.json"
        params = {"search": query, "limit": max(1, min(limit, 10))}

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return []

        results = payload.get("results", [])
        output: list[RetrievedEvidence] = []

        for row in results:
            title = (
                self._first_text(row, ["openfda.brand_name", "openfda.generic_name"]) or drug_name
            )
            citation_id = (
                self._first_text(row, ["set_id", "id", "spl_id"]) or f"openfda-{len(output) + 1}"
            )
            snippet, severity = self._build_label_snippet(row)

            output.append(
                RetrievedEvidence(
                    source="openfda",
                    citation_id=f"OPENFDA:{citation_id}",
                    title=f"FDA Drug Label: {title}",
                    snippet=snippet,
                    metadata={
                        "label_severity": severity,
                        "set_id": citation_id,
                    },
                    strength_score=severity,
                )
            )

        return output

    def _first_text(self, row: dict[str, Any], keys: list[str]) -> str | None:
        for key in keys:
            value: Any = row
            for segment in key.split("."):
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(segment)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _build_label_snippet(self, row: dict[str, Any]) -> tuple[str, int]:
        boxed = row.get("boxed_warning", [])
        contra = row.get("contraindications", [])
        warnings = row.get("warnings", [])
        adverse = row.get("adverse_reactions", [])

        parts: list[str] = []
        severity = 0

        if boxed:
            parts.append(f"Boxed warning: {boxed[0][:300]}")
            severity = max(severity, 3)
        if contra:
            parts.append(f"Contraindications: {contra[0][:250]}")
            severity = max(severity, 3)
        if warnings:
            parts.append(f"Warnings: {warnings[0][:250]}")
            severity = max(severity, 2)
        if adverse and len(parts) < 3:
            parts.append(f"Adverse reactions: {adverse[0][:220]}")
            severity = max(severity, 1)

        if not parts:
            return ("No key safety sections available from this label record.", 0)

        return (" ".join(parts), severity)
