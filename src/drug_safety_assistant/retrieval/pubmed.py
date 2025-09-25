from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

from ..config import settings
from ..types import RetrievedEvidence


class PubMedRetriever:
    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        self.base_url = (base_url or settings.pubmed_eutils_base).rstrip("/")
        self.timeout = timeout or settings.request_timeout_seconds
        self.api_key = settings.ncbi_api_key
        self.session = requests.Session()

    def search(
        self,
        query: str,
        max_results: int = 5,
        start_year: int = 2011,
    ) -> list[RetrievedEvidence]:
        if not query.strip():
            return []

        pmids = self._esearch(query, max_results=max_results, start_year=start_year)
        if not pmids:
            return []

        abstracts = self._efetch(pmids)
        output: list[RetrievedEvidence] = []
        for pmid, data in abstracts.items():
            title = data.get("title", f"PubMed {pmid}")
            abstract = data.get("abstract", "No abstract available")
            pubtypes = data.get("publication_types", [])
            year = data.get("year")
            strength = self._strength_from_pubtypes(pubtypes)

            output.append(
                RetrievedEvidence(
                    source="pubmed",
                    citation_id=f"PMID:{pmid}",
                    title=title,
                    snippet=abstract[:500],
                    metadata={
                        "publication_types": pubtypes,
                        "year": year,
                    },
                    strength_score=strength,
                )
            )

        return output

    def _esearch(self, query: str, max_results: int, start_year: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": f"({query}) AND ({start_year}:3000[pdat])",
            "retmax": max(1, min(max_results, 20)),
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.base_url}/esearch.fcgi"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return []

        return payload.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> dict[str, dict[str, object]]:
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            xml_text = response.text
        except requests.RequestException:
            return {}

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return {}

        output: dict[str, dict[str, object]] = {}

        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//MedlineCitation/PMID")
            pmid = (pmid_node.text or "").strip() if pmid_node is not None else ""
            if not pmid:
                continue

            title_node = article.find(".//Article/ArticleTitle")
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""

            abstract_parts = [
                "".join(node.itertext()).strip()
                for node in article.findall(".//Article/Abstract/AbstractText")
                if "".join(node.itertext()).strip()
            ]
            abstract = " ".join(abstract_parts)

            pubtype_nodes = article.findall(".//Article/PublicationTypeList/PublicationType")
            pubtypes = [
                "".join(node.itertext()).strip().lower()
                for node in pubtype_nodes
                if "".join(node.itertext()).strip()
            ]

            year_node = article.find(".//Article/Journal/JournalIssue/PubDate/Year")
            year = year_node.text.strip() if year_node is not None and year_node.text else None

            output[pmid] = {
                "title": title or f"PubMed {pmid}",
                "abstract": abstract or "No abstract text available.",
                "publication_types": pubtypes,
                "year": year,
            }

        return output

    def _strength_from_pubtypes(self, pubtypes: list[str]) -> int:
        lowered = {item.lower() for item in pubtypes}
        if any("meta-analysis" in item or "systematic review" in item for item in lowered):
            return 3
        if any("randomized" in item or "controlled trial" in item for item in lowered):
            return 2
        if lowered:
            return 1
        return 0
