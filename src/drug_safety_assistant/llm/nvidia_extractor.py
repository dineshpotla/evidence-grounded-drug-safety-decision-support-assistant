from __future__ import annotations

import json
import re
from typing import Any

import requests

from ..config import settings
from ..intents import classify_intent
from ..types import ExtractedEntities, Intent, SafetyRequest
from ..utils.clinical_text import normalize_drug_name


class NvidiaIntentExtractor:
    """NVIDIA NIM-backed intent/entity extractor with deterministic fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.base_url = (base_url or settings.nvidia_base_url).rstrip("/")
        self.model = model or settings.nvidia_extract_model
        self.session = requests.Session()

    def extract(self, request: SafetyRequest) -> ExtractedEntities:
        llm_entities = self._extract_with_nvidia(request)
        if llm_entities is not None:
            return self._merge_with_heuristics(request, llm_entities)

        return self._heuristic_extract(request)

    def enrich_request(self, request: SafetyRequest, entities: ExtractedEntities) -> SafetyRequest:
        enriched = request.model_copy(deep=True)

        if not enriched.drug and entities.drug_mentions:
            enriched.drug = entities.drug_mentions[0]

        if not enriched.drug_a and len(entities.drug_mentions) >= 2:
            enriched.drug_a = entities.drug_mentions[0]
        if not enriched.drug_b and len(entities.drug_mentions) >= 2:
            enriched.drug_b = entities.drug_mentions[1]

        if not enriched.age_group and entities.age_group:
            enriched.age_group = _normalize_age_group(entities.age_group)
        if not enriched.pregnancy_status and entities.pregnancy_status:
            enriched.pregnancy_status = _normalize_pregnancy_status(entities.pregnancy_status)
        if not enriched.trimester and entities.trimester:
            enriched.trimester = _normalize_trimester(entities.trimester)
        if not enriched.kidney_status and entities.kidney_status:
            enriched.kidney_status = _normalize_kidney_status(entities.kidney_status)
        if not enriched.liver_status and entities.liver_status:
            enriched.liver_status = entities.liver_status

        if enriched.drug:
            enriched.drug = _normalize_optional_drug_field(enriched.drug)
        if enriched.drug_a:
            enriched.drug_a = _normalize_optional_drug_field(enriched.drug_a)
        if enriched.drug_b:
            enriched.drug_b = _normalize_optional_drug_field(enriched.drug_b)

        return enriched

    def _extract_with_nvidia(self, request: SafetyRequest) -> ExtractedEntities | None:
        if not self.api_key:
            return None

        prompt = self._build_extraction_prompt(request.question)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical information extraction model. Return only JSON "
                        "without markdown or commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
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
            return None

        text = ""
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            text = str(message.get("content", ""))

        parsed = self._extract_payload_dict(text)
        if parsed is None:
            return None

        return self._parse_entities(parsed)

    def _extract_payload_dict(self, text: str) -> dict[str, Any] | None:
        json_blob = self._extract_json_block(text)
        if not json_blob:
            return None

        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _extract_json_block(self, text: str) -> str | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None

    def _parse_entities(self, parsed: dict[str, Any]) -> ExtractedEntities:
        intent_hint: Intent | None = None
        raw_intent = str(parsed.get("intent", "")).strip().lower()
        if raw_intent in {intent.value for intent in Intent}:
            intent_hint = Intent(raw_intent)

        drugs = parsed.get("drug_mentions", [])
        if not isinstance(drugs, list):
            drugs = []

        return ExtractedEntities(
            intent_hint=intent_hint,
            drug_mentions=[str(item).strip() for item in drugs if str(item).strip()],
            age_group=_optional_text(parsed.get("age_group")),
            pregnancy_status=_optional_text(parsed.get("pregnancy_status")),
            trimester=_optional_text(parsed.get("trimester")),
            kidney_status=_optional_text(parsed.get("kidney_status")),
            liver_status=_optional_text(parsed.get("liver_status")),
        )

    def _merge_with_heuristics(
        self,
        request: SafetyRequest,
        llm_entities: ExtractedEntities,
    ) -> ExtractedEntities:
        heuristic = self._heuristic_extract(request)
        merged = llm_entities.model_copy(deep=True)

        if not merged.drug_mentions:
            merged.drug_mentions = heuristic.drug_mentions
        if not merged.intent_hint:
            merged.intent_hint = heuristic.intent_hint
        if not merged.age_group:
            merged.age_group = heuristic.age_group
        if not merged.pregnancy_status:
            merged.pregnancy_status = heuristic.pregnancy_status
        if not merged.trimester:
            merged.trimester = heuristic.trimester
        if not merged.kidney_status:
            merged.kidney_status = heuristic.kidney_status
        if not merged.liver_status:
            merged.liver_status = heuristic.liver_status

        return merged

    def _heuristic_extract(self, request: SafetyRequest) -> ExtractedEntities:
        text = request.question
        lowered = text.lower()
        drug_mentions = _extract_drug_like_mentions(text)

        age_group = None
        if any(token in lowered for token in ["elderly", "older", "over 65", "geriatric"]):
            age_group = "over 65"

        pregnancy_status = None
        trimester = None
        if "pregnan" in lowered:
            pregnancy_status = "yes"
            trim_match = re.search(r"(first|second|third|1st|2nd|3rd|trimester\s*[123])", lowered)
            if trim_match:
                value = trim_match.group(1)
                if value.startswith("first") or value.startswith("1"):
                    trimester = "1"
                elif value.startswith("second") or value.startswith("2"):
                    trimester = "2"
                else:
                    trimester = "3"

        kidney_status = None
        if "dialysis" in lowered:
            kidney_status = "dialysis"
        elif "ckd" in lowered or "renal" in lowered or "kidney" in lowered:
            kidney_status = "CKD"

        liver_status = "none" if "no liver" in lowered else None

        return ExtractedEntities(
            intent_hint=classify_intent(request),
            drug_mentions=drug_mentions,
            age_group=age_group,
            pregnancy_status=pregnancy_status,
            trimester=trimester,
            kidney_status=kidney_status,
            liver_status=liver_status,
        )

    def _build_extraction_prompt(self, question: str) -> str:
        return (
            "Extract entities from this clinical drug safety question and respond with JSON only. "
            'JSON schema: {"intent":"interaction|pregnancy|renal|patient_specific|general", '
            '"drug_mentions":[],"age_group":null,"pregnancy_status":null,'
            '"trimester":null,"kidney_status":null,"liver_status":null}. '
            "No markdown. No prose.\n"
            f"Question: {question}"
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_age_group(value: str) -> str:
    lowered = value.lower()
    if "elder" in lowered or "over" in lowered or "geriatric" in lowered or "65" in lowered:
        return "over 65"
    if "pediatric" in lowered or "child" in lowered:
        return "pediatric"
    if "adult" in lowered:
        return "adult"
    return value


def _normalize_pregnancy_status(value: str) -> str:
    lowered = value.lower()
    if lowered in {"yes", "pregnant", "true"}:
        return "yes"
    if lowered in {"no", "not pregnant", "false"}:
        return "no"
    return value


def _normalize_trimester(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("1") or "first" in lowered:
        return "1"
    if lowered.startswith("2") or "second" in lowered:
        return "2"
    if lowered.startswith("3") or "third" in lowered:
        return "3"
    return value


def _normalize_kidney_status(value: str) -> str:
    lowered = value.lower()
    if "dialysis" in lowered:
        return "dialysis"
    if "ckd" in lowered or "kidney" in lowered or "renal" in lowered:
        return "CKD"
    return value


def _normalize_optional_drug_field(value: str) -> str:
    normalized = normalize_drug_name(value)
    return normalized or value.strip()


def _extract_drug_like_mentions(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
    stop = {
        "does",
        "with",
        "interact",
        "interaction",
        "safe",
        "safely",
        "pregnancy",
        "pregnant",
        "patient",
        "elderly",
        "first",
        "second",
        "third",
        "trimester",
        "kidney",
        "renal",
        "ckd",
        "side",
        "effects",
        "serious",
        "can",
        "take",
        "both",
        "stage",
    }
    candidates = [token for token in tokens if token.lower() not in stop]

    dedup: list[str] = []
    dedup_lower: set[str] = set()
    for token in candidates:
        lowered = token.lower()
        if lowered not in dedup_lower:
            dedup.append(token)
            dedup_lower.add(lowered)

    return dedup[:4]
