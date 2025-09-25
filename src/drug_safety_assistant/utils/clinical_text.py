from __future__ import annotations

import re
from difflib import SequenceMatcher

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|iu)(?:\b|(?=\s|$))",
    flags=re.IGNORECASE,
)
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s-]")
MULTISPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")

FORM_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "gel",
    "cream",
    "ointment",
    "solution",
    "suspension",
    "topical",
    "patch",
    "injection",
    "injectable",
    "oral",
    "iv",
    "sr",
    "xr",
    "er",
    "dr",
}


def normalize_drug_name(name: str | None) -> str:
    if not name:
        return ""

    text = name.lower().strip()
    text = DOSAGE_PATTERN.sub(" ", text)
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = text.replace("/", " ")
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    tokens = [token for token in text.split() if token not in FORM_TOKENS]
    normalized = " ".join(tokens).strip()
    return normalized


def contains_drug_term(text: str, term: str, fuzzy_threshold: float = 0.84) -> bool:
    normalized_text = normalize_drug_name(text)
    normalized_term = normalize_drug_name(term)

    if not normalized_text or not normalized_term:
        return False

    if normalized_term in normalized_text:
        return True

    text_tokens = TOKEN_PATTERN.findall(normalized_text)
    term_tokens = TOKEN_PATTERN.findall(normalized_term)
    if not text_tokens or not term_tokens:
        return False

    if len(term_tokens) == 1:
        target = term_tokens[0]
        if len(target) < 4:
            return target in text_tokens
        return any(
            SequenceMatcher(a=target, b=token).ratio() >= fuzzy_threshold for token in text_tokens
        )

    phrase = " ".join(term_tokens)
    if phrase in normalized_text:
        return True

    joined_tokens = " ".join(text_tokens)
    return SequenceMatcher(a=phrase, b=joined_tokens).ratio() >= 0.75


def unique_normalized_terms(values: list[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = normalize_drug_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)

    return output
