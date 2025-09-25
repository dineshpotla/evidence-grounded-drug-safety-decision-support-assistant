from __future__ import annotations

from .types import EvidencePack, RiskBreakdown, RiskLevel, SafetyRequest


def _label_severity_score(pack: EvidencePack) -> int:
    score = 0
    for item in pack.by_source("openfda"):
        from_meta = int(item.metadata.get("label_severity", 0))
        score = max(score, from_meta)

        text = item.snippet.lower()
        if "boxed warning" in text or "contraindication" in text:
            score = max(score, 3)
        elif "warning" in text or "precaution" in text:
            score = max(score, 2)
        elif "adverse" in text:
            score = max(score, 1)
    return min(score, 3)


def _literature_strength_score(pack: EvidencePack) -> int:
    return min(max((item.strength_score for item in pack.by_source("pubmed")), default=0), 3)


def _faers_signal_score(pack: EvidencePack) -> int:
    max_score = max((item.strength_score for item in pack.by_source("faers")), default=0)
    return min(max_score, 3)


def _patient_modifier_score(request: SafetyRequest) -> int:
    points = 0

    if request.age_group and any(
        token in request.age_group.lower() for token in ["65", "elder", "geriatric"]
    ):
        points += 1

    kidney_risk = request.kidney_status and request.kidney_status.lower() not in {
        "none",
        "normal",
        "unknown",
    }
    liver_risk = request.liver_status and request.liver_status.lower() not in {
        "none",
        "normal",
        "unknown",
    }
    if kidney_risk or liver_risk:
        points += 1

    pregnancy_risk = request.pregnancy_status and request.pregnancy_status.lower() in {
        "yes",
        "pregnant",
    }
    if pregnancy_risk:
        points += 1

    return min(points, 2)


def compute_risk_score(request: SafetyRequest, pack: EvidencePack) -> RiskBreakdown:
    label = _label_severity_score(pack)
    literature = _literature_strength_score(pack)
    faers = _faers_signal_score(pack)
    modifiers = _patient_modifier_score(request)

    # Weighted score scaled to 0-3.
    modifier_scaled = (modifiers / 2) * 3
    weighted = (0.35 * label) + (0.25 * literature) + (0.25 * faers) + (0.15 * modifier_scaled)

    if weighted >= 2.0:
        level = RiskLevel.HIGH
    elif weighted >= 1.0:
        level = RiskLevel.MODERATE
    else:
        level = RiskLevel.LOW

    drivers: list[str] = []
    if label >= 2:
        drivers.append("label warning severity")
    if literature >= 2:
        drivers.append("higher-strength literature")
    if faers >= 2:
        drivers.append("FAERS disproportionality signal")
    if modifiers > 0:
        drivers.append("patient-specific modifiers")

    explanation = (
        "Risk elevated due to " + ", ".join(drivers) + "."
        if drivers
        else "Risk estimate limited by weak or sparse evidence."
    )

    return RiskBreakdown(
        label_severity=label,
        literature_strength=literature,
        faers_signal=faers,
        patient_modifiers=modifiers,
        weighted_score=round(weighted, 2),
        risk_level=level,
        explanation=explanation,
    )
