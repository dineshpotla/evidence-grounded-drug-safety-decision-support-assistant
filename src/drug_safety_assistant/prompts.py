from __future__ import annotations

from pathlib import Path

import yaml

from .types import Intent, SafetyRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_BY_INTENT: dict[Intent, str] = {
    Intent.INTERACTION: "interaction_template.md",
    Intent.PREGNANCY: "pregnancy_template.md",
    Intent.RENAL: "renal_template.md",
    Intent.PATIENT_SPECIFIC: "general_safety_template.md",
    Intent.GENERAL: "general_safety_template.md",
}


class PromptRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        self.registry_path = registry_path or (PROJECT_ROOT / "prompt_registry.yaml")
        self._registry = self._load()

    def _load(self) -> dict:
        if not self.registry_path.exists():
            return {"prompts": []}
        with self.registry_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {"prompts": []}

    def pick_template_dir(self, intent: Intent, preferred_version: str) -> Path:
        for item in self._registry.get("prompts", []):
            if item.get("intent") == intent.value and item.get("directory"):
                return PROJECT_ROOT / item["directory"]

        # Fallback to the selected version directory.
        return PROJECT_ROOT / "prompts" / preferred_version


def render_prompt(
    intent: Intent,
    request: SafetyRequest,
    evidence_pack_text: str,
    prompt_version: str,
    patient_context_text: str,
) -> str:
    registry = PromptRegistry()
    template_dir = registry.pick_template_dir(intent=intent, preferred_version=prompt_version)
    template_path = template_dir / TEMPLATE_BY_INTENT[intent]

    if not template_path.exists():
        template_path = PROJECT_ROOT / "prompts" / "v1" / TEMPLATE_BY_INTENT[intent]

    template = template_path.read_text(encoding="utf-8")
    return template.format(
        question=request.question,
        drug=request.drug or request.drug_a or request.drug_b or "unknown",
        drug_a=request.drug_a or request.drug or "unknown",
        drug_b=request.drug_b or "unknown",
        trimester=request.trimester or "unknown",
        kidney_status=request.kidney_status or "unknown",
        patient_context=patient_context_text,
        evidence_pack=evidence_pack_text,
    )
