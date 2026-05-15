from __future__ import annotations

from dataclasses import dataclass


FAQ_DETAIL_LEVEL_ALIASES = {
    "easy": "low",
    "low": "low",
    "medium": "medium",
    "hard": "high",
    "high": "high",
}


def normalize_faq_detail_level(detail_level: str) -> str:
    normalized = str(detail_level or "medium").strip().lower()
    if normalized not in FAQ_DETAIL_LEVEL_ALIASES:
        raise ValueError("detailLevel must be one of: easy, medium, hard")
    return FAQ_DETAIL_LEVEL_ALIASES[normalized]


@dataclass
class FAQGenerationConfig:
    """Settings for FAQ generation."""

    language: str = "ru"
    num_questions: int = 10
    detail_level: str = "medium"
    additional_requirements: str = ""
