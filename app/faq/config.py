from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FAQGenerationConfig:
    """
    Настройки генерации FAQ.
    """
    language: str = "ru"  # Язык вопросов и ответов
    num_questions: int = 10  # Количество вопросов
    detail_level: str = "medium"  # Уровень детальности: low, medium, high
