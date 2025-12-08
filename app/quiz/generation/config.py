# app/quiz/generation/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuizGenerationConfig:
    """
    Настройки генерации квиза (аналог TS GenerationConfig, но в snake_case).

    language — язык формулировок вопросов и ответов (мы используем русский).
    Остальные поля задают, какие типы вопросов и в каком количестве генерировать.
    """

    language: str = "Русский"

    generate_true_false: bool = True
    num_true_false: int = 1

    generate_multiple_choice: bool = True
    num_multiple_choice: int = 1

    generate_select_all_that_apply: bool = True
    num_select_all_that_apply: int = 1

    generate_fill_in_the_blank: bool = True
    num_fill_in_the_blank: int = 1

    generate_matching: bool = True
    num_matching: int = 1

    generate_short_answer: bool = True
    num_short_answer: int = 1

    generate_long_answer: bool = True
    num_long_answer: int = 1
