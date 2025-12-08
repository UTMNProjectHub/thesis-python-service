from __future__ import annotations

from typing import List, Optional

from app.services.proxy_client import proxy_completion
from ..models import QuizQuestion
from .config import QuizGenerationConfig
from .prompts import GENERATION_SYSTEM_PROMPT, build_user_prompt
from .llm_parser import parse_questions_from_json


async def generate_quiz_from_text(
        note_text: str,
        cfg: Optional[QuizGenerationConfig] = None,
) -> List[QuizQuestion]:
    """
    Главная функция генерации квиза.

    note_text — текст конспекта (можно уже очищенный от markdown),
    cfg       — настройки типов и количества вопросов.
    """
    if cfg is None:
        cfg = QuizGenerationConfig()

    user_prompt = build_user_prompt(cfg)

    # Твой proxy_client: text = исходный конспект, user_prompt = инструкция по генерации
    raw_answer, _ = await proxy_completion(
        text=note_text,
        user_prompt=user_prompt,
        system_prompt=GENERATION_SYSTEM_PROMPT,
    )

    questions = parse_questions_from_json(raw_answer)
    return questions
