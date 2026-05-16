from __future__ import annotations

from typing import List, Optional

from app.services.proxy_client import proxy_completion
from .config import QuizGenerationConfig
from .llm_parser import parse_questions_from_json
from .prompts import GENERATION_SYSTEM_PROMPT, build_user_prompt
from ..models import Question


async def generate_quiz_from_text(
        note_text: str,
        cfg: Optional[QuizGenerationConfig] = None,
        theme_name: Optional[str] = None,
        existing_question_texts: Optional[List[str]] = None,
) -> List[Question]:
    """
    Главная функция генерации квиза.

    note_text — текст конспекта (можно уже очищенный от markdown),
    cfg       — настройки типов и количества вопросов.
    """
    if cfg is None:
        cfg = QuizGenerationConfig()

    user_prompt = build_user_prompt(
        cfg,
        theme_name=theme_name,
        existing_question_texts=existing_question_texts,
    )

    raw_answer, _ = await proxy_completion(
        text=note_text,
        user_prompt=user_prompt,
        system_prompt=GENERATION_SYSTEM_PROMPT,
    )

    return parse_questions_from_json(raw_answer)
