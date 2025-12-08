from __future__ import annotations

from typing import List, Optional, Tuple

from app.lectures.models import LecturePlan, LectureSection
from app.documents.models import DocumentChunk
from app.documents.indexers.base import BaseRetriever
from app.lectures.planner import _build_context_from_chunks, _difficulty_comment
from app.services.proxy_client import proxy_completion


async def generate_section_markdown(
        plan: LecturePlan,
        section: LectureSection,
        retriever: BaseRetriever,
        *,
        topic_description: Optional[str] = None,
        max_tokens: int = 900,
        top_k_chunks: int = 5,
) -> str:
    """
    Генерирует текст ОДНОЙ секции лекции в формате Markdown.

    Использует:
      - тему лекции (plan.topic_title),
      - заголовок секции,
      - summary и key_points,
      - уровень сложности (plan.difficulty),
      - релевантные фрагменты из документов (через retriever).

    Возвращает чистый Markdown без заголовка # (только содержимое секции).
    """
    # --- 1. Формируем запрос к retriever для этой секции ---
    query_parts: List[str] = [plan.topic_title, section.title]
    if section.summary:
        query_parts.append(section.summary)
    if section.key_points:
        query_parts.append(" ".join(section.key_points))
    if topic_description:
        query_parts.append(topic_description)

    query = " ".join(query_parts).strip()

    chunk_results: List[Tuple[DocumentChunk, float]] = retriever.search(
        query,
        top_k=top_k_chunks,
    )

    context_text = _build_context_from_chunks(
        chunk_results,
        max_chars_per_chunk=1000,
    ) if chunk_results else ""

    difficulty_hint = _difficulty_comment(plan.difficulty)

    # --- 2. Системный промпт ---
    system_prompt = (
        "Ты — преподаватель вуза и автор конспектов лекций.\n"
        "Тебе даётся тема лекции, тема конкретной секции, её краткое резюме, "
        "ключевые тезисы и фрагменты из учебных материалов.\n"
        "Твоя задача — написать развернутый текст СЕКЦИИ лекции на русском языке "
        "в формате Markdown.\n\n"
        "Требования:\n"
        "- не пиши заголовок уровня # (это делает внешний код),\n"
        "- можно использовать подзаголовки уровней ## и ### ВНУТРИ секции при необходимости,\n"
        "- раскрывай key_points в связный текст, а не просто копируй их списком,\n"
        "- при необходимости можно добавлять списки, таблицы, примеры кода и фрагменты SQL,\n"
        "- опирайся на переданные фрагменты контекста, но переформулируй своими словами,\n"
        "- объём — ориентировочно 1–3 печатные страницы в зависимости от сложности.\n"
    )

    # --- 3. User prompt: описание секции + контекст ---
    parts: List[str] = []

    parts.append("Тема курса:")
    parts.append(plan.topic_title)
    parts.append("")

    parts.append("Тема секции лекции:")
    parts.append(section.title)
    parts.append("")

    if topic_description:
        parts.append("Краткое описание темы курса:")
        parts.append(topic_description)
        parts.append("")

    if section.summary:
        parts.append("Краткое резюме этой секции:")
        parts.append(section.summary)
        parts.append("")

    if section.key_points:
        parts.append("Ключевые тезисы, которые нужно раскрыть в этой секции:")
        for kp in section.key_points:
            parts.append(f"- {kp}")
        parts.append("")

    parts.append("Целевой уровень сложности:")
    parts.append(f"{plan.difficulty.value} — {difficulty_hint}")
    parts.append("")

    if context_text:
        parts.append(
            "Ниже приведены фрагменты из учебных материалов, уже отобранные как "
            "релевантные этой секции. Каждый фрагмент имеет идентификатор F1, F2, ...\n"
            "Используй их как источник знаний, но не копируй дословно."
        )
        parts.append("")
        parts.append(context_text)
    else:
        parts.append(
            "Контекст из учебных материалов для этой секции недоступен. "
            "Опирайся на тему и ключевые тезисы."
        )

    user_prompt = "\n".join(parts)

    # --- 4. Запрос к LLM ---
    raw_text, _ = await proxy_completion(
        text="",
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=max_tokens,
    )

    return raw_text.strip()


async def generate_lecture_markdown(
        plan: LecturePlan,
        retriever: BaseRetriever,
        *,
        topic_description: Optional[str] = None,
        max_tokens_per_section: int = 900,
        top_k_chunks_per_section: int = 5,
) -> str:
    """
    Генерирует ПОЛНЫЙ Markdown-конспект лекции по плану.

    Для каждой секции:
      - вытаскивает релевантные чанки,
      - генерирует её текст,
      - собирает всё в одну Markdown-строку.

    Структура Markdown:
      # {topic_title}
      ## {section.title}
      (опционально список ключевых тезисов)
      (сгенерированный текст секции)

    Возвращает цельный Markdown-документ.
    """
    lines: List[str] = []

    # Заголовок лекции
    lines.append(f"# {plan.topic_title}")
    lines.append("")

    for section in plan.sorted_sections():
        # Заголовок секции
        lines.append(f"## {section.title}")
        lines.append("")

        # (опционально) явно выводим ключевые тезисы перед текстом
        if section.key_points:
            lines.append("**Ключевые тезисы секции:**")
            for kp in section.key_points:
                lines.append(f"- {kp}")
            lines.append("")

        # Генерируем текст секции
        section_md = await generate_section_markdown(
            plan=plan,
            section=section,
            retriever=retriever,
            topic_description=topic_description,
            max_tokens=max_tokens_per_section,
            top_k_chunks=top_k_chunks_per_section,
        )

        lines.append(section_md)
        lines.append("")  # пустая строка-разделитель

    return "\n".join(lines).strip()
