from __future__ import annotations

from typing import List, Optional, Tuple

from app.documents.chunking import estimate_tokens
from app.documents.indexers.base import BaseRetriever
from app.documents.models import DocumentChunk
from app.lectures.context_selection import select_balanced_chunks
from app.lectures.models import LecturePlan, LectureSection
from app.lectures.planner import _build_context_from_chunks, _difficulty_comment
from app.services.proxy_client import proxy_completion

LECTURE_SECTION_SYSTEM_PROMPT = """
Ты — преподаватель и автор учебных лекций на русском языке.

Твоя задача — написать содержательную секцию лекции по заданному плану и учебным фрагментам.

Правила:
- строго придерживайся темы секции;
- раскрой все переданные ключевые тезисы;
- не удаляй и не подменяй смысл ключевых тезисов;
- используй учебные фрагменты как фактическую основу;
- если в материалах есть определения, механизмы, этапы, формулы, примеры или ограничения — включай их в объяснение;
- не добавляй внутренние source ids, file ids, page references или chunk ids;
- не пиши вводную обо всей лекции, если это не первая секция;
- не пиши заключение всей лекции, если это не последняя секция;
- пиши связно, академично, но понятно студенту;
- не создавай отдельный блок "Ключевые тезисы": он будет добавлен внешним кодом перед текстом секции.

Формат:
- Markdown;
- не используй заголовок уровня #;
- допускаются ##, ###, списки, таблицы и короткие примеры;
- секция должна быть достаточно подробной для самостоятельного изучения.
""".strip()

LECTURE_POLISH_SYSTEM_PROMPT = """
Ты — редактор русскоязычных учебных лекций.

Улучши связность, ясность и академический стиль текста, не меняя структуру лекции.

Строгие правила:
- сохрани все заголовки секций;
- сохрани каждый блок "Ключевые тезисы";
- не удаляй тезисы и не объединяй их в обычный текст;
- не удаляй важные определения, этапы, примеры и выводы;
- убирай только повторы, шероховатости и слабые формулировки;
- не добавляй source ids, page references, file ids или технические комментарии;
- верни только итоговый Markdown.
""".strip()


def _format_key_points_block(section: LectureSection) -> str:
    key_points = [point.strip() for point in section.key_points if point and point.strip()]
    if not key_points:
        return ""
    lines = ["### Ключевые тезисы", ""]
    lines.extend(f"- {point}" for point in key_points)
    return "\n".join(lines).strip()


async def generate_section_markdown(
        plan: LecturePlan,
        section: LectureSection,
        retriever: BaseRetriever,
        *,
        topic_description: Optional[str] = None,
        max_tokens: int = 900,
        top_k_chunks: int = 5,
        retrieval_pool_k: int | None = None,
        context_token_budget: int | None = None,
        additional_requirements: str = "",
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
    if additional_requirements:
        query_parts.append(additional_requirements)

    query = " ".join(query_parts).strip()

    chunk_results: List[Tuple[DocumentChunk, float]] = select_balanced_chunks(
        retriever,
        query,
        pool_k=retrieval_pool_k or top_k_chunks,
        limit=top_k_chunks,
        token_budget=context_token_budget,
    )

    context_text = _build_context_from_chunks(
        chunk_results,
        max_chars_per_chunk=None,
        token_budget=context_token_budget,
    ) if chunk_results else ""

    difficulty_hint = _difficulty_comment(plan.difficulty)

    system_prompt = LECTURE_SECTION_SYSTEM_PROMPT

    # --- 3. User prompt: описание секции + контекст ---
    parts: List[str] = []

    parts.append("Тема курса:")
    parts.append(plan.topic_title)
    parts.append("")

    parts.append("Тема секции лекции:")
    parts.append(section.title)
    parts.append("")

    sorted_sections = plan.sorted_sections()
    if sorted_sections:
        parts.append("Позиция секции в лекции:")
        if section.id == sorted_sections[0].id:
            parts.append("первая секция")
        elif section.id == sorted_sections[-1].id:
            parts.append("последняя секция")
        else:
            parts.append("средняя секция")
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

    if additional_requirements.strip():
        parts.append("Дополнительные требования к лекции:")
        parts.append(additional_requirements.strip())
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


async def polish_lecture_markdown(
        markdown: str,
        plan: LecturePlan,
        *,
        additional_requirements: str = "",
        input_token_budget: int = 50_000,
) -> str:
    if not markdown.strip():
        return markdown

    if estimate_tokens(markdown) > input_token_budget:
        sections = markdown.split("\n## ")
        if len(sections) <= 1:
            return markdown
        polished_parts = [sections[0].strip()]
        for raw_section in sections[1:]:
            section_md = "## " + raw_section.strip()
            polished_parts.append(
                await polish_lecture_markdown(
                    section_md,
                    plan,
                    additional_requirements=additional_requirements,
                    input_token_budget=input_token_budget,
                )
            )
        return "\n\n".join(part for part in polished_parts if part.strip()).strip()

    system_prompt = LECTURE_POLISH_SYSTEM_PROMPT
    user_prompt = "\n".join(
        part
        for part in [
            f"Название лекции: {plan.topic_title}",
            f"Дополнительные требования: {additional_requirements}" if additional_requirements.strip() else "",
            "Верни только отредактированный Markdown.",
        ]
        if part
    )
    max_tokens = min(12_000, max(2500, int(estimate_tokens(markdown) * 1.15)))
    edited, _ = await proxy_completion(
        text=markdown,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return edited.strip() or markdown


async def generate_lecture_markdown(
        plan: LecturePlan,
        retriever: BaseRetriever,
        *,
        topic_description: Optional[str] = None,
        max_tokens_per_section: int = 900,
        top_k_chunks_per_section: int = 5,
        retrieval_pool_k: int | None = None,
        context_token_budget: int | None = None,
        additional_requirements: str = "",
        final_edit_enabled: bool = False,
        final_edit_input_token_budget: int = 50_000,
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

        key_points_block = _format_key_points_block(section)
        if key_points_block:
            lines.append(key_points_block)
            lines.append("")

        # Генерируем текст секции
        section_md = await generate_section_markdown(
            plan=plan,
            section=section,
            retriever=retriever,
            topic_description=topic_description,
            max_tokens=max_tokens_per_section,
            top_k_chunks=top_k_chunks_per_section,
            retrieval_pool_k=retrieval_pool_k,
            context_token_budget=context_token_budget,
            additional_requirements=additional_requirements,
        )

        lines.append(section_md)
        lines.append("")  # пустая строка-разделитель

    markdown = "\n".join(lines).strip()
    if final_edit_enabled:
        return await polish_lecture_markdown(
            markdown,
            plan,
            additional_requirements=additional_requirements,
            input_token_budget=final_edit_input_token_budget,
        )
    return markdown
