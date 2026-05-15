from __future__ import annotations

import json
import re
from typing import List, Sequence, Tuple

from app.curriculum.models import LectureTopic, DifficultyLevel
from app.documents.chunking import estimate_tokens
from app.documents.indexers.base import BaseRetriever
from app.documents.models import DocumentChunk
from app.lectures.context_selection import DocumentProfile, select_balanced_chunks
from app.lectures.models import LecturePlan, LectureSection, SectionKind
from app.services.proxy_client import proxy_completion


def _difficulty_comment(level: DifficultyLevel) -> str:
    """
    Текст-пояснение для LLM, как трактовать уровень сложности.
    """
    if level == DifficultyLevel.EASY:
        return (
            "Пиши максимально понятным языком для начинающих студентов: "
            "объясняй термины, больше интуитивных описаний, минимум формализма."
        )
    if level == DifficultyLevel.HARD:
        return (
            "Пиши для продвинутых студентов: допускается строгая терминология, "
            "формулы, ссылки на сложные концепции, менее подробное разжёвывание базовых вещей."
        )
    # MEDIUM
    return (
        "Пиши для среднего уровня: аккуратные определения, примеры, "
        "но без чрезмерного упрощения."
    )


def _build_context_from_chunks(
        chunks: List[Tuple[DocumentChunk, float]],
        max_chars_per_chunk: int | None = None,
        token_budget: int | None = None,
) -> str:
    """
    Формирует текстовый контекст для LLM из релевантных чанков.

    Каждый фрагмент получает идентификатор F1, F2, ...,
    чтобы модель могла ссылаться на них в поле `sources`.
    """
    lines: List[str] = []
    used_tokens = 0
    for idx, (chunk, score) in enumerate(chunks, start=1):
        frag_id = f"F{idx}"
        text = chunk.text.strip().replace("\r\n", "\n")
        chunk_tokens = estimate_tokens(text)
        if token_budget is not None and lines and used_tokens + chunk_tokens > token_budget:
            break
        used_tokens += chunk_tokens
        if max_chars_per_chunk is not None and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " [...]"

        header = (
            f"### Фрагмент {frag_id}\n"
            f"(документ={chunk.doc_id}, страницы {chunk.page_start}-{chunk.page_end}, "
            f"score={score:.3f})\n"
        )
        lines.append(header)
        lines.append(text)
        lines.append("")  # пустая строка-разделитель

    return "\n".join(lines).strip()


def _extract_json_block(raw: str) -> dict:
    """
    Пытается аккуратно вытащить JSON из ответа модели.

    Поддерживает варианты:
      - чистый JSON;
      - JSON внутри ```json ... ``` или ``` ... ```;
      - JSON с мусором до/после.

    Если не получилось — кидает ValueError.
    """
    raw = raw.strip()

    # 1. Пробуем сразу как есть
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2. Ищем блок внутри ```json ... ``` или ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3. Берём первый '{' и последний '}' и пробуем это как JSON
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Не удалось распарсить JSON из ответа модели")


async def build_lecture_plan_for_topic(
        topic: LectureTopic,
        retriever: BaseRetriever,
        *,
        top_k_chunks: int = 8,
        retrieval_pool_k: int | None = None,
        context_token_budget: int | None = None,
        document_profiles: Sequence[DocumentProfile] | None = None,
        additional_requirements: str = "",
        min_sections: int = 3,
        max_sections: int = 7,
) -> LecturePlan:
    """
    Строит план лекции по теме с использованием RAG:

      1. Формирует поисковый запрос из названия темы, описания и ключевых слов.
      2. Через retriever достаёт top_k_chunks релевантных фрагментов учебника.
      3. Отправляет фрагменты + метаданные по теме в LLM и просит
         сгенерировать план лекции (3–7 секций).
      4. Парсит JSON и возвращает LecturePlan.

    План состоит только из структуры (id, title, kind, summary, key_points, sources),
    без готового текста секций. Текст будем генерировать на следующем шаге.
    """
    # --- 1. Формируем запрос для retriever ---
    parts: List[str] = [topic.title]
    if topic.description:
        parts.append(topic.description)
    if topic.keywords:
        parts.append(" ".join(topic.keywords))

    query = " ".join(parts).strip()
    if not query:
        query = topic.title

    # --- 2. Поиск релевантных чанков ---
    chunk_results: List[Tuple[DocumentChunk, float]] = select_balanced_chunks(
        retriever,
        query,
        pool_k=retrieval_pool_k or top_k_chunks,
        limit=top_k_chunks,
        token_budget=context_token_budget,
        doc_ids=topic.source_docs,
    )

    # --- 3. Готовим контекст для модели ---
    context_text = _build_context_from_chunks(
        chunk_results,
        max_chars_per_chunk=None,
        token_budget=context_token_budget,
    ) if chunk_results else ""

    difficulty_hint = _difficulty_comment(topic.difficulty)

    system_prompt = (
        "Ты — методист и преподаватель вуза.\n"
        "Тебе даётся тема лекции, её краткое описание, уровень сложности, "
        "а также фрагменты из учебных материалов.\n"
        "Твоя задача — составить структурированный план лекции в виде JSON.\n\n"
        "Требования к плану:\n"
        f"- количество секций от {min_sections} до {max_sections};\n"
        "- секции должны идти в логическом порядке (от введения к деталям и итогам);\n"
        "- каждая секция должна быть достаточно крупной логической единицей, "
        "а не одним предложением;\n"
        "- используй фрагменты контекста как опору (но можешь переупорядочивать материал);\n"
        "- поле key_points должно содержать 3-6 конкретных тезисов секции;\n"
        "- каждый key_point должен быть самостоятельной смысловой мыслью, которую можно раскрыть в тексте лекции;\n"
        "- не заполняй key_points общими фразами вроде \"изучить тему\" или \"рассмотреть основные понятия\".\n\n"
        "Формат ответа — строго JSON:\n"
        "{"
        '  "sections": ['
        "    {"
        '      "id": "строковый_slug_секции",'
        '      "title": "Заголовок секции",'
        '      "kind": "intro|theory|examples|practice|summary|other",'
        '      "order": 1,'
        '      "summary": "2-4 предложения, кратко раскрывающие смысл секции",'
        '      "key_points": ["тезис 1", "тезис 2", "..."],'
        '      "sources": ["F1", "F3"]'
        "    },"
        "    ..."
        "  ]"
        "}"
        "\n\nНе добавляй никакого текста вне JSON.\n"
    )

    user_prompt_parts: List[str] = []

    user_prompt_parts.append("Тема лекции:")
    user_prompt_parts.append(topic.title)
    user_prompt_parts.append("")

    if topic.description:
        user_prompt_parts.append("Краткое описание темы (из РПД/МОК):")
        user_prompt_parts.append(topic.description)
        user_prompt_parts.append("")

    user_prompt_parts.append("Целевой уровень сложности:")
    user_prompt_parts.append(f"{topic.difficulty.value} — {difficulty_hint}")
    user_prompt_parts.append("")

    if topic.keywords:
        user_prompt_parts.append("Ключевые слова/понятия:")
        user_prompt_parts.append(", ".join(topic.keywords))
        user_prompt_parts.append("")

    if additional_requirements.strip():
        user_prompt_parts.append("Дополнительные требования к лекции:")
        user_prompt_parts.append(additional_requirements.strip())
        user_prompt_parts.append("")

    if document_profiles:
        user_prompt_parts.append("Краткие профили выбранных документов:")
        for profile in document_profiles:
            user_prompt_parts.append(
                f"- document={profile.doc_id}; title={profile.title}; pages={profile.pages}; chunks={profile.chunk_count}"
            )
            user_prompt_parts.append(profile.summary)
            if profile.coverage and profile.coverage != profile.summary:
                user_prompt_parts.append(profile.coverage)
        user_prompt_parts.append("")

    if context_text:
        user_prompt_parts.append(
            "Ниже приведены фрагменты из учебных материалов. "
            "Каждый имеет идентификатор вида F1, F2, ... Используй их в поле 'sources' секций."
        )
        user_prompt_parts.append("")
        user_prompt_parts.append(context_text)
    else:
        user_prompt_parts.append(
            "Контекст из учебных материалов недоступен, "
            "ориентируйся только на тему и описание."
        )

    user_prompt = "\n".join(user_prompt_parts)

    # --- 4. Запрос к LLM через proxy_completion ---
    raw_text, _ = await proxy_completion(
        text="",
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.3,  # для JSON-структуры лучше маленькая температура
        max_tokens=3000,  # даём модели договорить план до конца
    )

    # --- 5. Парсим JSON и собираем LecturePlan ---
    data = _extract_json_block(raw_text)

    if "sections" not in data or not isinstance(data["sections"], list):
        raise ValueError("LLM вернула JSON без корректного поля 'sections'")

    sections: List[LectureSection] = []
    for idx, sec_obj in enumerate(data["sections"], start=1):
        if not isinstance(sec_obj, dict):
            continue

        # Подстраховка: если order не задан, используем idx
        order = sec_obj.get("order") or idx

        raw_kind = (sec_obj.get("kind") or "other").lower().strip()
        try:
            kind = SectionKind(raw_kind)
        except ValueError:
            kind = SectionKind.OTHER

        # Слепим id, если модель не вернула
        sec_id = sec_obj.get("id") or f"section_{order}"

        # Источники — как есть (список строк-идентификаторов фрагментов)
        raw_sources = sec_obj.get("sources") or []
        if isinstance(raw_sources, list):
            sources = [str(s) for s in raw_sources]
        else:
            sources = []

        section = LectureSection(
            id=sec_id,
            title=sec_obj.get("title") or f"Секция {order}",
            kind=kind,
            order=int(order),
            difficulty=topic.difficulty,
            summary=sec_obj.get("summary") or "",
            key_points=sec_obj.get("key_points") or [],
            sources=sources,
        )
        sections.append(section)

    # На всякий случай отсортируем по order
    sections = sorted(sections, key=lambda s: s.order)

    plan = LecturePlan(
        topic_id=topic.id,
        topic_title=topic.title,
        difficulty=topic.difficulty,
        sections=sections,
    )
    return plan
