from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from app.curriculum.models import Curriculum, LectureTopic, DifficultyLevel
from app.documents.pdf_reader import load_pdf_document


def _slugify(text: str) -> str:
    """
    Грубый slug для id темы:
      - нижний регистр,
      - заменяем пробелы на '_',
      - убираем всё, кроме букв/цифр/подчёркиваний.
    """
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-zA-Zа-яёА-ЯЁ_]", "", text)
    return text or "topic"


def _extract_topics_from_text(full_text: str) -> List[LectureTopic]:
    """
    Очень упрощённый парсер РПД/МОК.

    Ищет строки вида:
      - 'Тема 1. Введение в ...'
      - 'Тема 2: ...'
      - 'Тема: Введение в ...'

    Всё, что после 'Тема ...' до конца строки — считаем названием темы.
    Описание и сложность пока не вытаскиваем автоматически.
    """
    topics: List[LectureTopic] = []

    text = full_text.replace("\r\n", "\n")

    lines = [ln.strip() for ln in text.split("\n")]

    patterns = [
        re.compile(r"^Тема\s+(\d+)[\.\:]\s*(.+)$", re.IGNORECASE),
        re.compile(r"^Тема\s*[:\-]\s*(.+)$", re.IGNORECASE),
    ]

    order_counter = 1

    for line in lines:
        if not line:
            continue

        matched_title: Optional[str] = None

        for pat in patterns:
            m = pat.match(line)
            if not m:
                continue

            # Вариант: 'Тема 1. Название'
            if len(m.groups()) == 2:
                _, title = m.groups()
                matched_title = title.strip()
            # Вариант: 'Тема: Название'
            elif len(m.groups()) == 1:
                (title,) = m.groups()
                matched_title = title.strip()

            if matched_title:
                break

        if not matched_title:
            continue

        topic_id = _slugify(matched_title)
        topic = LectureTopic(
            id=topic_id,
            title=matched_title,
            description="",
            difficulty=DifficultyLevel.MEDIUM,
            keywords=[],
            duration_min=None,
            source_docs=[],
            order=order_counter,
        )
        topics.append(topic)
        order_counter += 1

    unique_by_id: dict[str, LectureTopic] = {}
    for t in topics:
        if t.id not in unique_by_id:
            unique_by_id[t.id] = t

    return list(unique_by_id.values())


def parse_rpd_pdf_to_curriculum(
        path: str | Path,
        course_id: str,
        course_name: str,
        description: str = "",
) -> Curriculum:
    """
    Пробует разобрать РПД/МОК в PDF и построить Curriculum.

    Логика:
      1. Загружаем PDF как документ.
      2. Склеиваем все страницы в один большой текст.
      3. Ищем темы по шаблонам 'Тема ...'.
      4. Собираем Curriculum с упорядоченным списком LectureTopic.

    Это простой парсер-черновик, который можно улучшать:
      - привязать темы к страницам,
      - вытягивать результаты обучения,
      - определять сложность и т.п.
    """
    doc, pages_text = load_pdf_document(path)
    full_text = "\n".join(pages_text)
    topics = _extract_topics_from_text(full_text)

    curriculum = Curriculum(
        course_id=course_id,
        course_name=course_name or doc.title,
        description=description,
        topics=topics,
    )
    return curriculum
