from __future__ import annotations

from typing import List, Optional

from app.curriculum.models import Curriculum, LectureTopic, DifficultyLevel


def get_topic_by_id(curriculum: Curriculum, topic_id: str) -> Optional[LectureTopic]:
    """
    Возвращает тему по её id или None, если не найдено.
    """
    for topic in curriculum.topics:
        if topic.id == topic_id:
            return topic
    return None


def list_topics_by_difficulty(
        curriculum: Curriculum,
        difficulty: DifficultyLevel,
) -> List[LectureTopic]:
    """
    Фильтрация тем по уровню сложности.
    """
    return [t for t in curriculum.topics if t.difficulty == difficulty]


def search_topics(
        curriculum: Curriculum,
        query: str,
        max_results: int = 20,
) -> List[LectureTopic]:
    """
    Простой поиск тем по подстроке в title/description/keywords.

    В будущем можно заменить на что-то более умное (BM25, эмбеддинги и т.д.),
    но для начала этого достаточно.
    """
    q = query.lower().strip()
    if not q:
        return []

    results: List[LectureTopic] = []

    for topic in curriculum.sorted_topics():
        haystack = " ".join([
            topic.title,
            topic.description or "",
            " ".join(topic.keywords or []),
        ]).lower()

        if q in haystack:
            results.append(topic)
            if len(results) >= max_results:
                break

    return results
