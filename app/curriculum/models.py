from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    """
    Уровень сложности конспекта/лекции.

    Интерпретация:
      - EASY   — доступно для первокурсника/непрофильной аудитории;
      - MEDIUM — стандартный уровень для студентов курса;
      - HARD   — углублённый, с формулами, строгими определениями и т.п.
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class LectureTopic(BaseModel):
    """
    Отдельная тема лекции/занятия.

    id              — внутренний идентификатор (slug).
    title           — название темы (как в РПД/МОК).
    description     — краткое описание / формулировка результата обучения.
    difficulty      — уровень сложности из DifficultyLevel.
    keywords        — ключевые слова для поиска по документам.
    duration_min    — ориентировочная длительность темы (в минутах).
    source_docs     — список id документов (из app.documents), которые считаются основными.
    order           — номер темы в курсе (для сортировки).
    """
    id: str = Field(..., description="Уникальный идентификатор темы (slug)")
    title: str = Field(..., description="Название темы")
    description: str = Field("", description="Краткое описание темы")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Уровень сложности материала",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Ключевые слова для поиска релевантных фрагментов в документах",
    )
    duration_min: Optional[int] = Field(
        default=None,
        description="Ориентировочная длительность темы (в минутах)",
    )
    source_docs: List[str] = Field(
        default_factory=list,
        description="Идентификаторы документов (Document.id), связанных с темой",
    )
    order: Optional[int] = Field(
        default=None,
        description="Порядковый номер темы в курсе",
    )


class Curriculum(BaseModel):
    """
    Учебный план/модуль по одной дисциплине.

    course_id       — шифр дисциплины (как в РПД).
    course_name     — полное название.
    description     — краткое описание курса.
    topics          — список тем (LectureTopic).
    """
    course_id: str = Field(..., description="Шифр дисциплины, например 'DS101'")
    course_name: str = Field(..., description="Название дисциплины")
    description: str = Field("", description="Описание курса")
    topics: List[LectureTopic] = Field(
        default_factory=list,
        description="Список тем лекций/занятий",
    )

    def sorted_topics(self) -> List[LectureTopic]:
        """
        Возвращает темы в порядке:
          - сначала по полю order (если указано),
          - затем по title.
        """
        return sorted(
            self.topics,
            key=lambda t: (t.order if t.order is not None else 10_000, t.title.lower()),
        )
