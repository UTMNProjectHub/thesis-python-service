from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.curriculum.models import DifficultyLevel


class SectionKind(str, Enum):
    """
    Тип секции лекции. Используется как мягкий ориентир для генерации текста.
    """
    INTRO = "intro"          # Введение, постановка контекста
    THEORY = "theory"        # Теория, определения, формулы
    EXAMPLES = "examples"    # Примеры, разбор кейсов
    PRACTICE = "practice"    # Практика, задачи, упражнения
    SUMMARY = "summary"      # Итоги, выводы, повторение
    OTHER = "other"          # Другое / смешанный тип


class LectureSection(BaseModel):
    """
    Одна секция (глава) лекции в плане.

    id            — внутренний идентификатор (slug вида 'intro_pgvector').
    title         — заголовок секции.
    kind          — тип секции (введение, теория, примеры, и т.п.).
    order         — порядок следования в лекции (1..N).
    difficulty    — целевой уровень сложности подачи.
    summary       — короткое текстовое описание (2–4 предложения).
    key_points    — список ключевых тезисов, которые нужно раскрыть.
    sources       — список идентификаторов фрагментов (F1, F2, ...) из контекста,
                    на которые секция преимущественно опирается.
                    Это чисто логические ссылки; сами фрагменты живут на стороне RAG.
    """
    id: str = Field(..., description="Уникальный идентификатор секции (slug)")
    title: str = Field(..., description="Заголовок секции")
    kind: SectionKind = Field(default=SectionKind.OTHER, description="Тип секции")
    order: int = Field(..., description="Порядок секции в лекции (1..N)")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Целевой уровень сложности для этой секции",
    )
    summary: str = Field(
        default="",
        description="Краткое описание смысла секции",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Ключевые тезисы, которые необходимо раскрыть в секции",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Список идентификаторов фрагментов контекста (F1, F2, ...)",
    )


class LecturePlan(BaseModel):
    """
    План лекции по одной теме.

    topic_id       — id темы (LectureTopic.id).
    topic_title    — заголовок темы.
    difficulty     — общий уровень сложности.
    sections       — ordered list секций.
    """
    topic_id: str = Field(..., description="Идентификатор темы (LectureTopic.id)")
    topic_title: str = Field(..., description="Заголовок темы")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Общий уровень сложности лекции",
    )
    sections: List[LectureSection] = Field(
        default_factory=list,
        description="Список секций (глав) лекции в порядке прохождения",
    )

    def sorted_sections(self) -> List[LectureSection]:
        return sorted(self.sections, key=lambda s: s.order)
