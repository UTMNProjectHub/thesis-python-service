from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

QuestionType = Literal[
    "true_false", "multiple_choice", "select_all_that_apply",
    "fill_in_the_blank", "matching", "short_answer", "long_answer"
]


@dataclass
class AnswerVariant:
    id: str  # "A", "B", etc.
    text: str
    is_correct: bool
    explanation: str = ""  # Почему (не)верно


@dataclass
class MatchingPair:
    left_option: str
    right_option: str


class QuizQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    type: QuestionType
    variants: Optional[List[AnswerVariant]] = None
    correct_answer: Optional[str | List[str]] = None
    matching_pairs: Optional[List[MatchingPair]] = None  # Для matching
    general_explanation: str = ""  # Общее для правильного (open-ended)


class UserAnswer(BaseModel):
    question_index: int
    selected_option_ids: Optional[List[str]] = None
    text_answer: Optional[str] = None


class CheckResponseItem(BaseModel):
    question_index: int
    is_correct: bool
    explanation: str
    correct_options: List[str]


class CheckResponse(BaseModel):
    results: List[CheckResponseItem]
    total_correct: int
    total_questions: int


@dataclass
class TrueFalseQuestion:
    """Вопрос типа 'Верно/неверно'."""
    question: str
    answer: bool


@dataclass
class MultipleChoiceQuestion:
    """Один правильный ответ из нескольких вариантов."""
    question: str
    options: List[str]
    # индекс правильного варианта в options (0-based)
    answer: int


@dataclass
class SelectAllThatApplyQuestion:
    """Несколько правильных ответов из списка."""
    question: str
    options: List[str]
    # индексы правильных вариантов
    answer: List[int]


@dataclass
class FillInTheBlankQuestion:
    """Заполнить пропуски в тексте (`___`)."""
    question: str
    # правильные варианты для пропусков
    answer: List[str]


@dataclass
class MatchingQuestion:
    """Соответствия между двумя группами (A–M и N–Z в плагине)."""
    question: str
    answer: List[MatchingPair]


@dataclass
class ShortOrLongAnswerQuestion:
    """Открытый вопрос (короткий или длинный ответ — различаем по длине текста при сохранении)."""
    question: str
    answer: str


# Объединённый тип вопроса
Question = Union[
    TrueFalseQuestion,
    MultipleChoiceQuestion,
    SelectAllThatApplyQuestion,
    FillInTheBlankQuestion,
    MatchingQuestion,
    ShortOrLongAnswerQuestion,
]


@dataclass
class AnswerOption:
    id: str  # "A", "B", "C", "D"
    text: str
    is_correct: bool = False


@dataclass
class GeneratedQuiz:
    quiz_id: UUID
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    questions: List[QuizQuestion]
    source_file_ids: List[UUID] = field(default_factory=list)


@dataclass
class Quiz:
    title: str
    questions: List[QuizQuestion]
