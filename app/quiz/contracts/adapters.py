from __future__ import annotations

from typing import Iterable, List

from app.quiz.models import (
    Question,
    TrueFalseQuestion,
    MultipleChoiceQuestion,
    SelectAllThatApplyQuestion,
    FillInTheBlankQuestion,
    MatchingQuestion,
    ShortOrLongAnswerQuestion,
)
from .builders import (
    build_truefalse_question,
    build_multichoice_question,
    build_matching_question,
    build_text_question,
)
from .models import GeneratedQuestionBundle


def question_to_bundle(q: Question) -> GeneratedQuestionBundle:
    """
    Адаптер: наши внутренние типы (как в Obsidian-плагине)
    → контрактный GeneratedQuestionBundle (QuestionModel + VariantModel + MatchingConfig).
    """

    # True / False
    if isinstance(q, TrueFalse):
        return build_truefalse_question(
            text=q.question,
            correct_answer=bool(q.answer),
        )

    # Обычный multiple choice (один правильный)
    if isinstance(q, MultipleChoice):
        return build_multichoice_question(
            text=q.question,
            options=list(q.options),
            correct_indices=[int(q.answer)],
        )

    # Select all that apply -> multichoice с multiAnswer=True
    if isinstance(q, SelectAllThatApply):
        return build_multichoice_question(
            text=q.question,
            options=list(q.options),
            correct_indices=list(q.answer),
            multi_answer=True,
        )

    # Fill in the blank
    # Внутри нашего типа: answer: list[str]
    # В контракте: shortanswer с несколькими эталонными вариантами.
    if isinstance(q, FillInTheBlank):
        return build_text_question(
            text=q.question,
            correct_answers=list(q.answer),
            q_type="shortanswer",
        )

    # Matching
    if isinstance(q, Matching):
        pairs = [(p.leftOption, p.rightOption) for p in q.answer]
        return build_matching_question(
            text=q.question,
            pairs=pairs,
        )

    # Short / Long answer
    if isinstance(q, ShortOrLongAnswer):
        # определяем тип по длине ответа (как и раньше)
        q_type = "shortanswer" if len(q.answer) < 250 else "essay"
        return build_text_question(
            text=q.question,
            correct_answers=[q.answer],
            q_type=q_type,
        )

    raise TypeError(f"Неизвестный тип вопроса: {type(q)}")


def questions_to_bundles(questions: Iterable[Question]) -> List[GeneratedQuestionBundle]:
    return [question_to_bundle(q) for q in questions]
