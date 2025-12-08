from __future__ import annotations

import json
from typing import List

from ..models import (
    Question,
    TrueFalseQuestion,
    MultipleChoiceQuestion,
    SelectAllThatApplyQuestion,
    FillInTheBlankQuestion,
    MatchingPair,
    MatchingQuestion,
    ShortOrLongAnswerQuestion,
)


def _strip_code_fence(text: str) -> str:
    """
    Убираем ```json ... ``` вокруг ответа, если модель так ответила.
    """
    t = text.strip()
    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    # первая строка: ``` или ```json
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # последняя строка: ```
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_questions_from_json(raw: str) -> List[Question]:
    """
    Разбор JSON-ответа модели в список наших Question-объектов.

    Мы ожидаем либо объект {"questions": [...]}, либо просто массив вопросов.
    Каждый вопрос — dict с полями question/options/answer.
    """
    cleaned = _strip_code_fence(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    else:
        return []

    questions: List[Question] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        question_text = str(item.get("question", "")).strip()
        if not question_text:
            continue

        answer = item.get("answer")

        # True/False
        if isinstance(answer, bool):
            questions.append(TrueFalseQuestion(question=question_text, answer=answer))
            continue

        # Multiple choice (один индекс)
        if isinstance(answer, int):
            options = [str(o) for o in item.get("options", [])]
            if not options:
                continue
            questions.append(
                MultipleChoiceQuestion(
                    question=question_text,
                    options=options,
                    answer=answer,
                )
            )
            continue

        # Массив
        if isinstance(answer, list) and answer:
            # Select all that apply: массив индексов
            if all(isinstance(x, int) for x in answer):
                options = [str(o) for o in item.get("options", [])]
                if not options:
                    continue
                questions.append(
                    SelectAllThatApplyQuestion(
                        question=question_text,
                        options=options,
                        answer=answer,
                    )
                )
                continue

            # Fill in the blank: массив строк
            if all(isinstance(x, str) for x in answer):
                questions.append(
                    FillInTheBlankQuestion(
                        question=question_text,
                        answer=[str(x) for x in answer],
                    )
                )
                continue

            # Matching: массив объектов {leftOption, rightOption}
            if all(
                    isinstance(x, dict)
                    and "leftOption" in x
                    and "rightOption" in x
                    for x in answer
            ):
                pairs: List[MatchingPair] = []
                for pair in answer:
                    pairs.append(
                        MatchingPair(
                            left_option=str(pair["leftOption"]),
                            right_option=str(pair["rightOption"]),
                        )
                    )
                questions.append(
                    MatchingQuestion(
                        question=question_text,
                        answer=pairs,
                    )
                )
                continue

        # Всё остальное считаем Short/Long answer
        if isinstance(answer, str):
            questions.append(
                ShortOrLongAnswerQuestion(
                    question=question_text,
                    answer=answer,
                )
            )

    return questions
