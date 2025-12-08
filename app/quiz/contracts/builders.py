from __future__ import annotations

from typing import List, Iterable, Optional
from uuid import UUID

from .models import (
    QuestionModel,
    VariantModel,
    MatchingLeftItemModel,
    MatchingRightItemModel,
    MatchingCorrectPairModel,
    MatchingConfigModel,
    GeneratedQuestionBundle,
    QuestionType,
    new_uuid,
)


# === MULTICHOICE / SELECT-ALL-THAT-APPLY ===

def build_multichoice_question(
        text: str,
        options: List[str],
        correct_indices: List[int],
        *,
        multi_answer: bool | None = None,
        question_id: Optional[UUID] = None,
        explain_right_default: str = "Верный вариант.",
        explain_wrong_default: str = "Неверный вариант.",
) -> GeneratedQuestionBundle:
    """
    Вопрос типа multichoice (один или несколько правильных).
    correct_indices — индексы правильных вариантов в options.
    """
    q_id = question_id or new_uuid()

    if multi_answer is None:
        multi_answer = len(correct_indices) > 1

    variants: List[VariantModel] = []
    for idx, option_text in enumerate(options):
        base_var_id = new_uuid()
        variants.append(
            VariantModel(
                id=new_uuid(),
                text=option_text,
                explainRight=explain_right_default,
                explainWrong=explain_wrong_default,
                isRight=idx in correct_indices,
                questionId=q_id,
                variantId=base_var_id,
                questionsVariantsId=new_uuid(),
            )
        )

    question = QuestionModel(
        id=q_id,
        type="multichoice",
        multiAnswer=multi_answer,
        text=text,
    )

    return GeneratedQuestionBundle(
        question=question,
        variants=variants,
        matchingConfig=None,
        questionType="multichoice",
    )


# === TRUE / FALSE ===

def build_truefalse_question(
        text: str,
        correct_answer: bool,
        *,
        question_id: Optional[UUID] = None,
        explain_true_right: str = "Утверждение верно.",
        explain_true_wrong: str = "Утверждение неверно.",
        explain_false_right: str = "Утверждение верно.",
        explain_false_wrong: str = "Утверждение неверно.",
) -> GeneratedQuestionBundle:
    """
    True/False реализуем как два варианта.
    """
    q_id = question_id or new_uuid()

    variants: List[VariantModel] = []

    # True
    true_var_base = new_uuid()
    variants.append(
        VariantModel(
            id=new_uuid(),
            text="True",
            explainRight=explain_true_right,
            explainWrong=explain_true_wrong,
            isRight=correct_answer is True,
            questionId=q_id,
            variantId=true_var_base,
            questionsVariantsId=new_uuid(),
        )
    )

    # False
    false_var_base = new_uuid()
    variants.append(
        VariantModel(
            id=new_uuid(),
            text="False",
            explainRight=explain_false_right,
            explainWrong=explain_false_wrong,
            isRight=correct_answer is False,
            questionId=q_id,
            variantId=false_var_base,
            questionsVariantsId=new_uuid(),
        )
    )

    question = QuestionModel(
        id=q_id,
        type="truefalse",
        multiAnswer=False,
        text=text,
    )

    return GeneratedQuestionBundle(
        question=question,
        variants=variants,
        matchingConfig=None,
        questionType="truefalse",
    )


# === TEXT QUESTIONS (shortanswer / essay / numerical / description) ===

def build_text_question(
        text: str,
        correct_answers: List[str],
        *,
        question_id: Optional[UUID] = None,
        q_type: QuestionType = "shortanswer",
        explain_right_default: str = "Ответ соответствует ожидаемому.",
        explain_wrong_default: str = "Ответ отличается от ожидаемого.",
) -> GeneratedQuestionBundle:
    """
    Текстовые вопросы.

    correct_answers:
      - список эталонных ответов;
      - на их основе создаются VariantModel с isRight = True.

    Потом при проверке можно:
      - сравнивать ответ студента с этими вариантами по эмбеддингам;
      - или делать строгую проверку по строке — как захочешь.
    """
    if q_type not in ("shortanswer", "essay", "numerical", "description"):
        raise ValueError(f"Недопустимый тип текстового вопроса: {q_type}")

    q_id = question_id or new_uuid()

    # Если ответов несколько — логично считать, что multiAnswer = True.
    multi_answer: Optional[bool] = None
    if len(correct_answers) > 1:
        multi_answer = True
    elif len(correct_answers) == 1:
        multi_answer = False

    variants: List[VariantModel] = []
    for ans_text in correct_answers:
        base_id = new_uuid()
        variants.append(
            VariantModel(
                id=new_uuid(),
                text=ans_text,
                explainRight=explain_right_default,
                explainWrong=explain_wrong_default,
                isRight=True,  # это «правильный» эталонный ответ
                questionId=q_id,
                variantId=base_id,
                questionsVariantsId=new_uuid(),
            )
        )

    question = QuestionModel(
        id=q_id,
        type=q_type,
        multiAnswer=multi_answer,
        text=text,
    )

    return GeneratedQuestionBundle(
        question=question,
        variants=variants,
        matchingConfig=None,
        questionType=q_type,
    )


# === MATCHING ===

def build_matching_question(
        text: str,
        pairs: Iterable[tuple[str, str]],
        *,
        question_id: Optional[UUID] = None,
        explain_right_default: str = "Пара подобрана верно.",
        explain_wrong_default: str = "Пара подобрана неверно.",
) -> GeneratedQuestionBundle:
    """
    Matching: на вход список пар (left_text, right_text).
    Каждая пара считается корректной.
    """
    q_id = question_id or new_uuid()

    left_items: List[MatchingLeftItemModel] = []
    right_items: List[MatchingRightItemModel] = []
    correct_pairs: List[MatchingCorrectPairModel] = []

    for left_text, right_text in pairs:
        left_id = new_uuid()
        right_id = new_uuid()

        left_items.append(MatchingLeftItemModel(id=left_id, text=left_text))
        right_items.append(MatchingRightItemModel(id=right_id, text=right_text))

        correct_pairs.append(
            MatchingCorrectPairModel(
                leftVariantId=left_id,
                rightVariantId=right_id,
                explainRight=explain_right_default,
                explainWrong=explain_wrong_default,
            )
        )

    matching_config = MatchingConfigModel(
        leftItems=left_items,
        rightItems=right_items,
        correctPairs=correct_pairs,
    )

    variants: List[VariantModel] = []
    for item in left_items + right_items:
        base_id = new_uuid()
        variants.append(
            VariantModel(
                id=new_uuid(),
                text=item.text,
                explainRight=explain_right_default,
                explainWrong=explain_wrong_default,
                isRight=False,  # сама правильность хранится в correctPairs
                questionId=q_id,
                variantId=base_id,
                questionsVariantsId=new_uuid(),
            )
        )

    question = QuestionModel(
        id=q_id,
        type="matching",
        multiAnswer=None,
        text=text,
    )

    return GeneratedQuestionBundle(
        question=question,
        variants=variants,
        matchingConfig=matching_config,
        questionType="matching",
    )
