from __future__ import annotations

from typing import List

from app.quiz.contracts.models import GeneratedQuestionBundle, VariantModel
from app.quiz.models import AnswerVariant, MatchingPair, QuestionType, QuizQuestion

CONTRACT_TO_INTERNAL_TYPE: dict[str, QuestionType] = {
    "truefalse": "true_false",
    "multichoice": "multiple_choice",
    "single_choice": "multiple_choice",
    "matching": "matching",
    "shortanswer": "short_answer",
    "essay": "long_answer",
    "open": "long_answer",
    "numerical": "fill_in_the_blank",
}


def _contract_type_to_internal(type_name: str) -> QuestionType:
    return CONTRACT_TO_INTERNAL_TYPE.get(type_name, "multiple_choice")


def _matching_pairs(bundle: GeneratedQuestionBundle) -> list[MatchingPair] | None:
    pairs_from_variants = [
        MatchingPair(left_option=v.leftMatching, right_option=v.rightMatching)
        for v in bundle.variants
        if v.leftMatching and v.rightMatching
    ]
    if pairs_from_variants:
        return pairs_from_variants

    if not bundle.matchingConfig:
        return None

    left_by_id = {item.id: item.text for item in bundle.matchingConfig.leftItems}
    right_by_id = {item.id: item.text for item in bundle.matchingConfig.rightItems}
    pairs: list[MatchingPair] = []

    for pair in bundle.matchingConfig.correctPairs:
        left = left_by_id.get(pair.leftVariantId)
        right = right_by_id.get(pair.rightVariantId)
        if left and right:
            pairs.append(MatchingPair(left_option=left, right_option=right))

    return pairs or None


def contract_to_internal(contracts: List[GeneratedQuestionBundle]) -> List[QuizQuestion]:
    questions: list[QuizQuestion] = []

    for bundle in contracts:
        q = bundle.question
        q_type = _contract_type_to_internal(q.type)
        variants = [
            AnswerVariant(
                id=str(v.id),
                text=v.text,
                is_correct=v.isRight,
                explanation="",
            )
            for v in bundle.variants
        ]

        matching_pairs = _matching_pairs(bundle)
        correct_answer = None
        if q_type in {"short_answer", "long_answer", "fill_in_the_blank"}:
            correct = [v.text for v in bundle.variants if v.isRight]
            correct_answer = correct if len(correct) > 1 else (correct[0] if correct else None)
        elif q_type == "matching" and matching_pairs:
            correct_answer = [
                f"{pair.left_option} -> {pair.right_option}"
                for pair in matching_pairs
            ]

        questions.append(
            QuizQuestion(
                id=q.id,
                text=q.text,
                type=q_type,
                variants=None if q_type == "matching" else (variants or None),
                correct_answer=correct_answer,
                matching_pairs=matching_pairs,
                general_explanation="",
            )
        )

    return questions


def _variant_explanation(variant: AnswerVariant | dict) -> tuple[bool, str]:
    if isinstance(variant, dict):
        return bool(variant.get("is_correct", False)), str(variant.get("explanation", ""))
    return bool(variant.is_correct), variant.explanation


def internal_to_contract(
        q_internal: List[QuizQuestion],
        source: List[GeneratedQuestionBundle],
) -> List[GeneratedQuestionBundle]:
    out: list[GeneratedQuestionBundle] = []

    for internal, bundle in zip(q_internal, source):
        internal_variants = internal.variants or []
        updated_variants: list[VariantModel] = []

        for old_v, new_v in zip(bundle.variants, internal_variants):
            is_correct, explanation = _variant_explanation(new_v)
            old_v.explainRight = explanation if is_correct else ""
            old_v.explainWrong = "" if is_correct else explanation
            updated_variants.append(old_v)

        if updated_variants:
            bundle.variants = updated_variants
        out.append(bundle)

    return out
