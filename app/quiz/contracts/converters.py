from uuid import uuid4
from typing import List
from app.quiz.models import QuizQuestion
from app.quiz.contracts.models import GeneratedQuestionBundle, VariantModel, MatchingConfigModel

# CONTRACT -> INTERNAL
def contract_to_internal(contracts: List[GeneratedQuestionBundle]) -> List[QuizQuestion]:
    res = []

    for bundle in contracts:
        q = bundle.question

        # Варианты
        variants_data = []
        for v in bundle.variants:
            variants_data.append({
                "id": str(v.id),
                "text": v.text,
                "is_correct": v.isRight,
                "explanation": ""
            })

        q_internal = QuizQuestion(
            id=q.id,
            text=q.text,
            type=q.type,
            variants=variants_data,
            correct_answer=None,
            matching_pairs=None,
            general_explanation=""
        )
        res.append(q_internal)

    return res


# INTERNAL -> CONTRACT
def internal_to_contract(q_internal: List[QuizQuestion],
                         source: List[GeneratedQuestionBundle]) -> List[GeneratedQuestionBundle]:

    out = []

    for internal, bundle in zip(q_internal, source):

        new_variants = []
        for old_v, new_v in zip(bundle.variants, internal.variants):
            old_v.explainRight = new_v.explanation if new_v.is_correct else ""
            old_v.explainWrong = "" if new_v.is_correct else new_v.explanation
            new_variants.append(old_v)

        bundle.variants = new_variants
        out.append(bundle)

    return out
