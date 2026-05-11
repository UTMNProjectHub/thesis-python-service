from __future__ import annotations

import asyncio
from typing import List
from uuid import UUID, uuid4

from app.quiz.explainer import generate_explanations
from app.quiz.generation.config import QuizGenerationConfig
from app.quiz.generation.service import generate_quiz_from_text
from app.quiz.models import (
    AnswerVariant,
    FillInTheBlankQuestion,
    GeneratedQuiz,
    MatchingPair,
    MatchingQuestion,
    MultipleChoiceQuestion,
    Question,
    QuizQuestion,
    SelectAllThatApplyQuestion,
    ShortOrLongAnswerQuestion,
    TrueFalseQuestion,
)
from app.quiz.rag import SimpleVectorStore

UI_TYPE_TO_CONFIG_FIELD = {
    "truefalse": "true_false",
    "true_false": "true_false",
    "multichoice": "multiple_choice",
    "multiple_choice": "multiple_choice",
    "matching": "matching",
    "shortanswer": "short_answer",
    "short_answer": "short_answer",
    "essay": "long_answer",
    "long_answer": "long_answer",
    "numerical": "fill_in_the_blank",
    "fill_in_the_blank": "fill_in_the_blank",
}


def _build_generation_config(question_count: int, question_types: List[str]) -> QuizGenerationConfig:
    types = [UI_TYPE_TO_CONFIG_FIELD.get(t.lower()) for t in question_types]
    types = [t for t in types if t]
    if not types:
        types = ["multiple_choice"]

    counts = {t: 0 for t in set(types)}
    for idx in range(max(question_count, 0)):
        counts[types[idx % len(types)]] += 1

    return QuizGenerationConfig(
        generate_true_false=counts.get("true_false", 0) > 0,
        num_true_false=counts.get("true_false", 0),
        generate_multiple_choice=counts.get("multiple_choice", 0) > 0,
        num_multiple_choice=counts.get("multiple_choice", 0),
        generate_select_all_that_apply=False,
        num_select_all_that_apply=0,
        generate_fill_in_the_blank=counts.get("fill_in_the_blank", 0) > 0,
        num_fill_in_the_blank=counts.get("fill_in_the_blank", 0),
        generate_matching=counts.get("matching", 0) > 0,
        num_matching=counts.get("matching", 0),
        generate_short_answer=counts.get("short_answer", 0) > 0,
        num_short_answer=counts.get("short_answer", 0),
        generate_long_answer=counts.get("long_answer", 0) > 0,
        num_long_answer=counts.get("long_answer", 0),
    )


def _raw_to_quiz_question(raw: Question) -> QuizQuestion | None:
    if isinstance(raw, TrueFalseQuestion):
        variants = [
            AnswerVariant(id="True", text="True", is_correct=raw.answer),
            AnswerVariant(id="False", text="False", is_correct=not raw.answer),
        ]
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="true_false",
            variants=variants,
            correct_answer="True" if raw.answer else "False",
        )

    if isinstance(raw, MultipleChoiceQuestion):
        if raw.answer < 0 or raw.answer >= len(raw.options):
            return None
        variants = [
            AnswerVariant(id=chr(65 + idx), text=option, is_correct=idx == raw.answer)
            for idx, option in enumerate(raw.options)
        ]
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="multiple_choice",
            variants=variants,
            correct_answer=raw.options[raw.answer],
        )

    if isinstance(raw, SelectAllThatApplyQuestion):
        correct = set(raw.answer)
        variants = [
            AnswerVariant(id=chr(65 + idx), text=option, is_correct=idx in correct)
            for idx, option in enumerate(raw.options)
        ]
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="select_all_that_apply",
            variants=variants,
            correct_answer=[raw.options[idx] for idx in raw.answer if 0 <= idx < len(raw.options)],
        )

    if isinstance(raw, FillInTheBlankQuestion):
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="fill_in_the_blank",
            correct_answer=raw.answer,
        )

    if isinstance(raw, MatchingQuestion):
        pairs = [MatchingPair(left_option=p.left_option, right_option=p.right_option) for p in raw.answer]
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="matching",
            matching_pairs=pairs,
            correct_answer=[f"{p.left_option} -> {p.right_option}" for p in raw.answer],
        )

    if isinstance(raw, ShortOrLongAnswerQuestion):
        return QuizQuestion(
            id=uuid4(),
            text=raw.question,
            type="short_answer" if len(raw.answer) < 250 else "long_answer",
            correct_answer=raw.answer,
        )

    return None


async def generate_quiz(
        quiz_id: UUID,
        file_contents: List[str],
        difficulty: str,
        question_count: int,
        question_types: List[str],
        additional_requirements: str | None = None,
) -> GeneratedQuiz:
    full_text = "\n\n".join(file_contents)
    if additional_requirements:
        full_text = f"{full_text}\n\nAdditional requirements:\n{additional_requirements}"
    if not full_text.strip():
        raise ValueError("Empty document text")

    cfg = _build_generation_config(question_count, question_types)
    raw_questions = await generate_quiz_from_text(full_text, cfg=cfg)
    questions = [q for q in (_raw_to_quiz_question(raw) for raw in raw_questions) if q is not None]

    rag_store = SimpleVectorStore()
    await rag_store.add_document(full_text)

    batches = [questions[i:i + 5] for i in range(0, len(questions), 5)]
    for batch in batches:
        tasks = []
        for q in batch:
            context_chunks = await rag_store.search(q.text, top_k=6)
            tasks.append(generate_explanations(q, context_chunks, difficulty))
        await asyncio.gather(*tasks)

    return GeneratedQuiz(
        quiz_id=quiz_id or uuid4(),
        title=additional_requirements or "Generated quiz",
        difficulty=difficulty,
        questions=questions,
    )
