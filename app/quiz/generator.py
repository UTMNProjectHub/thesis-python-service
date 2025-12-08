from __future__ import annotations
import asyncio
from typing import List
from uuid import uuid4, UUID

from app.quiz.generation.service import generate_quiz_from_text
from app.quiz.generation.config import QuizGenerationConfig
from app.quiz.explainer import generate_explanations
from app.quiz.rag import SimpleVectorStore
from app.quiz.models import GeneratedQuiz, QuizQuestion

async def generate_quiz(
        quiz_id: UUID,
        file_contents: List[str],
        difficulty: str,
        question_count: int,
        question_types: List[str],
        additional_requirements: str | None = None,
) -> GeneratedQuiz:
    full_text = "\n\n".join(file_contents)
    if not full_text.strip():
        raise ValueError("Пустой текст документов")

    # Генерация вопросов через ваш LLM-сервис
    cfg = QuizGenerationConfig()  # Настройте по question_types и count
    # Адаптируйте cfg под input (например, num_true_false = count if "true_false" in types)
    raw_questions = await generate_quiz_from_text(full_text, cfg=cfg)

    # Конверсия в QuizQuestion (адаптируйте под вашу модель Question)
    questions: List[QuizQuestion] = []  # Логика конверсии из raw_questions

    rag_store = SimpleVectorStore()
    await rag_store.add_document(full_text)

    # Батчинг: по 5 вопросов для объяснений
    batches = [questions[i:i+5] for i in range(0, len(questions), 5)]
    for batch in batches:
        tasks = []
        for q in batch:
            context_chunks = await rag_store.search(q.text, top_k=6)
            tasks.append(generate_explanations(q, context_chunks, difficulty))
        await asyncio.gather(*tasks)

    return GeneratedQuiz(
        quiz_id=quiz_id or uuid4(),
        title=additional_requirements or "Автогенерированный тест",
        difficulty=difficulty,
        questions=questions,
    )