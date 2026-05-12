from __future__ import annotations

from typing import List

from app.quiz.models import GeneratedQuiz, UserAnswer, CheckResponse, CheckResponseItem, QuizQuestion
from app.quiz.rag import SimpleVectorStore
from app.services.proxy_client import proxy_completion

_DOCUMENT_STORE: dict[str, SimpleVectorStore] = {}


async def check_quiz_answers(
        quiz: GeneratedQuiz,
        user_answers: List[UserAnswer],
        source_text: str,
        document_id: str = "default"
) -> CheckResponse:
    store = _DOCUMENT_STORE.get(document_id)
    if store is None:
        store = SimpleVectorStore()
        _DOCUMENT_STORE[document_id] = store
    if not store.chunks:
        await store.add_document(source_text)

    results: List[CheckResponseItem] = []

    for ua in user_answers:
        if ua.question_index >= len(quiz.questions):
            continue
        q = quiz.questions[ua.question_index]

        is_correct = False
        correct_option_ids: List[str] = [v.id for v in (q.variants or []) if v.is_correct]

        if q.variants:  # С вариантами
            selected_ids = {str(item) for item in (ua.selected_option_ids or [])}
            correct_set = set(correct_option_ids)
            is_correct = bool(selected_ids) and selected_ids == correct_set
        elif ua.text_answer and q.correct_answer:  # Open-ended
            normalized_user = ua.text_answer.strip().lower()
            normalized_correct = str(q.correct_answer).strip().lower()
            is_correct = normalized_user in normalized_correct or normalized_correct in normalized_user

        # Explanation: сначала предгенерированное, если нет — динамика
        explanation = ""
        if is_correct and q.general_explanation:  # Для open-ended
            explanation = q.general_explanation
        elif q.variants:
            if is_correct:
                correct_vars = [v for v in q.variants if v.is_correct]
                explanation = " ".join(v.explanation for v in correct_vars) if correct_vars else ""
            else:
                selected_vars = [v for v in q.variants if v.id in (ua.selected_option_ids or [])]
                explanation = " ".join(v.explanation for v in selected_vars) if selected_vars else ""

        if not explanation:  # Fallback: динамическая генерация
            context_chunks = await store.search(q.text + " " + str(q.correct_answer), top_k=5)
            context = "\n\n".join(context_chunks)
            prompt = _build_check_prompt(q, ua, is_correct, context)
            explanation, _ = await proxy_completion(
                text="",
                user_prompt=prompt,
                system_prompt="Ты — доброжелательный преподаватель.",
                temperature=0.3,
                max_tokens=200,
            )

        results.append(CheckResponseItem(
            question_index=ua.question_index,
            is_correct=is_correct,
            explanation=explanation.strip() or "Нет объяснения.",
            correct_options=correct_option_ids,
        ))

    total_correct = sum(1 for r in results if r.is_correct)
    return CheckResponse(
        results=results,
        total_correct=total_correct,
        total_questions=len(quiz.questions)
    )


def _build_check_prompt(q: QuizQuestion, ua: UserAnswer, is_correct: bool, context: str) -> str:
    wrong_text = ua.text_answer or ", ".join(ua.selected_option_ids or [])
    return f"""
        Вопрос: {q.text}
        Пользователь ответил: {wrong_text}
        Правильно? {is_correct}
        Правильный: {q.correct_answer}
        
        Контекст: {context}
        
        {"Похвали и объясни почему верно." if is_correct else "Объясни ошибку, укажи правильный и почему."}
        На русском, 2-4 предложения.
        """
