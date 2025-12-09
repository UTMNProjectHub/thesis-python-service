from __future__ import annotations
from typing import List, Optional
import asyncio

from app.services.proxy_client import proxy_completion
from app.quiz.rag import SimpleVectorStore
from app.quiz.models import QuizQuestion, AnswerVariant, QuestionType

from uuid import uuid4
import asyncio


async def generate_explanations(
        question: QuizQuestion,
        context_chunks: List[str],
        difficulty: str = "medium"
        ) -> None:
    """
    Генерирует объяснения для вариантов или общее для open-ended.
    - Для TF: для каждого варианта.
    - Для других с вариантами: для каждого (почему верно/неверно).
    - Для open-ended: только общее для правильного.
    """
    context = "\n\n".join(context_chunks[:5]) if context_chunks else ""

    base_prompt = f"""
    Ты — преподаватель. Объясняй просто и по делу.
    Контекст из лекции:
    {context}
    
    Вопрос: {question.text}
    Тип: {question.type}
    """

    difficulty_hint = {
        "easy": "Объясняй как для новичка, с примерами.",
        "medium": "Средний уровень, без лишней сложности.",
        "hard": "Строго, с терминами, без упрощений."
    }.get(difficulty, "Средний уровень.")

    if question.variants:  # Вопросы с вариантами
        tasks = [asyncio.create_task(_explain_variant(
            base_prompt=base_prompt,
            variant=variant,
            question_type=question.type,
            difficulty_hint=difficulty_hint
        )) for variant in question.variants]
        explanations = await asyncio.gather(*tasks, return_exceptions=True)
        for variant, exp in zip(question.variants, explanations):
            if isinstance(exp, Exception):
                variant.explanation = "Ошибка генерации объяснения."  # Fallback
            else:
                variant.explanation = exp
    else:  # Open-ended: только общее для правильного
        question.general_explanation = await _explain_general_correct(
            base_prompt=base_prompt,
            correct_answer=question.correct_answer,
            difficulty_hint=difficulty_hint
        )

async def _explain_variant(
        base_prompt: str,
        variant: AnswerVariant,
        question_type: QuestionType,
        difficulty_hint: str
    ) -> str:
    if variant.is_correct:
        user_prompt = f"""
        Правильный вариант: "{variant.text}"
        
        Напиши подтверждение (2–4 предложения), почему верно.
        {difficulty_hint}
        Начинай с "Верно", "Правильно".
        """
    else:
        user_prompt = f"""
        Неправильный вариант: "{variant.text}"
        
        Объясни, почему неверно, и укажи правильный.
        {difficulty_hint}
        Начинай с "Неверно", "Ошибка".
        Максимум 4 предложения.
        """

    # Специально для True/False
    if question_type == "true_false":
        if variant.text.lower() == "true":
            user_prompt += "\nУчитывай: это утверждение."
        else:
            user_prompt += "\nУчитывай: отрицание утверждения."

    explanation, _ = await proxy_completion(
        text="",
        user_prompt=base_prompt + "\n" + user_prompt,
        system_prompt="Ты — доброжелательный преподаватель.",
        temperature=0.3,
        max_tokens=200,
    )
    return explanation.strip() or "Объяснение недоступно."  # Fallback

async def _explain_general_correct(
        base_prompt: str,
        correct_answer: Optional[str | List[str]],
        difficulty_hint: str
        ) -> str:
    ans_str = correct_answer if isinstance(correct_answer, str) else ", ".join(correct_answer or [])
    user_prompt = f"""
    Правильный ответ: "{ans_str}"
    
    Напиши объяснение (3–5 предложений), почему это правильно.
    {difficulty_hint}
    Начинай с "Правильный ответ потому что...".
    """

    explanation, _ = await proxy_completion(
        text="",
        user_prompt=base_prompt + "\n" + user_prompt,
        system_prompt="Ты — доброжелательный преподаватель.",
        temperature=0.3,
        max_tokens=300,
    )
    return explanation.strip() or "Объяснение недоступно."  # Fallback

def _get_variant_text(variant):
    if hasattr(variant, "text"):
        return getattr(variant, "text")
    if isinstance(variant, dict):
        return variant.get("text", "")
    return str(variant)

def _set_variant_explanation(variant, explanation: str):
    if hasattr(variant, "explanation"):
        setattr(variant, "explanation", explanation)
    elif isinstance(variant, dict):
        variant["explanation"] = explanation
    else:
        try:
            setattr(variant, "explanation", explanation)
        except Exception:
            pass

async def _explain_variant_via_proxy(base_prompt: str, variant_text: str, is_correct: bool,
                                     question_type: str, difficulty_hint: str) -> str:

    if is_correct:
        user = f'''
    Правильный вариант: "{variant_text}"
    
    Напиши подтверждение (2–4 предложения), почему верно.
    {difficulty_hint}
    Начинай с "Верно", "Правильно".
    '''
    else:
        user = f'''
    Неправильный вариант: "{variant_text}"
    
    Объясни, почему неверно, и укажи правильный.
    {difficulty_hint}
    Начинай с "Неверно", "Ошибка".
    Максимум 4 предложения.
    '''

    if question_type == "true_false":
        if variant_text.strip().lower() in ("true", "верно", "да"):
            user += "\nУчитывай: это утверждение."
        else:
            user += "\nУчитывай: отрицание утверждения."

    explanation, _ = await proxy_completion(
        text="",
        user_prompt=base_prompt + "\n" + user,
        system_prompt="Ты — доброжелательный преподаватель.",
        temperature=0.3,
        max_tokens=200,
    )
    explanation = (explanation or "").strip()
    return explanation or "Объяснение недоступно."


async def _explain_general_via_proxy(base_prompt: str, correct_answer, difficulty_hint: str) -> str:
    ans_str = correct_answer if isinstance(correct_answer, str) else ", ".join(correct_answer or [])

    user = f'''
    Правильный ответ: "{ans_str}"
    
    Напиши объяснение (3–5 предложений), почему это правильно.
    {difficulty_hint}
    Начинай с "Правильный ответ потому что...".
    '''

    explanation, _ = await proxy_completion(
        text="",
        user_prompt=base_prompt + "\n" + user,
        system_prompt="Ты — доброжелательный преподаватель.",
        temperature=0.3,
        max_tokens=300,
    )
    explanation = (explanation or "").strip()
    return explanation or "Объяснение недоступно."


async def generate_all_explanations_async(
        questions: List[QuizQuestion],
        rag_store: SimpleVectorStore,
        difficulty: str = "medium"
    ) -> None:

    difficulty_hint = {
        "easy": "Объясняй как для новичка, с примерами.",
        "medium": "Средний уровень, без лишней сложности.",
        "hard": "Строго, с терминами, без упрощений."
    }.get(difficulty, "Средний уровень.")

    for i, question in enumerate(questions, start=1):
        chunks = rag_store.search_sync(question.text, top_k=6)
        context = "\n\n".join(chunks[:5]) if chunks else ""

        base_prompt = f"""
        Ты — преподаватель. Объясняй просто и по делу.
        Контекст из лекции:
        {context}
        
        Вопрос: {question.text}
        Тип: {question.type}
        """

        variants = question.variants or []

        if variants:
            tasks = []

            for variant in variants:
                variant_text = _get_variant_text(variant)
                if hasattr(variant, "is_correct"):
                    is_correct = bool(variant.is_correct)
                else:
                    is_correct = bool(variant.get("is_correct", False))

                tasks.append(_explain_variant_via_proxy(
                    base_prompt,
                    variant_text,
                    is_correct,
                    question.type,
                    difficulty_hint
                ))

            explanations = await asyncio.gather(*tasks, return_exceptions=True)

            for variant, exp in zip(variants, explanations):
                if isinstance(exp, Exception):
                    _set_variant_explanation(variant, "Ошибка генерации объяснения.")
                else:
                    _set_variant_explanation(variant, exp)

        else:
            try:
                gen = await _explain_general_via_proxy(
                    base_prompt,
                    question.correct_answer,
                    difficulty_hint
                )
                question.general_explanation = gen
            except Exception:
                question.general_explanation = "Ошибка генерации общего объяснения."


# -------------------------------------------
# Markdown formatter
# -------------------------------------------

def format_markdown_with_explanations(questions: List[QuizQuestion]) -> str:
    lines: List[str] = []

    for idx, q in enumerate(questions, start=1):
        lines.append(f"### Вопрос {idx}:")
        lines.append("")
        # ... (оставляем весь ваш текст форматтера)
        # Но именно здесь нужно вставить полный текст форматтера из main_explainer_and_quiz_generate.py
        # Я опускаю повтор во избежание дублирования длинного кода
        pass

    return "\n".join(lines).strip()

