from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# Генерация вопросов
from app.quiz.generation import QuizGenerationConfig, generate_quiz_from_text

# RAG (синхронно)
from app.quiz.rag import SimpleVectorStore

# Pydantic модель для QuizQuestion
from app.quiz.models import QuizQuestion

# Прокси клиент (async)
from app.services.proxy_client import proxy_completion

import asyncio
import sys


# --- Утилиты для работы с вариантами, подстраховки типов --- #
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
    """
    Вызов LLM для объяснения одного варианта.
    Возвращает строку-объяснение.
    """
    if is_correct:
        user_prompt = f"""
Правильный вариант: "{variant_text}"

Напиши подтверждение (2–4 предложения), почему верно.
{difficulty_hint}
Начинай с "Верно", "Правильно".
"""
    else:
        user_prompt = f"""
Неправильный вариант: "{variant_text}"

Объясни, почему неверно, и укажи правильный.
{difficulty_hint}
Начинай с "Неверно", "Ошибка".
Максимум 4 предложения.
"""

    if question_type == "true_false":
        if variant_text.strip().lower() in ("true", "верно", "да"):
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
    return (explanation or "").strip() or "Объяснение недоступно."


async def _explain_general_via_proxy(base_prompt: str, correct_answer, difficulty_hint: str) -> str:
    """
    Общее объяснение для открытых вопросов.
    """
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
    return (explanation or "").strip() or "Объяснение недоступно."


async def generate_all_explanations_async(questions: List[QuizQuestion], rag_store: SimpleVectorStore,
                                          difficulty: str = "medium") -> None:
    """
    Асинхронно генерирует объяснения ко всем вопросам и записывает их
    прямо в переданные pydantic-объекты QuizQuestion.
    """
    difficulty_hint = {
        "easy": "Объясняй как для новичка, с примерами.",
        "medium": "Средний уровень, без лишней сложности.",
        "hard": "Строго, с терминами, без упрощений."
    }.get(difficulty, "Средний уровень.")

    for i, question in enumerate(questions, start=1):
        print(f"  Вопрос {i}/{len(questions)}: {question.text[:80]!s}...")
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
            # генерируем параллельно объяснения для вариантов текущего вопроса
            tasks = []
            for variant in variants:
                variant_text = _get_variant_text(variant)
                is_correct = False
                if hasattr(variant, "is_correct"):
                    is_correct = getattr(variant, "is_correct") or False
                elif isinstance(variant, dict):
                    is_correct = bool(variant.get("is_correct", False))

                tasks.append(_explain_variant_via_proxy(
                    base_prompt=base_prompt,
                    variant_text=variant_text,
                    is_correct=is_correct,
                    question_type=question.type,
                    difficulty_hint=difficulty_hint,
                ))

            explanations = await asyncio.gather(*tasks, return_exceptions=True)
            for variant, exp in zip(variants, explanations):
                if isinstance(exp, Exception):
                    _set_variant_explanation(variant, "Ошибка генерации объяснения.")
                else:
                    _set_variant_explanation(variant, str(exp))
        else:
            # открытый вопрос — одно общее объяснение
            try:
                general = await _explain_general_via_proxy(
                    base_prompt=base_prompt,
                    correct_answer=question.correct_answer,
                    difficulty_hint=difficulty_hint,
                )
                question.general_explanation = general
            except Exception:
                question.general_explanation = "Ошибка генерации общего объяснения."


def _format_markdown_with_explanations(questions: List[QuizQuestion]) -> str:
    """
    Фомирует markdown вручную из pydantic-объектов, включая пояснения.
    Это гарантирует, что пояснения (variant.explanation / general_explanation)
    будут видны в итоговом тексте.
    """
    lines: List[str] = []
    for idx, q in enumerate(questions, start=1):
        lines.append(f"### Вопрос {idx}:")
        lines.append("")
        # Заголовок вопроса
        if q.type == "true_false":
            lines.append(f"**[True/False]** {q.text}")
            lines.append("")
            if q.variants:
                for v in q.variants:
                    v_text = _get_variant_text(v)
                    is_corr = getattr(v, "is_correct", False) if hasattr(v, "is_correct") else (v.get("is_correct") if isinstance(v, dict) else False)
                    expl = getattr(v, "explanation", "") if hasattr(v, "explanation") else (v.get("explanation", "") if isinstance(v, dict) else "")
                    lines.append(f"- {v_text} {'(правильно)' if is_corr else ''}")
                    if expl:
                        lines.append(f"  - Пояснение: {expl}")
                lines.append("")
            else:
                lines.append(f"- Ответ: {q.correct_answer}")
                if q.general_explanation:
                    lines.append(f"  - Пояснение: {q.general_explanation}")
                lines.append("")

        elif q.type in ("multiple_choice", "select_all_that_apply"):
            label = "Multiple Choice" if q.type == "multiple_choice" else "Select All That Apply"
            lines.append(f"**[{label}]** {q.text}")
            lines.append("")
            if q.variants:
                for v in q.variants:
                    v_text = _get_variant_text(v)
                    vid = getattr(v, "id", v.get("id") if isinstance(v, dict) else "")
                    is_corr = getattr(v, "is_correct", False) if hasattr(v, "is_correct") else (v.get("is_correct", False) if isinstance(v, dict) else False)
                    expl = getattr(v, "explanation", "") if hasattr(v, "explanation") else (v.get("explanation", "") if isinstance(v, dict) else "")
                    correct_marker = "(✓)" if is_corr else ""
                    lines.append(f"- {vid}) {v_text} {correct_marker}")
                    if expl:
                        lines.append(f"  - Пояснение: {expl}")
                lines.append("")
            else:
                # на всякий случай: если нет variants, показываем correct_answer
                lines.append(f"- Ответ: {q.correct_answer}")
                if q.general_explanation:
                    lines.append(f"  - Пояснение: {q.general_explanation}")
                lines.append("")

        elif q.type == "fill_in_the_blank":
            lines.append(f"**[Fill in the Blank]** {q.text}")
            lines.append(f"- Ответ(ы): {', '.join(q.correct_answer or [])}")
            if q.general_explanation:
                lines.append(f"  - Пояснение: {q.general_explanation}")
            lines.append("")

        elif q.type == "matching":
            lines.append(f"**[Matching]** {q.text}")
            if q.matching_pairs:
                lines.append("Пары (правильные соответствия):")
                for p in q.matching_pairs:
                    # p может быть либо MatchingPair pydantic-объект либо dict
                    left = getattr(p, "left_option", p.get("left_option") if isinstance(p, dict) else "")
                    right = getattr(p, "right_option", p.get("right_option") if isinstance(p, dict) else "")
                    lines.append(f"- {left} → {right}")
                lines.append("")
            if q.general_explanation:
                lines.append(f"  - Пояснение: {q.general_explanation}")
            lines.append("")

        elif q.type in ("short_answer", "long_answer"):
            label = "Short Answer" if q.type == "short_answer" else "Long Answer"
            lines.append(f"**[{label}]** {q.text}")
            lines.append(f"- Эталонный ответ: {q.correct_answer}")
            if q.general_explanation:
                lines.append(f"  - Пояснение: {q.general_explanation}")
            lines.append("")

        else:
            # fallback
            lines.append(f"**[{q.type}]** {q.text}")
            if q.correct_answer:
                lines.append(f"- Ответ: {q.correct_answer}")
            if q.variants:
                for v in q.variants:
                    v_text = _get_variant_text(v)
                    expl = getattr(v, "explanation", "") if hasattr(v, "explanation") else (v.get("explanation", "") if isinstance(v, dict) else "")
                    lines.append(f"- {v_text}")
                    if expl:
                        lines.append(f"  - Пояснение: {expl}")
            if q.general_explanation:
                lines.append(f"  - Пояснение: {q.general_explanation}")
            lines.append("")

    return "\n".join(lines).strip()


def main() -> None:
    # 1. Чтение конспекта
    note_path = Path("субд1.md")
    if not note_path.exists():
        print(f"Файл {note_path} не найден. Положите исходный конспект в тот же каталог и повторите.")
        sys.exit(1)

    note_text = note_path.read_text(encoding="utf-8")

    # 2. Настройки генерации
    cfg = QuizGenerationConfig(
        language="Русский",
        generate_true_false=True,
        num_true_false=2,
        generate_multiple_choice=True,
        num_multiple_choice=3,
        generate_select_all_that_apply=True,
        num_select_all_that_apply=2,
        generate_fill_in_the_blank=True,
        num_fill_in_the_blank=2,
        generate_matching=True,
        num_matching=1,
        generate_short_answer=True,
        num_short_answer=2,
        generate_long_answer=True,
        num_long_answer=1,
    )

    # 3. Генерация вопросов (единственная асинхронная часть)
    print("=== Генерация вопросов LLM ===")
    raw_questions = asyncio.run(generate_quiz_from_text(note_text, cfg))

    # 4. Конвертация в Pydantic QuizQuestion — чтобы удобно хранить пояснения
    from app.quiz.models import (
        TrueFalseQuestion,
        MultipleChoiceQuestion,
        SelectAllThatApplyQuestion,
        FillInTheBlankQuestion,
        MatchingQuestion,
        ShortOrLongAnswerQuestion,
        QuizQuestion as _QuizQuestionModel,
    )

    questions: List[QuizQuestion] = []
    for q in raw_questions:
        variants_data: Optional[List[dict]] = None
        correct_answer: Optional[str | List[str]] = None
        matching_pairs_data: Optional[List[dict]] = None
        q_type: str

        if isinstance(q, TrueFalseQuestion):
            q_type = "true_false"
            variants_data = [
                {"id": "True", "text": "True", "is_correct": q.answer, "explanation": ""},
                {"id": "False", "text": "False", "is_correct": not q.answer, "explanation": ""},
            ]
            correct_answer = "True" if q.answer else "False"

        elif isinstance(q, MultipleChoiceQuestion):
            q_type = "multiple_choice"
            variants_data = [
                {"id": chr(65 + i), "text": opt, "is_correct": (i == q.answer), "explanation": ""}
                for i, opt in enumerate(q.options)
            ]
            correct_answer = q.options[q.answer]

        elif isinstance(q, SelectAllThatApplyQuestion):
            q_type = "select_all_that_apply"
            variants_data = [
                {"id": chr(65 + i), "text": opt, "is_correct": (i in q.answer), "explanation": ""}
                for i, opt in enumerate(q.options)
            ]
            correct_answer = [q.options[i] for i in q.answer]

        elif isinstance(q, FillInTheBlankQuestion):
            q_type = "fill_in_the_blank"
            correct_answer = q.answer

        elif isinstance(q, MatchingQuestion):
            q_type = "matching"
            matching_pairs_data = [
                {"left_option": p.left_option, "right_option": p.right_option}
                for p in q.answer
            ]
            correct_answer = [f"{p.left_option} → {p.right_option}" for p in q.answer]

        elif isinstance(q, ShortOrLongAnswerQuestion):
            q_type = "short_answer" if len(q.answer) < 250 else "long_answer"
            correct_answer = q.answer

        else:
            continue

        quiz_q = _QuizQuestionModel(
            id=uuid4(),
            text=q.question,
            type=q_type,
            variants=variants_data,
            correct_answer=correct_answer,
            matching_pairs=matching_pairs_data,
            general_explanation="",
        )
        questions.append(quiz_q)

    # 5. RAG — синхронно
    rag_store = SimpleVectorStore()
    rag_store.add_document_sync(note_text)

    # 6. Генерация объяснений (асинхронно — один run для всех)
    print("\nГенерация объяснений к ответам...")
    asyncio.run(generate_all_explanations_async(questions, rag_store, difficulty="medium"))

    # 7. Формирование и печать markdown с объяснениями
    markdown = _format_markdown_with_explanations(questions)

    print("\n" + "=" * 80)
    print("ГОТОВЫЙ КВИЗ С ПОЯСНЕНИЯМИ".center(80))
    print("=" * 80 + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
