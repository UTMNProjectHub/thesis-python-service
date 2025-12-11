# app/services/tasks_processor.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional
from uuid import uuid4, UUID
import random

from app.services.s3_client import S3Client
from app.services.postgres import PostgresClient
from app.services.rabbitmq import RabbitClient

# Для SUMMARY (конспект)
from app.documents.pdf_reader import load_pdf_document
from app.utils.pdf_utils import extract_text_from_pdf
from app.documents.chunking import chunk_document_pages
from app.documents.indexers import HybridRetriever
from app.curriculum.models import LectureTopic, DifficultyLevel
from app.lectures import build_lecture_plan_for_topic, generate_lecture_markdown

from app.utils.md_to_pdf import markdown_to_pdf


# FAQ
from app.faq.config import FAQGenerationConfig
from app.faq.generator import generate_faq_from_file

# QUIZ + RAG + EXPLAINER
from app.quiz.generation import QuizGenerationConfig, generate_quiz_from_text
from app.quiz.rag import SimpleVectorStore
from app.quiz.explainer import generate_explanations
from app.quiz.models import (
    TrueFalseQuestion,
    MultipleChoiceQuestion,
    SelectAllThatApplyQuestion,
    FillInTheBlankQuestion,
    MatchingQuestion,
    ShortOrLongAnswerQuestion,
    QuizQuestion,
    AnswerVariant,
    MatchingPair,
    Question,  # union тип
    QuestionType
)

UI_TO_INTERNAL: dict[str, QuestionType] = {
    "truefalse": "true_false",
    "multichoice": "multiple_choice",
    "matching": "matching",
    "shortanswer": "short_answer",
    "essay": "long_answer",
    "numerical": "fill_in_the_blank",  # либо отдельная логика, если понадобится
}

INTERNAL_TO_DB: dict[QuestionType, str] = {
    "true_false": "truefalse",
    "multiple_choice": "multichoice",
    "matching": "matching",
    "short_answer": "shortanswer",
    "long_answer": "essay",
}


def _distribute_question_counts(
        total: int,
        ui_types: list[str],
) -> dict[str, int]:
    """
    Распределяет total вопросов по типам из ui_types
    (multichoice|essay|matching|truefalse|shortanswer|numerical).

    - numerical игнорируется;
    - если total >= len(types) -> каждому минимум 1, остальное раскидывается случайно;
    - если total < len(types)  -> ровно total типов получают по 1 вопросу.
    """
    # отфильтровали numerical
    types = [t for t in ui_types if t != "numerical"]

    # если после фильтра ничего не осталось — дефолтный тип
    if not types:
        types = ["multichoice"]

    n = len(types)
    if total <= 0:
        return {t: 0 for t in types}

    # total >= n — каждому минимум по одному, плюс равномерный остаток
    if total >= n:
        base = total // n          # минимум для каждого типа
        remainder = total % n      # сколько "лишних" вопросов раскидываем

        counts = {t: base for t in types}

        # если базовое распределение дало всем >=1? при total>=n base всегда >=1
        # остаток распределяем случайно по типам
        if remainder > 0:
            extra_types = random.sample(types, k=remainder)
            for t in extra_types:
                counts[t] += 1
    else:
        # total < n — типов больше, чем вопросов.
        # Выбираем случайные total типов, им даём по 1 вопросу.
        counts = {t: 0 for t in types}
        chosen = random.sample(types, k=total)
        for t in chosen:
            counts[t] = 1

    return counts

class TaskProcessor:
    """
    Обработчик задач SummaryGen и QuizGen.

    SummaryGen пайплайн:
      1) из Postgres получаем s3Index для файлов;
      2) скачиваем файлы из S3 → files_materials;
      3) по PDF строим конспект (lecture_md);
      4) сохраняем summary в Postgres;
      5) на базе lecture_md генерируем Quiz (+объяснения);
      6) на базе lecture_md генерируем FAQ;
      7) сохраняем Quiz/FAQ в Postgres;
      8) отправляем SummaryGenComplete.

    QuizGen пайплайн:
      - аналогично, но без генерации summary (используем исходные файлы как текст).
    """

    def __init__(self, rabbit: RabbitClient, s3: S3Client, db: PostgresClient) -> None:
        self.rabbit = rabbit
        self.s3 = s3
        self.db = db

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: конвертация raw_questions → QuizQuestion
    # ------------------------------------------------------------------
    def _convert_raw_to_quiz_questions(
            self, raw_questions: List[Question],
            allowed_types: Optional[set[QuestionType]] = None,
    ) -> List[QuizQuestion]:
        questions: List[QuizQuestion] = []

        for q in raw_questions:
            q_type: QuestionType
            variants: List[AnswerVariant] | None = None
            correct_answer = None
            matching_pairs: List[MatchingPair] | None = None

            if isinstance(q, TrueFalseQuestion):
                q_type = "true_false"
                variants = [
                    AnswerVariant(id="True", text="True", is_correct=q.answer, explanation=""),
                    AnswerVariant(id="False", text="False", is_correct=not q.answer, explanation=""),
                ]
                correct_answer = "True" if q.answer else "False"

            elif isinstance(q, MultipleChoiceQuestion):
                q_type = "multiple_choice"
                variants = [
                    AnswerVariant(
                        id=chr(65 + i),
                        text=opt,
                        is_correct=(i == q.answer),
                        explanation="",
                    )
                    for i, opt in enumerate(q.options)
                ]
                correct_answer = q.options[q.answer]

            elif isinstance(q, SelectAllThatApplyQuestion):
                continue
                q_type = "select_all_that_apply"
                variants = [
                    AnswerVariant(
                        id=chr(65 + i),
                        text=opt,
                        is_correct=(i in q.answer),
                        explanation="",
                    )
                    for i, opt in enumerate(q.options)
                ]
                correct_answer = [q.options[i] for i in q.answer]

            elif isinstance(q, FillInTheBlankQuestion):
                continue
                q_type = "fill_in_the_blank"
                correct_answer = q.answer

            elif isinstance(q, MatchingQuestion):
                q_type = "matching"
                matching_pairs = [
                    MatchingPair(left_option=p.left_option, right_option=p.right_option)
                    for p in q.answer
                ]
                correct_answer = [f"{p.left_option} → {p.right_option}" for p in q.answer]

            elif isinstance(q, ShortOrLongAnswerQuestion):
                q_type = "short_answer" if len(q.answer) < 250 else "long_answer"
                correct_answer = q.answer

            else:
                # неизвестный тип — пропускаем
                continue

            if allowed_types is not None and q_type not in allowed_types:
                continue

            qq = QuizQuestion(
                id=uuid4(),
                text=q.question,
                type=q_type,  # QuestionType у вас Literal[…]
                variants=variants,
                correct_answer=correct_answer,
                matching_pairs=matching_pairs,
                general_explanation="",
            )
            questions.append(qq)

        return questions

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: генерация пояснений (explainer) по lecture_md
    # ------------------------------------------------------------------
    async def _generate_explanations_for_quiz(
            self,
            lecture_md: str,
            questions: List[QuizQuestion],
            difficulty: str,
    ) -> None:
        rag_store = SimpleVectorStore()
        await rag_store.add_document(lecture_md)

        for q in questions:
            chunks = await rag_store.search(q.text, top_k=6)
            await generate_explanations(q, chunks, difficulty=difficulty)

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНОЕ: сохранение квиза в Postgres
    # ------------------------------------------------------------------
    async def _persist_quiz(
            self,
            quiz_id: UUID,
            theme_id: Optional[int],
            questions: List[QuizQuestion],
            file_ids: List[UUID],
            theme_name: Optional[str] = None,
    ) -> None:
        # метаданные квиза
        quiz_name = f"Авто-квиз по теме «{theme_name}»" if theme_name else "Авто-квиз"
        await self.db.save_quiz_metadata(
            quiz_id=quiz_id,
            theme_id=theme_id,
            name=quiz_name,
            description="Квиз для провeрки знаний темы.",
        )

        for q in questions:
            q_db_id = uuid4()

            db_qtype = INTERNAL_TO_DB[q.type]
            multi_answer = q.type in ("select_all_that_apply", "matching")

            await self.db.insert_question(
                question_id=q_db_id,
                qtype=db_qtype,
                text=q.text,
                multi_answer=multi_answer,
            )

            # ссылки вопрос ↔ файл(ы)
            for f_id in file_ids:
                await self.db.insert_reference_question(
                    ref_id=uuid4(),
                    question_id=q_db_id,
                    file_id=f_id,
                )

            # варианты
            if q.type == "matching" and q.matching_pairs:
                # В matching складываем пары в matchingConfig,
                # variantId оставляем NULL.
                matching_config = {
                    "pairs": [
                        {"left": p.left_option, "right": p.right_option}
                        for p in q.matching_pairs
                    ]
                }
                await self.db.insert_question_variant_link(
                    link_id=uuid4(),
                    question_id=q_db_id,
                    variant_id=None,
                    is_right=True,
                    matching_config=matching_config,
                )
            elif q.variants:
                for v in q.variants:
                    v_db_id = uuid4()
                    # простая логика: пояснение для правильного → explainRight, для неверного → explainWrong
                    explain_right = v.explanation if v.is_correct else ""
                    explain_wrong = v.explanation if not v.is_correct else ""
                    await self.db.insert_variant(
                        variant_id=v_db_id,
                        text=v.text,
                        explain_right=explain_right,
                        explain_wrong=explain_wrong,
                    )
                    await self.db.insert_question_variant_link(
                        link_id=uuid4(),
                        question_id=q_db_id,
                        variant_id=v_db_id,
                        is_right=v.is_correct,
                        matching_config=None,
                    )

            # связь вопрос–квиз
            await self.db.link_question_to_quiz(
                link_id=uuid4(),
                quiz_id=quiz_id,
                question_id=q_db_id,
            )

    # ------------------------------------------------------------------
    # SUMMARY GEN
    # ------------------------------------------------------------------
    async def handle_summary_gen(self, payload: dict) -> None:
        """
        SummaryGen:
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            files: uuid[],
            additional_requirements: text
        }
        """
        summary_id = UUID(payload["summaryId"])
        subject_id = payload["subjectId"]
        theme_id = payload["themeId"]
        file_ids = [UUID(fid) for fid in payload["files"]]
        additional_req = payload.get("additional_requirements") or ""

        try:
            theme = await self.db.get_theme_name(theme_id)
            theme_name: str = str(theme)

            summary_file_id = uuid4()
            # 1) s3Index из БД
            s3_keys = await self.db.get_s3_keys_for_file_ids(file_ids)

            # 2) скачиваем файлы из S3 → files_materials
            local_files = [self.s3.download_to_materials(k) for k in s3_keys]

            # выбираем первый PDF для генерации конспекта
            pdf_files = [Path(f) for f in local_files if f and f.lower().endswith(".pdf")]
            if not pdf_files:
                raise RuntimeError("Не найден PDF-файл среди materials для SummaryGen")

            pdf_path = pdf_files[0]

            # 3) строим конспект (lecture_md) — логика как в main_lecture_full_test.py
            doc, pages = load_pdf_document(str(pdf_path))
            chunks = chunk_document_pages(doc, pages, max_tokens=700)
            retriever = HybridRetriever(alpha=0.7)
            retriever.index(chunks)

            topic = LectureTopic(
                id=str(summary_file_id),
                title=f"Авто-конспект по теме {theme_name}",
                description=f"Автоматически сгенерированный конспект.",
                difficulty=DifficultyLevel.MEDIUM,
                keywords=[],
                duration_min=90,
                source_docs=[doc.id],
                order=1,
            )

            plan = await build_lecture_plan_for_topic(
                topic,
                retriever=retriever,
                top_k_chunks=8,
                min_sections=3,
                max_sections=7,
            )

            lecture_md = await generate_lecture_markdown(
                plan=plan,
                retriever=retriever,
                topic_description=topic.description,
                max_tokens_per_section=1500,
                top_k_chunks_per_section=5,
            )

            # сохраняем md на диск (чтобы re-use для FAQ)
            summary_path = Path("files_materials") / f"summary_{summary_id}.md"
            summary_path.write_text(lecture_md, encoding="utf-8")

            pdf_path = Path("files_materials") / f"summary_{summary_id}.pdf"
            markdown_to_pdf(summary_path, pdf_path)

            # s3_key = f"quizy/summaries/{summary_file_id}.md"
            file_name = f"auto conspect on theme.pdf"

            s3_key = self.s3.upload_file_to_bucket(
                local_path=str(pdf_path),
                original_name=file_name,
                bucket='summaries/',
                user_id=None,
            )

            await self.db.save_summary_text(
                summary_id=summary_file_id,
                theme_id=theme_id,
                s3_key=s3_key,
                file_name=file_name,
                user_id=None,
            )

            # 9) отправляем SummaryGenComplete
            await self.rabbit.publish(
                self.rabbit.queue_summary_gen_complete,
                {
                    "summaryId": str(summary_id),
                    "subjectId": subject_id,
                    "themeId": theme_id,
                    "status": "SUCCESS",
                    "error": "",
                },
            )


        except Exception as e:
            await self.rabbit.publish(
                self.rabbit.queue_summary_gen_complete,
                {
                    "summaryId": str(summary_id),
                    "subjectId": subject_id,
                    "themeId": theme_id,
                    "status": "FAILED",
                    "error": str(e),
                },
            )


    # ------------------------------------------------------------------
    # QUIZ GEN (упрощённый — без конспекта)
    # ------------------------------------------------------------------
    async def handle_quiz_gen(self, payload: dict) -> None:
        """
        QuizGen:
        {
            quizId: uuid,
            files: uuid[],
            difficulty: string (easy|medium|hard),
            question_count: number,
            question_types: string[],
            additional_requirements: text
        }
        """
        quiz_id = UUID(payload["quizId"])
        file_ids = [UUID(fid) for fid in payload["files"]]
        difficulty = payload.get("difficulty", "medium")
        question_count = payload.get("question_count", 10)
        raw_qtypes = payload.get("question_types") or ["multichoice", "matching", "truefalse"]
        additional_req = payload.get("additional_requirements") or ""
        theme_id_raw = payload.get("themeId")
        theme_id: Optional[int] = int(theme_id_raw) if theme_id_raw is not None else None

        internal_allowed: set[QuestionType] = set()
        for t in raw_qtypes:
            t_norm = t.lower()
            if t_norm in UI_TO_INTERNAL:
                internal_allowed.add(UI_TO_INTERNAL[t_norm])

        if not internal_allowed:
            internal_allowed = {"multiple_choice", "matching", "true_false"}

        # Допустимые типы для БД (у вас они: multichoice|essay|matching|truefalse|shortanswer|numerical)
        allowed_db_types: set[str] = set()
        if "truefalse" in internal_allowed:
            allowed_db_types.add("truefalse")
        if "multichoice" in internal_allowed:
            allowed_db_types.add("multichoice")
        if "matching" in internal_allowed:
            allowed_db_types.add("matching")
        if "shortanswer" in internal_allowed:
            allowed_db_types.add("shortanswer")
        if "essay" in internal_allowed:
            allowed_db_types.add("essay")
        # numerical сейчас отдельного типа в моделях нет — можно трактовать как shortanswer или игнорировать.
        # if "numerical" in allowed:
        #     allowed_db_types.add("shortanswer")  # предположим, что числовой ответ — частный случай shortanswer

        if not allowed_db_types:
            allowed_db_types = {"multichoice", "matching", "truefalse"}

        theme_name: Optional[str] = None

        try:
            if theme_id is not None:
                theme_name = await self.db.get_theme_name(theme_id)

            # 1) s3Index → S3
            s3_keys = await self.db.get_s3_keys_for_file_ids(file_ids)
            local_files = [self.s3.download_to_materials(k) for k in s3_keys]

            # 2) просто склеиваем тексты файлов в один note_text
            note_text = ""
            for lf in local_files:
                if not lf:
                    continue
                p = Path(lf)
                if p.suffix.lower() in (".md", ".txt"):
                    note_text += p.read_text(encoding="utf-8") + "\n\n"
                elif p.suffix.lower() == ".pdf":
                    # читаем текст из PDF
                    try:
                        pdf_text = extract_text_from_pdf(str(p))
                    except Exception as e:
                        # при желании можно просто залогировать и continue
                        raise RuntimeError(f"Не удалось прочитать PDF '{p}': {e}")

                    if pdf_text.strip():
                        note_text += pdf_text + "\n\n"

            if not note_text.strip():
                raise RuntimeError("Нет текстового содержимого для генерации квиза")

            ui_counts = _distribute_question_counts(
                total=question_count,
                ui_types=raw_qtypes,
            )

            internal_allowed: set[QuestionType] = set()
            for ui_type, cnt in ui_counts.items():
                if cnt <= 0:
                    continue
                internal = UI_TO_INTERNAL.get(ui_type)
                if internal:
                    internal_allowed.add(internal)

            if not internal_allowed:
                internal_allowed = {"multiple_choice"}

            num_true_false = ui_counts.get("truefalse", 0)
            num_multichoice = ui_counts.get("multichoice", 0)
            num_matching = ui_counts.get("matching", 0)
            num_shortanswer = ui_counts.get("shortanswer", 0)
            num_essay = ui_counts.get("essay", 0)

            # 3) генерируем Quiz
            cfg = QuizGenerationConfig(
                language="Русский",

                generate_true_false=num_true_false > 0,
                num_true_false=num_true_false,

                generate_multiple_choice=num_multichoice > 0,
                num_multiple_choice=num_multichoice,

                generate_select_all_that_apply=False,
                num_select_all_that_apply=0,

                generate_fill_in_the_blank=False,
                num_fill_in_the_blank=0,

                generate_matching=num_matching > 0,
                num_matching=num_matching,

                generate_short_answer=num_shortanswer > 0,
                num_short_answer=num_shortanswer,

                generate_long_answer=num_essay > 0,
                num_long_answer=num_essay,
            )

            raw_questions = await generate_quiz_from_text(note_text, cfg, theme_name=theme_name)
            quiz_questions = self._convert_raw_to_quiz_questions(raw_questions, allowed_types=internal_allowed)

            # 4) пояснения
            await self._generate_explanations_for_quiz(
                lecture_md=note_text,
                questions=quiz_questions,
                difficulty=difficulty,
            )

            # 5) сохранение в Postgres
            await self._persist_quiz(
                quiz_id=quiz_id,
                theme_id=theme_id,
                questions=quiz_questions,
                file_ids=file_ids,
                theme_name=theme_name,
            )

            # 6) ответ
            await self.rabbit.publish(
                self.rabbit.queue_quiz_gen_complete,
                {
                    "quizId": str(quiz_id),
                    "status": "SUCCESS",
                    "error": "",
                },
            )


        except Exception as e:
            await self.rabbit.publish(
                self.rabbit.queue_quiz_gen_complete,
                {
                    "quizId": str(quiz_id),
                    "status": "FAILED",
                    "error": str(e),
                },
            )

