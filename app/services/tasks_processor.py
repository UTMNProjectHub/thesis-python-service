from __future__ import annotations

import asyncio
import inspect
import logging
import random
from pathlib import Path
from typing import List, Optional
from uuid import uuid4, UUID

from app.api.core.config import settings
from app.curriculum.models import LectureTopic, DifficultyLevel
from app.documents.chunking import chunk_document_pages
from app.documents.docx_reader import extract_docx_text
from app.documents.indexers import HybridRetriever
from app.documents.pdf_reader import load_pdf_document
from app.faq import FAQGenerationConfig, format_faq_as_markdown, generate_faq_from_text
from app.lectures import build_lecture_plan_for_topic, generate_lecture_markdown
from app.quiz.explainer import generate_explanations
from app.quiz.generation import QuizGenerationConfig, generate_quiz_from_text
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
    Question,
    QuestionType
)
from app.quiz.rag import SimpleVectorStore
from app.services.postgres import PostgresClient
from app.services.rabbitmq import RabbitClient
from app.services.s3_client import S3Client
from app.utils.md_to_pdf import markdown_to_pdf
from app.utils.pdf_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)

UI_TO_INTERNAL: dict[str, QuestionType] = {
    "truefalse": "true_false",
    "multichoice": "multiple_choice",
    "matching": "matching",
    "shortanswer": "short_answer",
    "essay": "long_answer",
    "numerical": "fill_in_the_blank",
}

INTERNAL_TO_DB: dict[QuestionType, str] = {
    "true_false": "truefalse",
    "multiple_choice": "multichoice",
    "select_all_that_apply": "multichoice",
    "fill_in_the_blank": "shortanswer",
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
    types = [t for t in ui_types if t in UI_TO_INTERNAL]

    if not types:
        types = ["multichoice"]

    n = len(types)
    if total <= 0:
        return {t: 0 for t in types}

    if total >= n:
        base = total // n
        remainder = total % n

        counts = {t: base for t in types}

        if remainder > 0:
            extra_types = random.sample(types, k=remainder)
            for t in extra_types:
                counts[t] += 1
    else:
        counts = {t: 0 for t in types}
        chosen = random.sample(types, k=total)
        for t in chosen:
            counts[t] = 1

    return counts


def _sample_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    part = max(max_chars // 3, 1)
    middle_start = max((len(text) - part) // 2, 0)
    return "\n\n".join(
        [
            text[:part],
            text[middle_start:middle_start + part],
            text[-part:],
        ]
    )


FAQ_DETAIL_LEVELS = {"easy", "medium", "hard"}


def _extract_faq_source_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        text = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = extract_text_from_pdf(str(file_path))
    elif suffix == ".docx":
        text = extract_docx_text(file_path)
    else:
        raise RuntimeError(f"Unsupported FAQ source file type: {suffix}")

    text = text.strip()
    if not text:
        raise RuntimeError("FAQ source file has no text content")
    return text


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

    @staticmethod
    def _work_dir(kind: str, task_id: UUID) -> Path:
        path = Path("files_materials") / kind / str(task_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _download_s3_keys(
            self,
            s3_keys: List[str],
            work_dir: Path,
            task_kind: str,
            task_id: UUID,
    ) -> List[str]:
        local_files: List[str] = []
        for index, key in enumerate(s3_keys, start=1):
            logger.info(
                "%s downloading source file taskId=%s index=%d total=%d key=%s work_dir=%s",
                task_kind,
                task_id,
                index,
                len(s3_keys),
                key,
                work_dir,
            )
            local_file = await asyncio.to_thread(self.s3.download_to_materials, key, work_dir)
            local_files.append(local_file)
        return local_files

    async def _publish(self, queue: str, payload: dict) -> None:
        logger.info("Publishing task result queue=%s payload=%s", queue, payload)
        result = self.rabbit.publish(queue, payload)
        if inspect.isawaitable(result):
            await result

    def _convert_raw_to_quiz_questions(
            self, raw_questions: List[Question],
            allowed_types: Optional[set[QuestionType]] = None,
    ) -> List[QuizQuestion]:
        questions: List[QuizQuestion] = []
        logger.info(
            "Converting generated questions raw_count=%d allowed_types=%s",
            len(raw_questions),
            sorted(allowed_types) if allowed_types else None,
        )

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
                if q.answer < 0 or q.answer >= len(q.options):
                    logger.warning(
                        "Skipping multiple choice question with invalid answer index answer=%s options=%d question=%s",
                        q.answer,
                        len(q.options),
                        q.question[:120],
                    )
                    continue
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
                q_type = "select_all_that_apply"
                correct_indices = {idx for idx in q.answer if 0 <= idx < len(q.options)}
                variants = [
                    AnswerVariant(
                        id=chr(65 + i),
                        text=opt,
                        is_correct=(i in correct_indices),
                        explanation="",
                    )
                    for i, opt in enumerate(q.options)
                ]
                correct_answer = [q.options[i] for i in sorted(correct_indices)]

            elif isinstance(q, FillInTheBlankQuestion):
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
                logger.warning("Skipping unsupported generated question class=%s", type(q).__name__)
                # неизвестный тип — пропускаем
                continue

            if allowed_types is not None and q_type not in allowed_types:
                logger.info("Skipping generated question by type filter type=%s question=%s", q_type, q.question[:120])
                continue

            qq = QuizQuestion(
                id=uuid4(),
                text=q.question,
                type=q_type,
                variants=variants,
                correct_answer=correct_answer,
                matching_pairs=matching_pairs,
                general_explanation="",
            )
            questions.append(qq)
            logger.info(
                "Converted generated question type=%s variants=%d matching_pairs=%d",
                q_type,
                len(variants or []),
                len(matching_pairs or []),
            )

        logger.info("Generated questions converted count=%d", len(questions))
        return questions

    async def _generate_explanations_for_quiz(
            self,
            lecture_md: str,
            questions: List[QuizQuestion],
            difficulty: str,
    ) -> None:
        rag_store = SimpleVectorStore()
        await rag_store.add_document(lecture_md)
        logger.info(
            "Generating explanations questions=%d difficulty=%s source_chars=%d",
            len(questions),
            difficulty,
            len(lecture_md),
        )

        for q in questions:
            chunks = await rag_store.search(q.text, top_k=6)
            logger.info("Generating explanation questionId=%s type=%s chunks=%d", q.id, q.type, len(chunks))
            await generate_explanations(q, chunks, difficulty=difficulty)
        logger.info("Explanations generated questions=%d", len(questions))

    async def _persist_quiz(
            self,
            quiz_id: UUID,
            theme_id: Optional[int],
            questions: List[QuizQuestion],
            file_ids: List[UUID],
            theme_name: Optional[str] = None,
    ) -> None:
        quiz_name = f"Авто-квиз по теме «{theme_name}»" if theme_name else "Авто-квиз"
        logger.info(
            "Persisting quiz metadata quizId=%s themeId=%s questions=%d files=%d",
            quiz_id,
            theme_id,
            len(questions),
            len(file_ids),
        )
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
            logger.info(
                "Persisting question quizId=%s questionId=%s type=%s db_type=%s",
                quiz_id,
                q_db_id,
                q.type,
                db_qtype,
            )

            await self.db.insert_question(
                question_id=q_db_id,
                qtype=db_qtype,
                text=q.text,
                multi_answer=multi_answer,
            )

            for f_id in file_ids:
                await self.db.insert_reference_question(
                    ref_id=uuid4(),
                    question_id=q_db_id,
                    file_id=f_id,
                )

            if q.type == "matching" and q.matching_pairs:
                logger.info(
                    "Persisting matching pairs questionId=%s pairs=%d",
                    q_db_id,
                    len(q.matching_pairs),
                )
                for pair in q.matching_pairs:
                    v_db_id = uuid4()
                    await self.db.insert_variant(
                        variant_id=v_db_id,
                        text=f"{pair.left_option} -> {pair.right_option}",
                        explain_right="Верно",
                        explain_wrong="Неверно",
                        left_matching=pair.left_option,
                        right_matching=pair.right_option,
                    )
                    await self.db.insert_question_variant_link(
                        link_id=uuid4(),
                        question_id=q_db_id,
                        variant_id=v_db_id,
                        is_right=True,
                    )
            elif q.variants:
                logger.info("Persisting variants questionId=%s variants=%d", q_db_id, len(q.variants))
                for v in q.variants:
                    v_db_id = uuid4()
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
                    )

            await self.db.link_question_to_quiz(
                link_id=uuid4(),
                quiz_id=quiz_id,
                question_id=q_db_id,
            )
        logger.info("Quiz persisted quizId=%s", quiz_id)

    async def handle_faq_gen(self, payload: dict) -> None:
        """
        FAQGen:
        {
            summaryId: int,
            faqId: uuid,
            userId: uuid,
            title: str,
            numQuestions: int,
            detailLevel: "easy"|"medium"|"hard",
            additionalRequirements: text
        }
        """
        faq_id: UUID | None = None
        user_id: UUID | None = None
        response_faq_id = str(payload.get("faqId", ""))
        response_user_id = str(payload.get("userId", ""))
        s3_keys: List[str] = []

        try:
            summary_id = int(payload["summaryId"])
            faq_id = UUID(str(payload["faqId"]))
            user_id = UUID(str(payload["userId"]))
            response_faq_id = str(faq_id)
            response_user_id = str(user_id)

            title = str(payload.get("title") or "").strip() or f"FAQ {summary_id}"
            num_questions = int(payload.get("numQuestions", 10))
            if num_questions <= 0:
                raise ValueError("numQuestions must be greater than 0")

            detail_level = str(payload.get("detailLevel", "medium")).strip().lower()
            if detail_level not in FAQ_DETAIL_LEVELS:
                raise ValueError("detailLevel must be one of: easy, medium, hard")

            additional_requirements = str(payload.get("additionalRequirements") or "").strip()
            logger.info(
                "FAQGen started faqId=%s summaryId=%s userId=%s questions=%d detailLevel=%s additional_requirements_len=%d",
                faq_id,
                summary_id,
                user_id,
                num_questions,
                detail_level,
                len(additional_requirements),
            )

            summary_context = await self.db.get_summary_context(summary_id)
            if summary_context is None:
                raise RuntimeError(f"Summary not found: {summary_id}")

            theme_id = summary_context.get("themeId")
            subject_id = summary_context.get("subjectId")
            source_file_id = summary_context.get("sourceFileId")
            expected_s3_key = str(summary_context.get("lectureS3Key") or "").strip()
            if theme_id is None:
                raise RuntimeError(f"Summary {summary_id} has no themeId")
            if source_file_id is None:
                raise RuntimeError(f"Summary {summary_id} has no source fileId")
            if not expected_s3_key:
                raise RuntimeError(f"Summary {summary_id} source file has no s3Index")

            lecture_s3_key = expected_s3_key
            s3_keys = [lecture_s3_key]
            work_dir = self._work_dir("faq", faq_id)
            logger.info(
                "FAQGen context resolved faqId=%s summaryId=%s subjectId=%s themeId=%s sourceFileId=%s work_dir=%s",
                faq_id,
                summary_id,
                subject_id,
                theme_id,
                source_file_id,
                work_dir,
            )

            local_file = await asyncio.to_thread(self.s3.download_to_materials, lecture_s3_key, work_dir)
            source_path = Path(local_file)
            logger.info("FAQGen source downloaded faqId=%s path=%s", faq_id, source_path)

            source_text = await asyncio.to_thread(_extract_faq_source_text, source_path)
            if len(source_text) > settings.quiz_source_max_chars:
                original_chars = len(source_text)
                source_text = _sample_text(source_text, settings.quiz_source_max_chars)
                logger.info(
                    "FAQGen source text sampled faqId=%s original_chars=%d sampled_chars=%d",
                    faq_id,
                    original_chars,
                    len(source_text),
                )

            cfg = FAQGenerationConfig(
                language="ru",
                num_questions=num_questions,
                detail_level=detail_level,
                additional_requirements=additional_requirements,
            )
            faq = await generate_faq_from_text(source_text, title=title, cfg=cfg)
            if not faq.items:
                raise RuntimeError("FAQ generation returned no items")
            logger.info("FAQGen items generated faqId=%s count=%d", faq_id, len(faq.items))

            markdown = format_faq_as_markdown(faq)
            md_path = work_dir / f"faq_{faq_id}.md"
            pdf_path = work_dir / f"faq_{faq_id}.pdf"
            await asyncio.to_thread(md_path.write_text, markdown, encoding="utf-8")
            await asyncio.to_thread(markdown_to_pdf, md_path, pdf_path)
            logger.info("FAQGen PDF exported faqId=%s path=%s", faq_id, pdf_path)

            file_name = f"faq_{faq_id}.pdf"
            s3_key = await asyncio.to_thread(
                self.s3.upload_file_to_bucket,
                local_path=str(pdf_path),
                original_name=file_name,
                bucket="faqs",
                user_id=str(user_id),
            )

            faq_file_id = uuid4()
            await self.db.save_faq(
                faq_id=faq_id,
                summary_id=summary_id,
                theme_id=int(theme_id),
                source_file_id=source_file_id,
                faq_file_id=faq_file_id,
                s3_key=s3_key,
                file_name=file_name,
                user_id=user_id,
                difficulty_level=detail_level,
                num_questions=num_questions,
            )
            logger.info("FAQGen persisted faqId=%s fileId=%s s3_key=%s", faq_id, faq_file_id, s3_key)

            await self._publish(
                self.rabbit.queue_faq_gen_complete,
                {
                    "faqId": str(faq_id),
                    "userId": str(user_id),
                    "status": "SUCCESS",
                    "error": "",
                },
            )
            logger.info("FAQGen completed faqId=%s status=SUCCESS", faq_id)

        except Exception as e:
            logger.exception("FAQGen failed faqId=%s error=%s", response_faq_id, e)
            await self._publish(
                self.rabbit.queue_faq_gen_complete,
                {
                    "faqId": response_faq_id,
                    "userId": response_user_id,
                    "status": "FAILED",
                    "error": str(e),
                },
            )
        finally:
            await asyncio.to_thread(self.s3.finish_task_cache_usage, s3_keys)

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
        logger.info(
            "SummaryGen started summaryId=%s subjectId=%s themeId=%s files=%d additional_requirements_len=%d",
            summary_id,
            subject_id,
            theme_id,
            len(file_ids),
            len(additional_req),
        )

        s3_keys: List[str] = []
        try:
            work_dir = self._work_dir("summary", summary_id)
            logger.info("SummaryGen work dir prepared summaryId=%s path=%s", summary_id, work_dir)

            theme = await self.db.get_theme_name(theme_id)
            theme_name: str = str(theme)
            logger.info("SummaryGen theme resolved summaryId=%s theme=%s", summary_id, theme_name)

            summary_file_id = uuid4()
            # 1) s3Index из БД
            s3_keys = await self.db.get_s3_keys_for_file_ids(file_ids)
            logger.info("SummaryGen S3 keys loaded summaryId=%s count=%d", summary_id, len(s3_keys))
            if len(s3_keys) != len(file_ids):
                logger.warning(
                    "SummaryGen S3 key count mismatch summaryId=%s requested_files=%d found_keys=%d",
                    summary_id,
                    len(file_ids),
                    len(s3_keys),
                )

            local_files = await self._download_s3_keys(s3_keys, work_dir, "SummaryGen", summary_id)
            logger.info("SummaryGen files downloaded summaryId=%s count=%d", summary_id, len(local_files))

            pdf_files = [Path(f) for f in local_files if f and f.lower().endswith(".pdf")]
            if not pdf_files:
                raise RuntimeError("Не найден PDF-файл среди materials для SummaryGen")

            pdf_path = pdf_files[0]
            logger.info("SummaryGen selected PDF summaryId=%s path=%s", summary_id, pdf_path)

            doc, pages = await asyncio.to_thread(load_pdf_document, str(pdf_path))
            chunks = await asyncio.to_thread(chunk_document_pages, doc, pages, max_tokens=700)
            logger.info(
                "SummaryGen PDF parsed summaryId=%s document=%s pages=%d chunks=%d",
                summary_id,
                doc.id,
                doc.pages,
                len(chunks),
            )
            retriever = HybridRetriever(alpha=0.7)
            await asyncio.to_thread(retriever.index, chunks)

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
            logger.info(
                "SummaryGen lecture plan built summaryId=%s sections=%d",
                summary_id,
                len(plan.sections),
            )

            lecture_md = await generate_lecture_markdown(
                plan=plan,
                retriever=retriever,
                topic_description=topic.description,
                max_tokens_per_section=1500,
                top_k_chunks_per_section=5,
            )
            logger.info("SummaryGen lecture markdown generated summaryId=%s chars=%d", summary_id, len(lecture_md))

            summary_path = work_dir / f"summary_{summary_id}.md"
            await asyncio.to_thread(summary_path.write_text, lecture_md, encoding="utf-8")

            pdf_path = work_dir / f"summary_{summary_id}.pdf"
            await asyncio.to_thread(markdown_to_pdf, summary_path, pdf_path)
            logger.info("SummaryGen PDF exported summaryId=%s path=%s", summary_id, pdf_path)

            file_name = f"auto conspect on theme.pdf"

            s3_key = await asyncio.to_thread(
                self.s3.upload_file_to_bucket,
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
            logger.info("SummaryGen metadata saved summaryId=%s fileId=%s s3_key=%s", summary_id, summary_file_id,
                        s3_key)

            await self._publish(
                self.rabbit.queue_summary_gen_complete,
                {
                    "summaryId": str(summary_id),
                    "subjectId": subject_id,
                    "themeId": theme_id,
                    "status": "SUCCESS",
                    "error": "",
                },
            )
            logger.info("SummaryGen completed summaryId=%s status=SUCCESS", summary_id)


        except Exception as e:
            logger.exception("SummaryGen failed summaryId=%s error=%s", summary_id, e)
            await self._publish(
                self.rabbit.queue_summary_gen_complete,
                {
                    "summaryId": str(summary_id),
                    "subjectId": subject_id,
                    "themeId": theme_id,
                    "status": "FAILED",
                    "error": str(e),
                },
            )
        finally:
            await asyncio.to_thread(self.s3.finish_task_cache_usage, s3_keys)

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
        raw_qtypes = [str(t).lower() for t in raw_qtypes]
        additional_req = payload.get("additional_requirements") or ""
        theme_id_raw = payload.get("themeId")
        theme_id: Optional[int] = int(theme_id_raw) if theme_id_raw is not None else None
        logger.info(
            "QuizGen started quizId=%s themeId=%s files=%d difficulty=%s question_count=%s question_types=%s additional_requirements_len=%d",
            quiz_id,
            theme_id,
            len(file_ids),
            difficulty,
            question_count,
            raw_qtypes,
            len(additional_req),
        )

        internal_allowed: set[QuestionType] = set()
        for t in raw_qtypes:
            t_norm = t.lower()
            if t_norm in UI_TO_INTERNAL:
                internal_allowed.add(UI_TO_INTERNAL[t_norm])

        if not internal_allowed:
            internal_allowed = {"multiple_choice", "matching", "true_false"}

        # Допустимые типы для БД (multichoice|essay|matching|truefalse|shortanswer|numerical)
        allowed_db_types: set[str] = set()
        if "true_false" in internal_allowed:
            allowed_db_types.add("truefalse")
        if "multiple_choice" in internal_allowed or "select_all_that_apply" in internal_allowed:
            allowed_db_types.add("multichoice")
        if "matching" in internal_allowed:
            allowed_db_types.add("matching")
        if "short_answer" in internal_allowed or "fill_in_the_blank" in internal_allowed:
            allowed_db_types.add("shortanswer")
        if "long_answer" in internal_allowed:
            allowed_db_types.add("essay")
        # if "numerical" in allowed:
        #     allowed_db_types.add("shortanswer")

        if not allowed_db_types:
            allowed_db_types = {"multichoice", "matching", "truefalse"}

        theme_name: Optional[str] = None

        s3_keys: List[str] = []
        try:
            work_dir = self._work_dir("quiz", quiz_id)
            logger.info("QuizGen work dir prepared quizId=%s path=%s", quiz_id, work_dir)

            if theme_id is not None:
                theme_name = await self.db.get_theme_name(theme_id)
                logger.info("QuizGen theme resolved quizId=%s theme=%s", quiz_id, theme_name)

            # 1) s3Index → S3
            s3_keys = await self.db.get_s3_keys_for_file_ids(file_ids)
            logger.info("QuizGen S3 keys loaded quizId=%s count=%d", quiz_id, len(s3_keys))
            if len(s3_keys) != len(file_ids):
                logger.warning(
                    "QuizGen S3 key count mismatch quizId=%s requested_files=%d found_keys=%d",
                    quiz_id,
                    len(file_ids),
                    len(s3_keys),
                )
            local_files = await self._download_s3_keys(s3_keys, work_dir, "QuizGen", quiz_id)
            logger.info("QuizGen files downloaded quizId=%s count=%d", quiz_id, len(local_files))

            note_text = ""
            for lf in local_files:
                if not lf:
                    continue
                p = Path(lf)
                if p.suffix.lower() in (".md", ".txt"):
                    file_text = await asyncio.to_thread(p.read_text, encoding="utf-8")
                    sampled_text = _sample_text(file_text, settings.quiz_source_max_chars_per_file)
                    note_text += sampled_text + "\n\n"
                    logger.info(
                        "QuizGen text file read quizId=%s path=%s chars=%d sampled_chars=%d",
                        quiz_id,
                        p,
                        len(file_text),
                        len(sampled_text),
                    )
                elif p.suffix.lower() == ".pdf":
                    try:
                        pdf_text = await asyncio.to_thread(extract_text_from_pdf, str(p))
                    except Exception as e:
                        logger.exception("QuizGen PDF text extraction failed quizId=%s path=%s error=%s", quiz_id, p, e)
                        raise RuntimeError(f"Не удалось прочитать PDF '{p}': {e}")

                    if pdf_text.strip():
                        sampled_text = _sample_text(pdf_text, settings.quiz_source_max_chars_per_file)
                        note_text += sampled_text + "\n\n"
                        logger.info(
                            "QuizGen PDF text extracted quizId=%s path=%s chars=%d sampled_chars=%d",
                            quiz_id,
                            p,
                            len(pdf_text),
                            len(sampled_text),
                        )

            if not note_text.strip():
                raise RuntimeError("Нет текстового содержимого для генерации квиза")
            if len(note_text) > settings.quiz_source_max_chars:
                original_chars = len(note_text)
                note_text = _sample_text(note_text, settings.quiz_source_max_chars)
                logger.info(
                    "QuizGen source text sampled quizId=%s original_chars=%d sampled_chars=%d",
                    quiz_id,
                    original_chars,
                    len(note_text),
                )
            logger.info("QuizGen source text ready quizId=%s chars=%d", quiz_id, len(note_text))

            ui_counts = _distribute_question_counts(
                total=question_count,
                ui_types=raw_qtypes,
            )
            logger.info("QuizGen question distribution quizId=%s counts=%s", quiz_id, ui_counts)

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
            logger.info("QuizGen raw questions generated quizId=%s count=%d", quiz_id, len(raw_questions))
            quiz_questions = self._convert_raw_to_quiz_questions(raw_questions, allowed_types=internal_allowed)
            logger.info("QuizGen questions converted quizId=%s count=%d", quiz_id, len(quiz_questions))

            await self._generate_explanations_for_quiz(
                lecture_md=note_text,
                questions=quiz_questions,
                difficulty=difficulty,
            )
            logger.info("QuizGen explanations generated quizId=%s", quiz_id)

            await self._persist_quiz(
                quiz_id=quiz_id,
                theme_id=theme_id,
                questions=quiz_questions,
                file_ids=file_ids,
                theme_name=theme_name,
            )
            logger.info("QuizGen persisted quizId=%s questions=%d", quiz_id, len(quiz_questions))

            await self._publish(
                self.rabbit.queue_quiz_gen_complete,
                {
                    "quizId": str(quiz_id),
                    "status": "SUCCESS",
                    "error": "",
                },
            )
            logger.info("QuizGen completed quizId=%s status=SUCCESS", quiz_id)


        except Exception as e:
            logger.exception("QuizGen failed quizId=%s error=%s", quiz_id, e)
            await self._publish(
                self.rabbit.queue_quiz_gen_complete,
                {
                    "quizId": str(quiz_id),
                    "status": "FAILED",
                    "error": str(e),
                },
            )
        finally:
            await asyncio.to_thread(self.s3.finish_task_cache_usage, s3_keys)
