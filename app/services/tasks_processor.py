from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from uuid import uuid4, UUID

import numpy as np
from pydantic import ValidationError

from app.api.core.config import settings
from app.curriculum.models import LectureTopic, DifficultyLevel
from app.documents.chunking import chunk_document_pages
from app.documents.docx_reader import extract_docx_text
from app.documents.index_cache import DocumentIndexCache, DocumentIndexData
from app.documents.indexers import EmbeddingsRetriever, HybridRetriever, TfidfRetriever
from app.documents.pdf_reader import load_pdf_document
from app.faq import FAQGenerationConfig, format_faq_as_markdown, generate_faq_from_text
from app.lectures.context_policy import build_context_policy
from app.lectures.context_selection import build_document_profiles
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
from app.services.contracts import (
    QuizGenComplete,
    QuizGenRequest,
    SummaryGenComplete,
    SummaryGenRequest,
    to_payload,
)
from app.services.embeddings_client import OpenAIEmbeddingsClient
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


@dataclass
class LectureSource:
    summary_id: int
    subject_id: int | None
    theme_id: int | None
    source_file_id: UUID
    s3_key: str
    path: Path
    text: str


def _extract_source_text(file_path: Path) -> str:
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
        raise RuntimeError("Source file has no text content")
    return text


_extract_faq_source_text = _extract_source_text


def _dedupe_file_ids(values: list[UUID | str]) -> list[UUID]:
    seen: set[UUID] = set()
    result: list[UUID] = []
    for value in values:
        file_id = value if isinstance(value, UUID) else UUID(str(value))
        if file_id in seen:
            continue
        seen.add(file_id)
        result.append(file_id)
    return result


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
            local_file = await asyncio.to_thread(
                self.s3.download_to_materials,
                key,
                work_dir,
                str(task_id),
            )
            local_files.append(local_file)
        return local_files

    async def _load_lecture_source(
            self,
            summary_id: int,
            work_dir: Path,
            task_kind: str,
            task_id: UUID,
    ) -> LectureSource:
        summary_context = await self.db.get_summary_context(summary_id)
        if summary_context is None:
            raise RuntimeError(f"Summary not found: {summary_id}")

        theme_id = summary_context.get("themeId")
        subject_id = summary_context.get("subjectId")
        source_file_id_raw = summary_context.get("sourceFileId")
        lecture_s3_key = str(summary_context.get("lectureS3Key") or "").strip()
        if theme_id is None:
            raise RuntimeError(f"Summary {summary_id} has no themeId")
        if source_file_id_raw is None:
            raise RuntimeError(f"Summary {summary_id} has no source fileId")
        if not lecture_s3_key:
            raise RuntimeError(f"Summary {summary_id} source file has no s3Index")

        source_file_id = source_file_id_raw if isinstance(source_file_id_raw, UUID) else UUID(str(source_file_id_raw))
        logger.info(
            "%s lecture source resolved taskId=%s summaryId=%s subjectId=%s themeId=%s sourceFileId=%s key=%s",
            task_kind,
            task_id,
            summary_id,
            subject_id,
            theme_id,
            source_file_id,
            lecture_s3_key,
        )

        local_file = await asyncio.to_thread(
            self.s3.download_to_materials,
            lecture_s3_key,
            work_dir,
            str(task_id),
        )
        source_path = Path(local_file)
        logger.info("%s lecture source downloaded taskId=%s path=%s", task_kind, task_id, source_path)

        source_text = await asyncio.to_thread(_extract_source_text, source_path)
        if len(source_text) > settings.quiz_source_max_chars:
            original_chars = len(source_text)
            source_text = _sample_text(source_text, settings.quiz_source_max_chars)
            logger.info(
                "%s lecture source text sampled taskId=%s source_chars=%d used_chars=%d total_limit=%d",
                task_kind,
                task_id,
                original_chars,
                len(source_text),
                settings.quiz_source_max_chars,
            )

        logger.info(
            "%s lecture source text ready taskId=%s used_chars=%d",
            task_kind,
            task_id,
            len(source_text),
        )
        return LectureSource(
            summary_id=summary_id,
            subject_id=int(subject_id) if subject_id is not None else None,
            theme_id=int(theme_id) if theme_id is not None else None,
            source_file_id=source_file_id,
            s3_key=lecture_s3_key,
            path=source_path,
            text=source_text,
        )

    @staticmethod
    def _build_pdf_document_index(
            pdf_path: Path,
            file_id: str,
            file_name: str | None,
            chunk_tokens: int,
            embedding_model: str,
    ) -> DocumentIndexData:
        started = time.perf_counter()
        doc, pages = load_pdf_document(str(pdf_path), file_id, file_name)
        parsed_at = time.perf_counter()
        chunks = chunk_document_pages(doc, pages, max_tokens=chunk_tokens)
        chunked_at = time.perf_counter()
        embedding_chunks = EmbeddingsRetriever.prepare_embedding_chunks(chunks, max_tokens=chunk_tokens)

        if embedding_chunks:
            embeddings_client = OpenAIEmbeddingsClient(model=embedding_model)
            embeddings = np.asarray(
                embeddings_client.embed_texts([chunk.text for chunk in embedding_chunks]),
                dtype="float32",
            )
        else:
            embeddings = np.empty((0, 0), dtype="float32")

        embedded_at = time.perf_counter()
        logger.info(
            "Document index built path=%s fileId=%s pages=%d chunks=%d embedding_chunks=%d parse_seconds=%.3f chunk_seconds=%.3f embedding_seconds=%.3f total_seconds=%.3f",
            pdf_path,
            file_id,
            doc.pages,
            len(chunks),
            len(embedding_chunks),
            parsed_at - started,
            chunked_at - parsed_at,
            embedded_at - chunked_at,
            embedded_at - started,
        )
        return DocumentIndexData(
            document=doc,
            chunks=chunks,
            embedding_chunks=embedding_chunks,
            embeddings=embeddings,
        )

    @staticmethod
    def _build_summary_retriever(
            all_chunks: list,
            embedding_chunks: list | None = None,
            embeddings: np.ndarray | None = None,
    ) -> HybridRetriever:
        if embedding_chunks is not None and embeddings is not None:
            tfidf = TfidfRetriever(stop_words=None)
            tfidf.index(all_chunks)
            embeddings_retriever = EmbeddingsRetriever()
            embeddings_retriever.index_precomputed(embedding_chunks, embeddings)
            return HybridRetriever(
                tfidf_retriever=tfidf,
                embeddings_retriever=embeddings_retriever,
                alpha=0.7,
            )

        retriever = HybridRetriever(alpha=0.7)
        retriever.index(all_chunks)
        return retriever

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

            work_dir = self._work_dir("faq", faq_id)
            lecture_source = await self._load_lecture_source(summary_id, work_dir, "FAQGen", faq_id)
            theme_id = lecture_source.theme_id
            subject_id = lecture_source.subject_id
            source_file_id = lecture_source.source_file_id
            s3_keys = [lecture_source.s3_key]
            if settings.faq_generation_use_source_documents:
                logger.warning(
                    "FAQ_GENERATION_USE_SOURCE_DOCUMENTS=true is not supported with current summary context; "
                    "FAQGen will use lecture text only faqId=%s summaryId=%s",
                    faq_id,
                    summary_id,
                )
            logger.info(
                "FAQGen context resolved faqId=%s summaryId=%s subjectId=%s themeId=%s sourceFileId=%s work_dir=%s",
                faq_id,
                summary_id,
                subject_id,
                theme_id,
                source_file_id,
                work_dir,
            )

            source_text = lecture_source.text

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
            await asyncio.to_thread(
                self.s3.finish_task_cache_usage,
                s3_keys,
                str(faq_id) if faq_id is not None else None,
            )

    async def handle_summary_gen(self, payload: dict) -> None:
        """
        SummaryGen:
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            userId: uuid,
            files: uuid[],
            additional_requirements: text
        }
        """
        s3_keys: List[str] = []
        summary_id = str(payload.get("summaryId", ""))
        subject_id = payload.get("subjectId")
        theme_id = payload.get("themeId")
        user_id = str(payload.get("userId", ""))

        try:
            request = SummaryGenRequest.model_validate(payload)
            summary_id = request.summary_id
            subject_id = request.subject_id
            theme_id = request.theme_id
            user_id = request.user_id
            file_ids = request.files
            additional_req = request.additional_requirements or ""
            logger.info(
                "SummaryGen started summaryId=%s subjectId=%s themeId=%s userId=%s files=%d additional_requirements_len=%d",
                summary_id,
                subject_id,
                theme_id,
                user_id,
                len(file_ids),
                len(additional_req),
            )

            work_dir = self._work_dir("summary", summary_id)
            logger.info("SummaryGen work dir prepared summaryId=%s path=%s", summary_id, work_dir)

            theme = await self.db.get_theme_name(theme_id)
            theme_name: str = str(theme)
            logger.info("SummaryGen theme resolved summaryId=%s theme=%s", summary_id, theme_name)

            summary_file_id = uuid4()
            file_records = await self.db.get_file_records_for_file_ids(file_ids)
            if len(file_records) != len(file_ids):
                found_ids = {record.file_id for record in file_records}
                missing = [str(file_id) for file_id in file_ids if file_id not in found_ids]
                raise RuntimeError(f"SummaryGen source files not found in DB: {missing}")

            s3_keys = [record.s3_index for record in file_records]
            logger.info("SummaryGen file records loaded summaryId=%s count=%d", summary_id, len(file_records))

            local_files = await self._download_s3_keys(s3_keys, work_dir, "SummaryGen", summary_id)
            logger.info("SummaryGen files downloaded summaryId=%s count=%d", summary_id, len(local_files))

            document_index_cache = DocumentIndexCache() if settings.document_index_cache_enabled else None
            if document_index_cache is None:
                logger.info("Document index cache disabled for SummaryGen summaryId=%s", summary_id)
            else:
                logger.info(
                    "Document index cache enabled for SummaryGen summaryId=%s db_path=%s",
                    summary_id,
                    document_index_cache.db_path,
                )

            documents = []
            chunks_by_doc = {}
            all_chunks = []
            all_embedding_chunks = []
            embedding_arrays = []
            for record, local_file in zip(file_records, local_files):
                if not local_file:
                    continue
                pdf_path = Path(local_file)
                if pdf_path.suffix.lower() != ".pdf":
                    logger.warning(
                        "SummaryGen skipped non-PDF file summaryId=%s fileId=%s path=%s",
                        summary_id,
                        record.file_id,
                        pdf_path,
                    )
                    continue

                if document_index_cache is not None:
                    cache_entry = await asyncio.to_thread(
                        document_index_cache.get_or_build,
                        file_path=pdf_path,
                        file_id=str(record.file_id),
                        file_name=record.name,
                        s3_index=record.s3_index,
                        chunk_tokens=settings.lecture_chunk_tokens,
                        embedding_model=settings.embedding_model,
                        builder=lambda pdf_path=pdf_path, record=record: self._build_pdf_document_index(
                            pdf_path,
                            str(record.file_id),
                            record.name,
                            settings.lecture_chunk_tokens,
                            settings.embedding_model,
                        ),
                    )
                    doc = cache_entry.document
                    chunks = cache_entry.chunks
                    all_embedding_chunks.extend(cache_entry.embedding_chunks)
                    if cache_entry.embeddings.shape[0] > 0:
                        embedding_arrays.append(cache_entry.embeddings)
                    logger.info(
                        "SummaryGen PDF index ready summaryId=%s fileId=%s document=%s pages=%d chunks=%d embedding_chunks=%d cache_hit=%s",
                        summary_id,
                        record.file_id,
                        doc.id,
                        doc.pages,
                        len(chunks),
                        len(cache_entry.embedding_chunks),
                        cache_entry.cache_hit,
                    )
                else:
                    parsed_started = time.perf_counter()
                    doc, pages = await asyncio.to_thread(
                        load_pdf_document,
                        str(pdf_path),
                        str(record.file_id),
                        record.name,
                    )
                    chunks = await asyncio.to_thread(
                        chunk_document_pages,
                        doc,
                        pages,
                        max_tokens=settings.lecture_chunk_tokens,
                    )
                    logger.info(
                        "SummaryGen PDF parsed summaryId=%s fileId=%s document=%s pages=%d chunks=%d seconds=%.3f",
                        summary_id,
                        record.file_id,
                        doc.id,
                        doc.pages,
                        len(chunks),
                        time.perf_counter() - parsed_started,
                    )
                documents.append(doc)
                chunks_by_doc[doc.id] = chunks
                all_chunks.extend(chunks)

            if not documents:
                raise RuntimeError("No PDF files found for SummaryGen")
            if not all_chunks:
                raise RuntimeError("No text chunks extracted from selected PDF files")

            policy = build_context_policy(
                doc_count=len(documents),
                total_chunks=len(all_chunks),
                document_profiles_enabled=settings.lecture_document_profiles_enabled,
            )
            logger.info(
                "SummaryGen context policy summaryId=%s target_words=%d sections=%d plan_chunks=%d section_chunks=%d plan_pool=%d section_pool=%d profiles_enabled=%s",
                summary_id,
                policy.target_words,
                policy.target_sections,
                policy.plan_context_chunks,
                policy.section_context_chunks,
                policy.plan_pool_k,
                policy.section_pool_k,
                settings.lecture_document_profiles_enabled,
            )

            retriever_started = time.perf_counter()
            if document_index_cache is not None:
                all_embeddings = (
                    np.vstack(embedding_arrays)
                    if embedding_arrays
                    else np.empty((0, 0), dtype="float32")
                )
                if all_embeddings.shape[0] != len(all_embedding_chunks):
                    raise RuntimeError(
                        "Cached embeddings count mismatch: "
                        f"chunks={len(all_embedding_chunks)} embeddings={all_embeddings.shape[0]}"
                    )
                retriever = await asyncio.to_thread(
                    self._build_summary_retriever,
                    all_chunks,
                    all_embedding_chunks,
                    all_embeddings,
                )
                logger.info(
                    "SummaryGen retriever built from document index cache summaryId=%s chunks=%d embedding_chunks=%d seconds=%.3f",
                    summary_id,
                    len(all_chunks),
                    len(all_embedding_chunks),
                    time.perf_counter() - retriever_started,
                )
            else:
                retriever = await asyncio.to_thread(self._build_summary_retriever, all_chunks)
                logger.info(
                    "SummaryGen retriever built without document index cache summaryId=%s chunks=%d seconds=%.3f",
                    summary_id,
                    len(all_chunks),
                    time.perf_counter() - retriever_started,
                )

            topic = LectureTopic(
                id=str(summary_file_id),
                title=f"Авто-конспект по теме {theme_name}",
                description="Автоматически сгенерированный конспект.",
                difficulty=DifficultyLevel.MEDIUM,
                keywords=[],
                duration_min=90,
                source_docs=[doc.id for doc in documents],
                order=1,
            )

            document_profiles = []
            if settings.lecture_document_profiles_enabled:
                document_profiles = await build_document_profiles(
                    documents,
                    chunks_by_doc,
                    topic=topic,
                    additional_requirements=additional_req,
                    chunks_per_doc=policy.profile_chunks_per_doc,
                    max_tokens=settings.lecture_doc_profile_max_tokens,
                )
                logger.info(
                    "SummaryGen document profiles built summaryId=%s count=%d",
                    summary_id,
                    len(document_profiles),
                )

            plan = await build_lecture_plan_for_topic(
                topic,
                retriever=retriever,
                top_k_chunks=policy.plan_context_chunks,
                retrieval_pool_k=policy.plan_pool_k,
                context_token_budget=policy.plan_context_token_budget,
                document_profiles=document_profiles,
                additional_requirements=additional_req,
                min_sections=policy.target_sections,
                max_sections=policy.target_sections,
            )
            logger.info("SummaryGen lecture plan built summaryId=%s sections=%d", summary_id, len(plan.sections))

            lecture_md = await generate_lecture_markdown(
                plan=plan,
                retriever=retriever,
                topic_description=topic.description,
                max_tokens_per_section=policy.section_output_tokens,
                top_k_chunks_per_section=policy.section_context_chunks,
                retrieval_pool_k=policy.section_pool_k,
                context_token_budget=policy.section_context_token_budget,
                additional_requirements=additional_req,
                final_edit_enabled=settings.lecture_final_edit_enabled,
                final_edit_input_token_budget=policy.final_edit_input_token_budget,
            )
            logger.info("SummaryGen lecture markdown generated summaryId=%s chars=%d", summary_id, len(lecture_md))

            summary_path = work_dir / f"summary_{summary_id}.md"
            await asyncio.to_thread(summary_path.write_text, lecture_md, encoding="utf-8")

            pdf_path = work_dir / f"summary_{summary_id}.pdf"
            await asyncio.to_thread(markdown_to_pdf, summary_path, pdf_path)
            logger.info("SummaryGen PDF exported summaryId=%s path=%s", summary_id, pdf_path)

            file_name = "auto conspect on theme.pdf"
            s3_key = await asyncio.to_thread(
                self.s3.upload_file_to_bucket,
                local_path=str(pdf_path),
                original_name=file_name,
                bucket="summaries",
                user_id=str(user_id),
            )

            await self.db.save_summary_text(
                summary_id=summary_file_id,
                theme_id=theme_id,
                s3_key=s3_key,
                file_name=file_name,
                user_id=user_id,
            )
            logger.info("SummaryGen metadata saved summaryId=%s fileId=%s s3_key=%s", summary_id, summary_file_id, s3_key)

            await self._publish(
                self.rabbit.queue_summary_gen_complete,
                to_payload(
                    SummaryGenComplete(
                        summaryId=summary_id,
                        subjectId=subject_id,
                        themeId=theme_id,
                        userId=user_id,
                        status="SUCCESS",
                        error="",
                    )
                ),
            )
            logger.info("SummaryGen completed summaryId=%s status=SUCCESS", summary_id)

        except Exception as e:
            logger.exception("SummaryGen failed summaryId=%s error=%s", summary_id, e)
            await self._publish(
                self.rabbit.queue_summary_gen_complete,
                to_payload(
                    SummaryGenComplete(
                        summaryId=summary_id,
                        subjectId=subject_id,
                        themeId=theme_id,
                        userId=user_id,
                        status="FAILED",
                        error=str(e),
                    )
                ),
            )
        finally:
            await asyncio.to_thread(
                self.s3.finish_task_cache_usage,
                s3_keys,
                str(summary_id) if summary_id is not None else None,
            )

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
        quiz_id = str(payload.get("quizId", ""))
        user_id = str(payload.get("userId", ""))
        try:
            request = QuizGenRequest.model_validate(payload)
        except Exception as e:
            error_message = str(e)
            if isinstance(e, ValidationError):
                for error in e.errors():
                    if tuple(error.get("loc", ())) == ("summaryId",) and error.get("type") == "missing":
                        error_message = "Invalid QuizGen payload: required field summaryId is missing"
                        break
            logger.exception("QuizGen payload validation failed quizId=%s error=%s", quiz_id, error_message)
            await self._publish(
                self.rabbit.queue_quiz_gen_complete,
                to_payload(
                    QuizGenComplete(
                        quizId=quiz_id,
                        userId=user_id,
                        status="FAILED",
                        error=error_message,
                    )
                ),
            )
            return

        quiz_id = request.quiz_id
        user_id = request.user_id
        summary_id = request.summary_id
        file_ids = request.files
        difficulty = request.difficulty
        question_count = request.question_count
        raw_qtypes = request.question_types or ["multichoice", "matching", "truefalse"]
        raw_qtypes = [str(t).lower() for t in raw_qtypes]
        additional_req = request.additional_requirements or ""
        theme_id_raw = payload.get("themeId")
        theme_id: Optional[int] = int(theme_id_raw) if theme_id_raw is not None else None
        logger.info(
            "QuizGen started quizId=%s userId=%s summaryId=%s themeId=%s files=%d difficulty=%s question_count=%s question_types=%s additional_requirements_len=%d",
            quiz_id,
            user_id,
            summary_id,
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

            lecture_source = await self._load_lecture_source(summary_id, work_dir, "QuizGen", quiz_id)
            s3_keys = [lecture_source.s3_key]
            if theme_id is None:
                theme_id = lecture_source.theme_id

            if theme_id is not None:
                theme_name = await self.db.get_theme_name(theme_id)
                logger.info("QuizGen theme resolved quizId=%s theme=%s", quiz_id, theme_name)

            source_mode = (
                "lecture_and_documents"
                if settings.quiz_generation_use_source_documents
                else "lecture_only"
            )
            note_text = lecture_source.text + "\n\n"
            reference_file_ids: list[UUID] = [lecture_source.source_file_id]
            document_texts_used = 0
            logger.info(
                "QuizGen source mode quizId=%s source_mode=%s lecture_chars=%d requested_files=%d",
                quiz_id,
                source_mode,
                len(lecture_source.text),
                len(file_ids),
            )

            file_records = []
            local_files = []
            if settings.quiz_generation_use_source_documents:
                file_records = await self.db.get_file_records_for_file_ids(file_ids)
                if len(file_records) != len(file_ids):
                    found_ids = {record.file_id for record in file_records}
                    missing = [str(file_id) for file_id in file_ids if file_id not in found_ids]
                    raise RuntimeError(f"QuizGen source files not found in DB: {missing}")

                document_s3_keys = [record.s3_index for record in file_records]
                s3_keys.extend(document_s3_keys)
                logger.info("QuizGen file records loaded quizId=%s count=%d", quiz_id, len(file_records))
                local_files = await self._download_s3_keys(document_s3_keys, work_dir, "QuizGen", quiz_id)
                logger.info("QuizGen files downloaded quizId=%s count=%d", quiz_id, len(local_files))
            else:
                logger.info(
                    "QuizGen source documents disabled quizId=%s ignored_payload_files=%d",
                    quiz_id,
                    len(file_ids),
                )
            for lf in local_files:
                if not lf:
                    continue
                p = Path(lf)
                if p.suffix.lower() in (".md", ".txt"):
                    file_text = await asyncio.to_thread(p.read_text, encoding="utf-8")
                    sampled_text = _sample_text(file_text, settings.quiz_source_max_chars_per_file)
                    note_text += sampled_text + "\n\n"
                    document_texts_used += 1
                    logger.info(
                        "QuizGen text file read quizId=%s path=%s source_chars=%d used_chars=%d per_file_limit=%d",
                        quiz_id,
                        p,
                        len(file_text),
                        len(sampled_text),
                        settings.quiz_source_max_chars_per_file,
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
                        document_texts_used += 1
                        logger.info(
                            "QuizGen PDF text extracted quizId=%s path=%s source_chars=%d used_chars=%d per_file_limit=%d",
                            quiz_id,
                            p,
                            len(pdf_text),
                            len(sampled_text),
                            settings.quiz_source_max_chars_per_file,
                        )

            if file_records:
                reference_file_ids = _dedupe_file_ids(reference_file_ids + [record.file_id for record in file_records])

            if not note_text.strip():
                raise RuntimeError("Нет текстового содержимого для генерации квиза")
            if len(note_text) > settings.quiz_source_max_chars:
                original_chars = len(note_text)
                note_text = _sample_text(note_text, settings.quiz_source_max_chars)
                logger.info(
                    "QuizGen source text sampled quizId=%s source_chars=%d used_chars=%d total_limit=%d",
                    quiz_id,
                    original_chars,
                    len(note_text),
                    settings.quiz_source_max_chars,
                )
            logger.info(
                "QuizGen source text ready quizId=%s source_mode=%s used_chars=%d document_texts_used=%d references=%d",
                quiz_id,
                source_mode,
                len(note_text),
                document_texts_used,
                len(reference_file_ids),
            )

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
                file_ids=reference_file_ids,
                theme_name=theme_name,
            )
            logger.info("QuizGen persisted quizId=%s questions=%d", quiz_id, len(quiz_questions))

            await self._publish(
                self.rabbit.queue_quiz_gen_complete,
                to_payload(
                    QuizGenComplete(
                        quizId=quiz_id,
                        userId=user_id,
                        status="SUCCESS",
                        error="",
                    )
                ),
            )
            logger.info("QuizGen completed quizId=%s status=SUCCESS", quiz_id)


        except Exception as e:
            logger.exception("QuizGen failed quizId=%s error=%s", quiz_id, e)
            await self._publish(
                self.rabbit.queue_quiz_gen_complete,
                to_payload(
                    QuizGenComplete(
                        quizId=quiz_id,
                        userId=user_id,
                        status="FAILED",
                        error=str(e),
                    )
                ),
            )
        finally:
            await asyncio.to_thread(self.s3.finish_task_cache_usage, s3_keys, str(quiz_id) if quiz_id else None)
