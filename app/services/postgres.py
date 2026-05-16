from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, List, Optional
from uuid import UUID, uuid4

import asyncpg

from app.api.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    file_id: UUID
    name: str
    s3_index: str


@dataclass(frozen=True)
class QuizAnswerDialogMessage:
    message_id: UUID
    dialog_id: UUID
    user_id: UUID | None
    role: str
    content: str
    sequence_no: int
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class QuizAnswerDialogVariant:
    link_id: UUID
    variant_id: UUID | None
    text: str | None
    is_right: bool | None
    explain_right: str | None
    explain_wrong: str | None
    left_matching: str | None
    right_matching: str | None


@dataclass(frozen=True)
class QuizAnswerDialogChosenAnswer:
    submit_id: UUID
    chosen_id: UUID | None
    answer: str | None
    answer_left: str | None
    answer_right: str | None
    is_right: bool | None
    explanation: str | None
    variant_text: str | None
    variant_is_right: bool | None
    explain_right: str | None
    explain_wrong: str | None
    left_matching: str | None
    right_matching: str | None


@dataclass(frozen=True)
class QuizAnswerDialogContext:
    dialog_id: UUID
    user_id: UUID
    question_submission_id: UUID
    session_id: UUID
    quiz_id: UUID
    question_id: UUID
    context_snapshot: dict[str, Any]
    question_type: str
    question_text: str
    question_multi_answer: bool | None
    submission_is_right: bool | None
    quiz_summary_id: int | None
    summary_file_record: FileRecord | None
    current_message: QuizAnswerDialogMessage
    messages: list[QuizAnswerDialogMessage]
    variants: list[QuizAnswerDialogVariant]
    chosen_answers: list[QuizAnswerDialogChosenAnswer]
    reference_file_ids: list[UUID]


class PostgresClient:
    """Async PostgreSQL client for worker persistence."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> asyncpg.Pool:
        if self._pool:
            return self._pool

        logger.info("Postgres pool connecting")
        self._pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
        )
        logger.info("Postgres pool connected min_size=%d max_size=%d", 1, 10)
        return self._pool

    @staticmethod
    def _json_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _message_from_row(row: asyncpg.Record) -> QuizAnswerDialogMessage:
        return QuizAnswerDialogMessage(
            message_id=row["id"],
            dialog_id=row["dialogId"],
            user_id=row["userId"],
            role=row["role"],
            content=row["content"],
            sequence_no=row["sequenceNo"],
            metadata=PostgresClient._json_dict(row["metadata"]),
        )

    @asynccontextmanager
    async def quiz_answer_dialog_lock(self, dialog_id: UUID, timeout_seconds: float):
        pool = await self.connect()
        timeout_ms = max(1, int(timeout_seconds * 1000))
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('lock_timeout', $1, true)", f"{timeout_ms}ms")
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", str(dialog_id))
                logger.info("Postgres quiz answer dialog lock acquired dialogId=%s", dialog_id)
                try:
                    yield
                finally:
                    logger.info("Postgres quiz answer dialog lock released dialogId=%s", dialog_id)

    async def get_s3_keys_for_file_ids(self, file_ids: List[UUID]) -> List[str]:
        pool = await self.connect()
        rows = await pool.fetch(
            """
            SELECT id, "s3Index"
            FROM thesis.files
            WHERE id = ANY ($1::uuid[])
            """,
            file_ids,
        )
        result = [r["s3Index"] for r in rows]
        logger.info("Postgres loaded S3 keys requested=%d found=%d", len(file_ids), len(result))
        for row in rows:
            logger.info("Postgres S3 key fileId=%s s3Index=%s", row["id"], row["s3Index"])
        return result

    async def get_file_records_for_file_ids(self, file_ids: List[UUID]) -> List[FileRecord]:
        pool = await self.connect()
        rows = await pool.fetch(
            """
            SELECT id, name, "s3Index"
            FROM thesis.files
            WHERE id = ANY ($1::uuid[])
            """,
            file_ids,
        )
        by_id = {
            row["id"]: FileRecord(
                file_id=row["id"],
                name=row["name"],
                s3_index=row["s3Index"],
            )
            for row in rows
        }
        result = [by_id[file_id] for file_id in file_ids if file_id in by_id]
        logger.info("Postgres loaded file records requested=%d found=%d", len(file_ids), len(result))
        for record in result:
            logger.info(
                "Postgres file record fileId=%s name=%s s3Index=%s",
                record.file_id,
                record.name,
                record.s3_index,
            )
        return result

    async def get_summary_context(self, summary_id: int) -> Optional[dict[str, Any]]:
        pool = await self.connect()
        logger.info("Postgres loading summary context summaryId=%s", summary_id)
        row = await pool.fetchrow(
            """
            SELECT
                s.id AS "summaryId",
                s."subjectId",
                s."themeId",
                s."fileId" AS "sourceFileId",
                f.name AS "sourceFileName",
                f."s3Index" AS "lectureS3Key"
            FROM thesis.summaries s
            JOIN thesis.files f ON f.id = s."fileId"
            WHERE s.id = $1
            """,
            summary_id,
        )
        if row is None:
            logger.warning("Postgres summary context not found summaryId=%s", summary_id)
            return None

        result = dict(row)
        logger.info(
            "Postgres summary context loaded summaryId=%s subjectId=%s themeId=%s sourceFileId=%s",
            summary_id,
            result.get("subjectId"),
            result.get("themeId"),
            result.get("sourceFileId"),
        )
        return result

    async def save_faq(
            self,
            *,
            faq_id: UUID,
            summary_id: int,
            theme_id: int,
            source_file_id: UUID,
            faq_file_id: UUID,
            s3_key: str,
            file_name: str,
            user_id: UUID,
            difficulty_level: str,
            num_questions: int,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres saving FAQ faqId=%s summaryId=%s themeId=%s fileId=%s sourceFileId=%s",
            faq_id,
            summary_id,
            theme_id,
            faq_file_id,
            source_file_id,
        )

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO thesis.files(id, name, "s3Index", "userId")
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        "s3Index" = EXCLUDED."s3Index",
                        "userId" = EXCLUDED."userId"
                    """,
                    faq_file_id,
                    file_name,
                    s3_key,
                    user_id,
                )

                await conn.execute(
                    """
                    INSERT INTO thesis.faqs(id, "themeId", "difficultyLevel", num_questions, "fileId", "summaryId")
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE
                    SET "themeId" = EXCLUDED."themeId",
                        "difficultyLevel" = EXCLUDED."difficultyLevel",
                        num_questions = EXCLUDED.num_questions,
                        "fileId" = EXCLUDED."fileId",
                        "summaryId" = EXCLUDED."summaryId"
                    """,
                    faq_id,
                    theme_id,
                    difficulty_level,
                    num_questions,
                    faq_file_id,
                    summary_id,
                )

                await conn.execute(
                    """
                    DELETE FROM thesis.references_faq
                    WHERE "faqId" = $1
                    """,
                    faq_id,
                )
                await conn.execute(
                    """
                    INSERT INTO thesis.references_faq("faqId", "fileId")
                    VALUES ($1, $2)
                    """,
                    faq_id,
                    source_file_id,
                )

        logger.info("Postgres FAQ saved faqId=%s fileId=%s", faq_id, faq_file_id)

    async def get_quiz_answer_dialog_context(
            self,
            *,
            dialog_id: UUID,
            user_id: UUID,
            message_id: UUID,
    ) -> QuizAnswerDialogContext | None:
        pool = await self.connect()
        logger.info(
            "Postgres loading quiz answer dialog context dialogId=%s userId=%s messageId=%s",
            dialog_id,
            user_id,
            message_id,
        )
        dialog_row = await pool.fetchrow(
            """
            SELECT
                d.id,
                d."questionSubmissionId",
                d."userId",
                d."sessionId",
                d."quizId",
                d."questionId",
                d."contextSnapshot",
                q.type AS "questionType",
                q.text AS "questionText",
                q."multiAnswer" AS "questionMultiAnswer",
                qs."isRight" AS "submissionIsRight",
                qu."summaryId" AS "quizSummaryId",
                sf.id AS "summaryFileId",
                sf.name AS "summaryFileName",
                sf."s3Index" AS "summaryS3Index"
            FROM thesis.quiz_answer_dialogs d
            JOIN thesis.questions q ON q.id = d."questionId"
            LEFT JOIN thesis.question_submissions qs ON qs.id = d."questionSubmissionId"
            LEFT JOIN thesis.quizes qu ON qu.id = d."quizId"
            LEFT JOIN thesis.summaries s ON s.id = qu."summaryId"
            LEFT JOIN thesis.files sf ON sf.id = s."fileId"
            WHERE d.id = $1 AND d."userId" = $2
            """,
            dialog_id,
            user_id,
        )
        if dialog_row is None:
            logger.warning("Postgres quiz answer dialog not found dialogId=%s userId=%s", dialog_id, user_id)
            return None

        current_row = await pool.fetchrow(
            """
            SELECT id, "dialogId", "userId", role, content, "sequenceNo", metadata
            FROM thesis.quiz_answer_dialog_messages
            WHERE id = $1 AND "dialogId" = $2 AND "userId" = $3
            """,
            message_id,
            dialog_id,
            user_id,
        )
        if current_row is None:
            logger.warning(
                "Postgres quiz answer dialog current message not found dialogId=%s userId=%s messageId=%s",
                dialog_id,
                user_id,
                message_id,
            )
            return None

        message_rows = await pool.fetch(
            """
            SELECT id, "dialogId", "userId", role, content, "sequenceNo", metadata
            FROM thesis.quiz_answer_dialog_messages
            WHERE "dialogId" = $1
            ORDER BY "sequenceNo" ASC
            """,
            dialog_id,
        )
        messages = [self._message_from_row(row) for row in message_rows]

        variant_rows = await pool.fetch(
            """
            SELECT
                qv.id AS "linkId",
                qv."variantId",
                qv."isRight",
                v.text,
                v."explainRight",
                v."explainWrong",
                v."leftMatching",
                v."rightMatching"
            FROM thesis.questions_variants qv
            LEFT JOIN thesis.variants v ON v.id = qv."variantId"
            WHERE qv."questionId" = $1
            ORDER BY v.text NULLS LAST, qv.id
            """,
            dialog_row["questionId"],
        )
        variants = [
            QuizAnswerDialogVariant(
                link_id=row["linkId"],
                variant_id=row["variantId"],
                text=row["text"],
                is_right=row["isRight"],
                explain_right=row["explainRight"],
                explain_wrong=row["explainWrong"],
                left_matching=row["leftMatching"],
                right_matching=row["rightMatching"],
            )
            for row in variant_rows
        ]

        chosen_rows = await pool.fetch(
            """
            SELECT
                cv.id AS "submitId",
                cv."chosenId",
                cv.answer,
                cv."answerLeft",
                cv."answerRight",
                cv."isRight",
                cv.explanation,
                qv."isRight" AS "variantIsRight",
                v.text AS "variantText",
                v."explainRight",
                v."explainWrong",
                v."leftMatching",
                v."rightMatching"
            FROM thesis.session_submits ss
            JOIN thesis.chosen_variants cv ON cv.id = ss."submitId"
            LEFT JOIN thesis.questions_variants qv ON qv.id = cv."chosenId"
            LEFT JOIN thesis.variants v ON v.id = qv."variantId"
            WHERE ss."sessionId" = $1
              AND cv."quizId" = $2
              AND cv."questionId" = $3
            ORDER BY cv.id
            """,
            dialog_row["sessionId"],
            dialog_row["quizId"],
            dialog_row["questionId"],
        )
        chosen_answers = [
            QuizAnswerDialogChosenAnswer(
                submit_id=row["submitId"],
                chosen_id=row["chosenId"],
                answer=row["answer"],
                answer_left=row["answerLeft"],
                answer_right=row["answerRight"],
                is_right=row["isRight"],
                explanation=row["explanation"],
                variant_text=row["variantText"],
                variant_is_right=row["variantIsRight"],
                explain_right=row["explainRight"],
                explain_wrong=row["explainWrong"],
                left_matching=row["leftMatching"],
                right_matching=row["rightMatching"],
            )
            for row in chosen_rows
        ]

        reference_rows = await pool.fetch(
            """
            SELECT "fileId"
            FROM thesis.references_question
            WHERE "questionId" = $1
            ORDER BY id
            """,
            dialog_row["questionId"],
        )
        if not reference_rows:
            reference_rows = await pool.fetch(
                """
                SELECT "fileId"
                FROM thesis.references_quiz
                WHERE "quizId" = $1
                ORDER BY id
                """,
                dialog_row["quizId"],
            )
        reference_file_ids = [row["fileId"] for row in reference_rows]

        summary_file_record = None
        if dialog_row["summaryFileId"] and dialog_row["summaryS3Index"]:
            summary_file_record = FileRecord(
                file_id=dialog_row["summaryFileId"],
                name=dialog_row["summaryFileName"] or "summary.pdf",
                s3_index=dialog_row["summaryS3Index"],
            )

        context = QuizAnswerDialogContext(
            dialog_id=dialog_row["id"],
            user_id=dialog_row["userId"],
            question_submission_id=dialog_row["questionSubmissionId"],
            session_id=dialog_row["sessionId"],
            quiz_id=dialog_row["quizId"],
            question_id=dialog_row["questionId"],
            context_snapshot=self._json_dict(dialog_row["contextSnapshot"]),
            question_type=dialog_row["questionType"],
            question_text=dialog_row["questionText"],
            question_multi_answer=dialog_row["questionMultiAnswer"],
            submission_is_right=dialog_row["submissionIsRight"],
            quiz_summary_id=dialog_row["quizSummaryId"],
            summary_file_record=summary_file_record,
            current_message=self._message_from_row(current_row),
            messages=messages,
            variants=variants,
            chosen_answers=chosen_answers,
            reference_file_ids=reference_file_ids,
        )
        logger.info(
            "Postgres quiz answer dialog context loaded dialogId=%s messages=%d variants=%d chosen=%d references=%d has_summary_file=%s",
            dialog_id,
            len(messages),
            len(variants),
            len(chosen_answers),
            len(reference_file_ids),
            summary_file_record is not None,
        )
        return context

    async def insert_quiz_answer_dialog_assistant_message(
            self,
            *,
            dialog_id: UUID,
            user_id: UUID,
            content: str,
            metadata: dict[str, Any] | None = None,
    ) -> QuizAnswerDialogMessage:
        pool = await self.connect()
        message_id = uuid4()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        async with pool.acquire() as conn:
            async with conn.transaction():
                sequence_no = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX("sequenceNo"), 0) + 1
                    FROM thesis.quiz_answer_dialog_messages
                    WHERE "dialogId" = $1
                    """,
                    dialog_id,
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO thesis.quiz_answer_dialog_messages(
                        id, "dialogId", "userId", role, content, "sequenceNo", metadata
                    )
                    VALUES ($1, $2, $3, 'assistant', $4, $5, $6::jsonb)
                    RETURNING id, "dialogId", "userId", role, content, "sequenceNo", metadata
                    """,
                    message_id,
                    dialog_id,
                    user_id,
                    content,
                    sequence_no,
                    metadata_json,
                )
                await conn.execute(
                    """
                    UPDATE thesis.quiz_answer_dialogs
                    SET "updatedAt" = now()
                    WHERE id = $1
                    """,
                    dialog_id,
                )

        result = self._message_from_row(row)
        logger.info(
            "Postgres quiz answer dialog assistant message inserted dialogId=%s messageId=%s sequenceNo=%d content_len=%d",
            dialog_id,
            result.message_id,
            result.sequence_no,
            len(content),
        )
        return result

    async def save_quiz_metadata(
            self,
            quiz_id: UUID,
            theme_id: Optional[int],
            name: str,
            description: str,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres saving quiz metadata quizId=%s themeId=%s name=%s",
            quiz_id,
            theme_id,
            name,
        )
        await pool.execute(
            """
            INSERT INTO thesis.quizes(id, type, name, description, "themeId")
            VALUES ($1, 'generated', $2, $3, $4) ON CONFLICT (id) DO
            UPDATE
                SET name = EXCLUDED.name,
                description = EXCLUDED.description,
                "themeId" = EXCLUDED."themeId"
            """,
            quiz_id,
            name,
            description,
            theme_id,
        )
        logger.info("Postgres quiz metadata saved quizId=%s", quiz_id)

    async def insert_question(
            self,
            question_id: UUID,
            qtype: str,
            text: str,
            multi_answer: bool,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres inserting question questionId=%s type=%s multiAnswer=%s",
            question_id,
            qtype,
            multi_answer,
        )
        await pool.execute(
            """
            INSERT INTO thesis.questions(id, type, text, "multiAnswer")
            VALUES ($1, $2, $3, $4)
            """,
            question_id,
            qtype,
            text,
            multi_answer,
        )
        logger.info("Postgres question inserted questionId=%s", question_id)

    async def insert_variant(
            self,
            variant_id: UUID,
            text: str,
            explain_right: str,
            explain_wrong: str,
            left_matching: Optional[str] = None,
            right_matching: Optional[str] = None,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres inserting variant variantId=%s hasMatching=%s text_len=%d",
            variant_id,
            left_matching is not None or right_matching is not None,
            len(text),
        )
        await pool.execute(
            """
            INSERT INTO thesis.variants(id, text, "explainRight", "explainWrong", "leftMatching", "rightMatching")
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            variant_id,
            text,
            explain_right,
            explain_wrong,
            left_matching,
            right_matching,
        )
        logger.info("Postgres variant inserted variantId=%s", variant_id)

    async def insert_question_variant_link(
            self,
            link_id: UUID,
            question_id: UUID,
            variant_id: Optional[UUID],
            is_right: bool,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres linking question variant linkId=%s questionId=%s variantId=%s isRight=%s",
            link_id,
            question_id,
            variant_id,
            is_right,
        )
        await pool.execute(
            """
            INSERT INTO thesis.questions_variants(id, "questionId", "variantId", "isRight")
            VALUES ($1, $2, $3, $4)
            """,
            link_id,
            question_id,
            variant_id,
            is_right,
        )
        logger.info("Postgres question variant linked linkId=%s", link_id)

    async def link_question_to_quiz(
            self,
            link_id: UUID,
            quiz_id: UUID,
            question_id: UUID,
    ) -> None:
        pool = await self.connect()
        logger.info("Postgres linking quiz question linkId=%s quizId=%s questionId=%s", link_id, quiz_id, question_id)
        await pool.execute(
            """
            INSERT INTO thesis.quizes_questions(id, "quizId", "questionId")
            VALUES ($1, $2, $3)
            """,
            link_id,
            quiz_id,
            question_id,
        )
        logger.info("Postgres quiz question linked linkId=%s", link_id)

    async def get_theme_name(self, theme_id: int) -> Optional[str]:
        pool = await self.connect()
        logger.info("Postgres loading theme name themeId=%s", theme_id)
        row = await pool.fetchrow(
            """
            SELECT name
            FROM thesis.themes
            WHERE id = $1
            """,
            theme_id,
        )
        theme_name = row["name"] if row else None
        logger.info("Postgres theme name loaded themeId=%s found=%s", theme_id, theme_name is not None)
        return theme_name

    async def get_quiz_answer_dialog_context(
            self,
            *,
            dialog_id: UUID,
            user_id: UUID,
            message_id: UUID,
    ) -> QuizAnswerDialogContext | None:
        pool = await self.connect()
        logger.info(
            "Postgres loading quiz answer dialog context dialogId=%s userId=%s messageId=%s",
            dialog_id,
            user_id,
            message_id,
        )
        dialog_row = await pool.fetchrow(
            """
            SELECT
                d.id,
                d."questionSubmissionId",
                d."userId",
                d."sessionId",
                d."quizId",
                d."questionId",
                d."contextSnapshot",
                q.type AS "questionType",
                q.text AS "questionText",
                q."multiAnswer" AS "questionMultiAnswer",
                qs."isRight" AS "submissionIsRight",
                qu."summaryId" AS "quizSummaryId",
                sf.id AS "summaryFileId",
                sf.name AS "summaryFileName",
                sf."s3Index" AS "summaryS3Index"
            FROM thesis.quiz_answer_dialogs d
            JOIN thesis.questions q ON q.id = d."questionId"
            LEFT JOIN thesis.question_submissions qs ON qs.id = d."questionSubmissionId"
            LEFT JOIN thesis.quizes qu ON qu.id = d."quizId"
            LEFT JOIN thesis.summaries s ON s.id = qu."summaryId"
            LEFT JOIN thesis.files sf ON sf.id = s."fileId"
            WHERE d.id = $1 AND d."userId" = $2
            """,
            dialog_id,
            user_id,
        )
        if dialog_row is None:
            logger.warning("Postgres quiz answer dialog not found dialogId=%s userId=%s", dialog_id, user_id)
            return None

        current_row = await pool.fetchrow(
            """
            SELECT id, "dialogId", "userId", role, content, "sequenceNo", metadata
            FROM thesis.quiz_answer_dialog_messages
            WHERE id = $1 AND "dialogId" = $2 AND "userId" = $3
            """,
            message_id,
            dialog_id,
            user_id,
        )
        if current_row is None:
            logger.warning(
                "Postgres quiz answer dialog current message not found dialogId=%s userId=%s messageId=%s",
                dialog_id,
                user_id,
                message_id,
            )
            return None

        message_rows = await pool.fetch(
            """
            SELECT id, "dialogId", "userId", role, content, "sequenceNo", metadata
            FROM thesis.quiz_answer_dialog_messages
            WHERE "dialogId" = $1
            ORDER BY "sequenceNo" ASC
            """,
            dialog_id,
        )
        messages = [self._message_from_row(row) for row in message_rows]

        variant_rows = await pool.fetch(
            """
            SELECT
                qv.id AS "linkId",
                qv."variantId",
                qv."isRight",
                v.text,
                v."explainRight",
                v."explainWrong",
                v."leftMatching",
                v."rightMatching"
            FROM thesis.questions_variants qv
            LEFT JOIN thesis.variants v ON v.id = qv."variantId"
            WHERE qv."questionId" = $1
            ORDER BY v.text NULLS LAST, qv.id
            """,
            dialog_row["questionId"],
        )
        variants = [
            QuizAnswerDialogVariant(
                link_id=row["linkId"],
                variant_id=row["variantId"],
                text=row["text"],
                is_right=row["isRight"],
                explain_right=row["explainRight"],
                explain_wrong=row["explainWrong"],
                left_matching=row["leftMatching"],
                right_matching=row["rightMatching"],
            )
            for row in variant_rows
        ]

        chosen_rows = await pool.fetch(
            """
            SELECT
                cv.id AS "submitId",
                cv."chosenId",
                cv.answer,
                cv."answerLeft",
                cv."answerRight",
                cv."isRight",
                cv.explanation,
                qv."isRight" AS "variantIsRight",
                v.text AS "variantText",
                v."explainRight",
                v."explainWrong",
                v."leftMatching",
                v."rightMatching"
            FROM thesis.session_submits ss
            JOIN thesis.chosen_variants cv ON cv.id = ss."submitId"
            LEFT JOIN thesis.questions_variants qv ON qv.id = cv."chosenId"
            LEFT JOIN thesis.variants v ON v.id = qv."variantId"
            WHERE ss."sessionId" = $1
              AND cv."quizId" = $2
              AND cv."questionId" = $3
            ORDER BY cv.id
            """,
            dialog_row["sessionId"],
            dialog_row["quizId"],
            dialog_row["questionId"],
        )
        if not chosen_rows:
            logger.warning(
                "Postgres quiz answer dialog has no chosen answers scoped to session dialogId=%s sessionId=%s quizId=%s questionId=%s",
                dialog_id,
                dialog_row["sessionId"],
                dialog_row["quizId"],
                dialog_row["questionId"],
            )
        chosen_answers = [
            QuizAnswerDialogChosenAnswer(
                submit_id=row["submitId"],
                chosen_id=row["chosenId"],
                answer=row["answer"],
                answer_left=row["answerLeft"],
                answer_right=row["answerRight"],
                is_right=row["isRight"],
                explanation=row["explanation"],
                variant_text=row["variantText"],
                variant_is_right=row["variantIsRight"],
                explain_right=row["explainRight"],
                explain_wrong=row["explainWrong"],
                left_matching=row["leftMatching"],
                right_matching=row["rightMatching"],
            )
            for row in chosen_rows
        ]

        reference_rows = await pool.fetch(
            """
            SELECT "fileId"
            FROM thesis.references_question
            WHERE "questionId" = $1
            ORDER BY id
            """,
            dialog_row["questionId"],
        )
        if not reference_rows:
            reference_rows = await pool.fetch(
                """
                SELECT "fileId"
                FROM thesis.references_quiz
                WHERE "quizId" = $1
                ORDER BY id
                """,
                dialog_row["quizId"],
            )
        reference_file_ids = [row["fileId"] for row in reference_rows]

        summary_file_record = None
        if dialog_row["summaryFileId"] and dialog_row["summaryS3Index"]:
            summary_file_record = FileRecord(
                file_id=dialog_row["summaryFileId"],
                name=dialog_row["summaryFileName"] or "summary.pdf",
                s3_index=dialog_row["summaryS3Index"],
            )

        context = QuizAnswerDialogContext(
            dialog_id=dialog_row["id"],
            user_id=dialog_row["userId"],
            question_submission_id=dialog_row["questionSubmissionId"],
            session_id=dialog_row["sessionId"],
            quiz_id=dialog_row["quizId"],
            question_id=dialog_row["questionId"],
            context_snapshot=self._json_dict(dialog_row["contextSnapshot"]),
            question_type=dialog_row["questionType"],
            question_text=dialog_row["questionText"],
            question_multi_answer=dialog_row["questionMultiAnswer"],
            submission_is_right=dialog_row["submissionIsRight"],
            quiz_summary_id=dialog_row["quizSummaryId"],
            summary_file_record=summary_file_record,
            current_message=self._message_from_row(current_row),
            messages=messages,
            variants=variants,
            chosen_answers=chosen_answers,
            reference_file_ids=reference_file_ids,
        )
        logger.info(
            "Postgres quiz answer dialog context loaded dialogId=%s messages=%d variants=%d chosen=%d references=%d has_summary_file=%s",
            dialog_id,
            len(messages),
            len(variants),
            len(chosen_answers),
            len(reference_file_ids),
            summary_file_record is not None,
        )
        return context

    async def insert_quiz_answer_dialog_assistant_message(
            self,
            *,
            dialog_id: UUID,
            user_id: UUID,
            content: str,
            metadata: dict[str, Any] | None = None,
    ) -> QuizAnswerDialogMessage:
        pool = await self.connect()
        message_id = uuid4()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        async with pool.acquire() as conn:
            async with conn.transaction():
                sequence_no = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX("sequenceNo"), 0) + 1
                    FROM thesis.quiz_answer_dialog_messages
                    WHERE "dialogId" = $1
                    """,
                    dialog_id,
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO thesis.quiz_answer_dialog_messages(
                        id, "dialogId", "userId", role, content, "sequenceNo", metadata
                    )
                    VALUES ($1, $2, $3, 'assistant', $4, $5, $6::jsonb)
                    RETURNING id, "dialogId", "userId", role, content, "sequenceNo", metadata
                    """,
                    message_id,
                    dialog_id,
                    user_id,
                    content,
                    sequence_no,
                    metadata_json,
                )
                await conn.execute(
                    """
                    UPDATE thesis.quiz_answer_dialogs
                    SET "updatedAt" = now()
                    WHERE id = $1
                    """,
                    dialog_id,
                )

        result = self._message_from_row(row)
        logger.info(
            "Postgres quiz answer dialog assistant message inserted dialogId=%s messageId=%s sequenceNo=%d content_len=%d",
            dialog_id,
            result.message_id,
            result.sequence_no,
            len(content),
        )
        return result

    async def insert_reference_question(
            self,
            ref_id: UUID,
            question_id: UUID,
            file_id: UUID,
    ) -> None:
        pool = await self.connect()
        logger.info(
            "Postgres inserting question reference refId=%s questionId=%s fileId=%s",
            ref_id,
            question_id,
            file_id,
        )
        await pool.execute(
            """
            INSERT INTO thesis.references_question(id, "questionId", "fileId")
            VALUES ($1, $2, $3)
            """,
            ref_id,
            question_id,
            file_id,
        )
        logger.info("Postgres question reference inserted refId=%s", ref_id)

    async def insert_faq_item(
            self,
            faq_id: UUID,
            question: str,
            answer: str,
            category: Optional[str],
    ) -> None:
        pool = await self.connect()
        logger.info("Postgres inserting FAQ item faqId=%s category=%s", faq_id, category)
        await pool.execute(
            """
            INSERT INTO thesis.questions(id, type, text)
            VALUES ($1, 'faq', $2)
            """,
            faq_id,
            question,
        )
        logger.info("Postgres FAQ item inserted faqId=%s answer_len=%d", faq_id, len(answer))

    async def save_summary_text(
            self,
            summary_id: UUID,
            theme_id: int,
            s3_key: str,
            file_name: str,
            user_id: Optional[UUID] = None,
    ) -> int:
        pool = await self.connect()
        logger.info(
            "Postgres saving summary metadata fileId=%s themeId=%s s3_key=%s userId=%s",
            summary_id,
            theme_id,
            s3_key,
            user_id,
        )

        await pool.execute(
            """
            INSERT INTO thesis.files(id, name, "s3Index", "userId")
            VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO
            UPDATE
                SET name = EXCLUDED.name,
                "s3Index" = EXCLUDED."s3Index",
                "userId" = EXCLUDED."userId"
            """,
            summary_id,
            file_name,
            s3_key,
            user_id,
        )
        logger.info("Postgres summary file metadata saved fileId=%s", summary_id)

        row = await pool.fetchrow(
            """
            SELECT "subjectId"
            FROM thesis.themes
            WHERE id = $1
            """,
            theme_id,
        )
        subject_id = row["subjectId"] if row else None
        logger.info("Postgres summary subject resolved fileId=%s subjectId=%s", summary_id, subject_id)

        summary_row = await pool.fetchrow(
            """
            INSERT INTO thesis.summaries("subjectId", "themeId", "fileId", name)
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            subject_id,
            theme_id,
            summary_id,
            file_name,
        )
        summary_db_id: int = summary_row["id"]
        logger.info("Postgres summary row inserted summaryDbId=%s fileId=%s", summary_db_id, summary_id)

        await pool.execute(
            """
            INSERT INTO thesis.references_summary("summaryId", "fileId")
            VALUES ($1, $2)
            """,
            summary_db_id,
            summary_id,
        )
        logger.info("Postgres summary reference inserted summaryDbId=%s fileId=%s", summary_db_id, summary_id)

        return summary_db_id


def get_postgres_client() -> PostgresClient:
    return PostgresClient()
