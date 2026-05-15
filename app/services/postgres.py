from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional
from uuid import UUID

import asyncpg

from app.api.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileRecord:
    file_id: UUID
    name: str
    s3_index: str


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
            INSERT INTO thesis.summaries("subjectId", "themeId", "fileId")
            VALUES ($1, $2, $3) RETURNING id
            """,
            subject_id,
            theme_id,
            summary_id,
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
