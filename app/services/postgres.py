# app/services/postgres.py
from __future__ import annotations

from typing import List, Optional
from uuid import UUID
import json

import asyncpg

from app.api.core.config import settings


class PostgresClient:
    """
    Асинхронный клиент PostgreSQL.
    Отвечает за:
    - получение s3Index для файлов;
    - сохранение summary (пока в таблицу quizes как type='summary');
    - сохранение квиза, вопросов, вариантов, FAQ.
    """

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> asyncpg.Pool:
        if self._pool:
            return self._pool
        self._pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
        )
        return self._pool

    # ------------------------------------------------------------------
    # FILES → s3Index
    # ------------------------------------------------------------------
    async def get_s3_keys_for_file_ids(self, file_ids: List[UUID]) -> List[str]:
        """
        SELECT "s3Index" FROM thesis.files WHERE id IN (...)
        """
        pool = await self.connect()
        rows = await pool.fetch(
            """
            SELECT "s3Index"
            FROM thesis.files
            WHERE id = ANY($1::uuid[])
            """,
            file_ids,
        )
        return [r["s3Index"] for r in rows]

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    async def save_summary_text(
            self,
            summary_id: UUID,
            theme_id: int,
            markdown: str,
    ) -> None:
        """
        Простая реализация: используем таблицу quizes как хранилище summary.
        type = 'summary', name = 'Авто-конспект'.
        """
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.quizes(id, type, name, description, "themeId")
            VALUES ($1, 'summary', 'Авто-конспект', $2, $3)
            ON CONFLICT (id) DO UPDATE
            SET description = EXCLUDED.description,
                "themeId"   = EXCLUDED."themeId"
            """,
            summary_id,
            markdown,
            theme_id,
        )

    # ------------------------------------------------------------------
    # QUIZ
    # ------------------------------------------------------------------
    async def save_quiz_metadata(
            self,
            quiz_id: UUID,
            theme_id: Optional[int],
            name: str,
            description: str,
    ) -> None:
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.quizes(id, type, name, description, "themeId")
            VALUES ($1, 'generated', $2, $3, $4)
            ON CONFLICT (id) DO UPDATE
            SET name = EXCLUDED.name,
                description = EXCLUDED.description,
                "themeId"   = EXCLUDED."themeId"
            """,
            quiz_id,
            name,
            description,
            theme_id,
        )

    async def insert_question(
            self,
            question_id: UUID,
            qtype: str,
            text: str,
            multi_answer: bool,
    ) -> None:
        pool = await self.connect()
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

    async def insert_variant(
            self,
            variant_id: UUID,
            text: str,
            explain_right: str,
            explain_wrong: str,
    ) -> None:
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.variants(id, text, "explainRight", "explainWrong")
            VALUES ($1, $2, $3, $4)
            """,
            variant_id,
            text,
            explain_right,
            explain_wrong,
        )

    async def insert_question_variant_link(
            self,
            link_id: UUID,
            question_id: UUID,
            variant_id: Optional[UUID],
            is_right: bool,
            matching_config: Optional[dict],
    ) -> None:
        pool = await self.connect()

        if matching_config is not None and not isinstance(matching_config, str):
            matching_config_db = json.dumps(matching_config, ensure_ascii=False)
        else:
            matching_config_db = matching_config

        await pool.execute(
            """
            INSERT INTO thesis.questions_variants(
                id, "questionId", "variantId", "isRight", "matchingConfig"
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            link_id,
            question_id,
            variant_id,
            is_right,
            matching_config_db,
        )

    async def link_question_to_quiz(
            self,
            link_id: UUID,
            quiz_id: UUID,
            question_id: UUID,
    ) -> None:
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.quizes_questions(id, "quizId", "questionId")
            VALUES ($1, $2, $3)
            """,
            link_id,
            quiz_id,
            question_id,
        )

    async def get_theme_name(self, theme_id: int) -> Optional[str]:
        """
        Возвращает name из thesis.themes по id.
        """
        pool = await self.connect()
        row = await pool.fetchrow(
            """
            SELECT name
            FROM thesis.themes
            WHERE id = $1
            """,
            theme_id,
        )
        return row["name"] if row else None

    async def insert_reference_question(
            self,
            ref_id: UUID,
            question_id: UUID,
            file_id: UUID,
    ) -> None:
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.references_question(id, "questionId", "fileId")
            VALUES ($1, $2, $3)
            """,
            ref_id,
            question_id,
            file_id,
        )

    # ------------------------------------------------------------------
    # FAQ (минимальная реализация)
    # ------------------------------------------------------------------
    async def insert_faq_item(
            self,
            faq_id: UUID,
            question: str,
            answer: str,
            category: Optional[str],
    ) -> None:
        """
        Пока складываем FAQ в таблицу questions с типом 'faq'.
        При необходимости можно вынести в отдельную таблицу.
        """
        pool = await self.connect()
        await pool.execute(
            """
            INSERT INTO thesis.questions(id, type, text)
            VALUES ($1, 'faq', $2)
            """,
            faq_id,
            question,
        )
        # answer/category можно хранить в отдельной таблице — сейчас опускаем

        # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    async def save_summary_text(
            self,
            summary_id: UUID,          # это будет id в thesis.files.id
            theme_id: int,
            s3_key: str,               # ключ в S3 (s3Index)
            file_name: str,            # читаемое имя файла
            user_id: Optional[UUID] = None,
    ) -> int:
        """
        Сохранение метаданных конспекта.

        Делает три вещи:

        1) thesis.files:
           - id        = summary_id (uuid)
           - name      = file_name
           - s3Index   = s3_key
           - userId    = user_id

        2) thesis.summaries:
           - subjectId = берём из thesis.themes.subjectId (если есть)
           - themeId   = theme_id
           - fileId    = summary_id

        3) thesis.references_summary:
           - summaryId = id из thesis.summaries
           - fileId    = summary_id

        Возвращает integer id из thesis.summaries.
        """
        pool = await self.connect()

        # 1. создаём/обновляем запись в thesis.files
        await pool.execute(
            """
            INSERT INTO thesis.files(id, name, "s3Index", "userId")
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE
            SET name    = EXCLUDED.name,
                "s3Index" = EXCLUDED."s3Index",
                "userId"  = EXCLUDED."userId"
            """,
            summary_id,
            file_name,
            s3_key,
            user_id,
        )

        # 2. вытаскиваем subjectId из thesis.themes (он nullable)
        row = await pool.fetchrow(
            """
            SELECT "subjectId"
            FROM thesis.themes
            WHERE id = $1
            """,
            theme_id,
        )
        subject_id = row["subjectId"] if row else None

        # 3. создаём запись в thesis.summaries
        summary_row = await pool.fetchrow(
            """
            INSERT INTO thesis.summaries("subjectId", "themeId", "fileId")
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            subject_id,
            theme_id,
            summary_id,
        )
        summary_db_id: int = summary_row["id"]

        # 4. создаём связь в thesis.references_summary
        await pool.execute(
            """
            INSERT INTO thesis.references_summary("summaryId", "fileId")
            VALUES ($1, $2)
            """,
            summary_db_id,
            summary_id,
        )

        return summary_db_id



def get_postgres_client() -> PostgresClient:
    return PostgresClient()
