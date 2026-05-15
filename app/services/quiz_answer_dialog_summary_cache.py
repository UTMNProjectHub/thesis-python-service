from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.api.core.config import settings
from app.services.postgres import QuizAnswerDialogMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogSummaryChunk:
    cache_key: str
    sequence_start: int
    sequence_end: int
    message_count: int
    summary: str
    cache_hit: bool


class QuizAnswerDialogSummaryCache:
    def __init__(
            self,
            db_path: str | Path | None = None,
            *,
            busy_timeout_ms: int | None = None,
    ) -> None:
        self.db_path = Path(db_path or settings.quiz_answer_dialog_summary_cache_db_path)
        self.cache_dir = self.db_path.parent
        self.busy_timeout_ms = (
            settings.quiz_answer_dialog_summary_cache_busy_timeout_ms
            if busy_timeout_ms is None
            else busy_timeout_ms
        )
        self._prepare_database()

    @staticmethod
    def build_cache_key(
            *,
            dialog_id: str,
            messages: Iterable[QuizAnswerDialogMessage],
            model: str,
    ) -> tuple[str, str, int, int, int]:
        message_list = list(messages)
        payload = [
            {
                "id": str(message.message_id),
                "role": message.role,
                "content": message.content,
                "sequenceNo": message.sequence_no,
            }
            for message in message_list
        ]
        content_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        sequence_start = message_list[0].sequence_no if message_list else 0
        sequence_end = message_list[-1].sequence_no if message_list else 0
        raw_key = "\0".join([dialog_id, str(sequence_start), str(sequence_end), content_hash, model])
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return cache_key, content_hash, sequence_start, sequence_end, len(message_list)

    def get(self, cache_key: str) -> DialogSummaryChunk | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cache_key, sequence_start, sequence_end, message_count, summary
                FROM dialog_summary_chunks
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            summary = str(row["summary"]).strip()
            if not summary:
                raise ValueError("empty summary")
            return DialogSummaryChunk(
                cache_key=str(row["cache_key"]),
                sequence_start=int(row["sequence_start"]),
                sequence_end=int(row["sequence_end"]),
                message_count=int(row["message_count"]),
                summary=summary,
                cache_hit=True,
            )
        except Exception as exc:
            logger.warning("Invalid quiz answer dialog summary cache entry cacheKey=%s error=%s", cache_key, exc)
            self.delete(cache_key)
            return None

    def set(
            self,
            *,
            cache_key: str,
            dialog_id: str,
            content_hash: str,
            sequence_start: int,
            sequence_end: int,
            message_count: int,
            model: str,
            summary: str,
    ) -> DialogSummaryChunk:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO dialog_summary_chunks(
                    cache_key,
                    dialog_id,
                    content_hash,
                    sequence_start,
                    sequence_end,
                    message_count,
                    model,
                    summary,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM dialog_summary_chunks WHERE cache_key = ?),
                    ?
                ), ?)
                """,
                (
                    cache_key,
                    dialog_id,
                    content_hash,
                    int(sequence_start),
                    int(sequence_end),
                    int(message_count),
                    model,
                    summary,
                    cache_key,
                    now,
                    now,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        logger.info(
            "Quiz answer dialog summary cache stored dialogId=%s cacheKey=%s sequence=%d-%d messages=%d",
            dialog_id,
            cache_key,
            sequence_start,
            sequence_end,
            message_count,
        )
        return DialogSummaryChunk(
            cache_key=cache_key,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            message_count=message_count,
            summary=summary,
            cache_hit=False,
        )

    def delete(self, cache_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM dialog_summary_chunks WHERE cache_key = ?", (cache_key,))
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=max(self.busy_timeout_ms / 1000.0, 1.0))
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _prepare_database(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dialog_summary_chunks (
                    cache_key TEXT PRIMARY KEY,
                    dialog_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    sequence_start INTEGER NOT NULL,
                    sequence_end INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dialog_summary_chunks_dialog
                ON dialog_summary_chunks(dialog_id, sequence_start, sequence_end)
                """
            )
            conn.commit()
