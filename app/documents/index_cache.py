from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from app.api.core.config import settings
from app.documents.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

DOCUMENT_INDEX_CACHE_SCHEMA_VERSION = 1
DOCUMENT_CHUNKING_VERSION = "chunk_document_pages:v1"
EMBEDDING_SPLIT_VERSION = "embeddings_prepare_chunks:v1"


class _FileLock:
    def __init__(self, path: Path, timeout_seconds: float, stale_seconds: float | None = None) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self._fd: int | None = None

    def _remove_if_stale(self) -> bool:
        if self.stale_seconds is None or self.stale_seconds <= 0 or not self.path.exists():
            return False
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False
        if age < self.stale_seconds:
            return False
        try:
            self.path.unlink()
            logger.warning("Removed stale document index cache lock path=%s age_seconds=%.1f", self.path, age)
            return True
        except OSError as e:
            logger.warning("Could not remove stale document index cache lock path=%s error=%s", self.path, e)
            return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(self.timeout_seconds, 0.0)
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n"
                os.write(self._fd, payload.encode("utf-8"))
                return
            except FileExistsError:
                if self._remove_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for document index cache lock: {self.path}")
                time.sleep(0.2)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Could not remove document index cache lock path=%s error=%s", self.path, e)


@dataclass
class DocumentIndexData:
    document: Document
    chunks: list[DocumentChunk]
    embedding_chunks: list[DocumentChunk]
    embeddings: np.ndarray


@dataclass
class CachedDocumentIndex(DocumentIndexData):
    cache_key: str
    file_hash: str
    cache_hit: bool


class DocumentIndexCache:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        lock_timeout_seconds: float | None = None,
        stale_lock_seconds: float | None = None,
        busy_timeout_ms: int | None = None,
    ) -> None:
        self.db_path = Path(db_path or settings.document_index_cache_db_path)
        self.cache_dir = self.db_path.parent
        self.locks_dir = self.cache_dir / "locks"
        self.lock_timeout_seconds = (
            settings.document_index_cache_lock_timeout_seconds
            if lock_timeout_seconds is None
            else lock_timeout_seconds
        )
        self.stale_lock_seconds = (
            settings.document_index_cache_stale_lock_seconds
            if stale_lock_seconds is None
            else stale_lock_seconds
        )
        self.busy_timeout_ms = (
            settings.document_index_cache_busy_timeout_ms
            if busy_timeout_ms is None
            else busy_timeout_ms
        )
        self._prepare_database()
        self._cleanup_stale_locks()

    @staticmethod
    def compute_file_hash(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def build_cache_key(
        *,
        file_hash: str,
        file_id: str,
        s3_index: str,
        chunk_tokens: int,
        embedding_model: str,
        schema_version: int = DOCUMENT_INDEX_CACHE_SCHEMA_VERSION,
        chunking_version: str = DOCUMENT_CHUNKING_VERSION,
        embedding_split_version: str = EMBEDDING_SPLIT_VERSION,
    ) -> str:
        payload = "\0".join(
            [
                file_hash,
                str(file_id),
                s3_index,
                str(chunk_tokens),
                embedding_model,
                str(schema_version),
                chunking_version,
                embedding_split_version,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_or_build(
        self,
        *,
        file_path: str | Path,
        file_id: str,
        file_name: str | None,
        s3_index: str,
        chunk_tokens: int,
        embedding_model: str,
        builder: Callable[[], DocumentIndexData],
    ) -> CachedDocumentIndex:
        pdf_path = Path(file_path)
        file_hash = self.compute_file_hash(pdf_path)
        cache_key = self.build_cache_key(
            file_hash=file_hash,
            file_id=file_id,
            s3_index=s3_index,
            chunk_tokens=chunk_tokens,
            embedding_model=embedding_model,
        )

        cached = self._load_entry(cache_key, pdf_path)
        if cached is not None:
            logger.info(
                "Document index cache hit fileId=%s s3Index=%s chunks=%d embedding_chunks=%d",
                file_id,
                s3_index,
                len(cached.chunks),
                len(cached.embedding_chunks),
            )
            return cached

        logger.info("Document index cache miss fileId=%s s3Index=%s", file_id, s3_index)
        lock = _FileLock(
            self._entry_lock_path(cache_key),
            timeout_seconds=self.lock_timeout_seconds,
            stale_seconds=self.stale_lock_seconds,
        )
        lock.acquire()
        try:
            cached = self._load_entry(cache_key, pdf_path)
            if cached is not None:
                logger.info(
                    "Document index cache hit after wait fileId=%s s3Index=%s chunks=%d embedding_chunks=%d",
                    file_id,
                    s3_index,
                    len(cached.chunks),
                    len(cached.embedding_chunks),
                )
                return cached

            started = time.perf_counter()
            data = builder()
            self._save_entry(
                cache_key=cache_key,
                file_hash=file_hash,
                file_id=file_id,
                s3_index=s3_index,
                chunk_tokens=chunk_tokens,
                embedding_model=embedding_model,
                data=data,
            )
            logger.info(
                "Document index cache stored fileId=%s s3Index=%s chunks=%d embedding_chunks=%d seconds=%.3f",
                file_id,
                s3_index,
                len(data.chunks),
                len(data.embedding_chunks),
                time.perf_counter() - started,
            )
            return CachedDocumentIndex(
                document=data.document,
                chunks=data.chunks,
                embedding_chunks=data.embedding_chunks,
                embeddings=np.asarray(data.embeddings, dtype="float32"),
                cache_key=cache_key,
                file_hash=file_hash,
                cache_hit=False,
            )
        finally:
            lock.release()

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
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    s3_index TEXT NOT NULL,
                    chunk_tokens INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    chunking_version TEXT NOT NULL,
                    embedding_split_version TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    embedding_chunks_json TEXT NOT NULL,
                    embeddings_blob BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_entries_file_hash ON cache_entries(file_hash)")
            conn.commit()

    def _cleanup_stale_locks(self) -> None:
        if not self.locks_dir.exists() or self.stale_lock_seconds <= 0:
            return
        now = time.time()
        for path in self.locks_dir.glob("*.lock"):
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age < self.stale_lock_seconds:
                continue
            try:
                path.unlink()
                logger.warning("Removed stale document index cache lock path=%s age_seconds=%.1f", path, age)
            except OSError as e:
                logger.warning("Could not remove stale document index cache lock path=%s error=%s", path, e)

    def _entry_lock_path(self, cache_key: str) -> Path:
        return self.locks_dir / f"{cache_key}.lock"

    def _load_entry(self, cache_key: str, file_path: Path) -> CachedDocumentIndex | None:
        started = time.perf_counter()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)).fetchone()
        if row is None:
            return None

        try:
            document = self._document_from_json(row["document_json"], file_path)
            chunks = [self._chunk_from_json(item) for item in json.loads(row["chunks_json"])]
            embedding_chunks = [
                self._chunk_from_json(item) for item in json.loads(row["embedding_chunks_json"])
            ]
            embedding_dim = int(row["embedding_dim"])
            embedding_count = int(row["embedding_count"])
            if embedding_count > 0 and embedding_dim <= 0:
                raise ValueError(f"Invalid embedding_dim={embedding_dim}")
            if embedding_count != len(embedding_chunks):
                raise ValueError(
                    f"Embedding count mismatch in cache: stored={embedding_count} chunks={len(embedding_chunks)}"
                )
            embeddings = np.frombuffer(row["embeddings_blob"], dtype=np.float32)
            expected_size = embedding_count * embedding_dim
            if embeddings.size != expected_size:
                raise ValueError(f"Embedding blob size mismatch: size={embeddings.size} expected={expected_size}")
            embeddings = embeddings.reshape((embedding_count, embedding_dim)).copy()
            logger.info(
                "Document index cache loaded cacheKey=%s chunks=%d embedding_chunks=%d seconds=%.3f",
                cache_key,
                len(chunks),
                len(embedding_chunks),
                time.perf_counter() - started,
            )
            return CachedDocumentIndex(
                document=document,
                chunks=chunks,
                embedding_chunks=embedding_chunks,
                embeddings=embeddings,
                cache_key=cache_key,
                file_hash=str(row["file_hash"]),
                cache_hit=True,
            )
        except Exception as e:
            logger.warning("Document index cache entry is invalid cacheKey=%s error=%s", cache_key, e, exc_info=True)
            self._delete_entry(cache_key)
            return None

    def _save_entry(
        self,
        *,
        cache_key: str,
        file_hash: str,
        file_id: str,
        s3_index: str,
        chunk_tokens: int,
        embedding_model: str,
        data: DocumentIndexData,
    ) -> None:
        embeddings = np.asarray(data.embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}")
        if embeddings.shape[0] != len(data.embedding_chunks):
            raise ValueError(
                f"Embeddings count mismatch: chunks={len(data.embedding_chunks)} embeddings={embeddings.shape[0]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        document_json = json.dumps(self._document_to_json(data.document), ensure_ascii=False)
        chunks_json = json.dumps([self._chunk_to_json(chunk) for chunk in data.chunks], ensure_ascii=False)
        embedding_chunks_json = json.dumps(
            [self._chunk_to_json(chunk) for chunk in data.embedding_chunks],
            ensure_ascii=False,
        )
        blob = np.ascontiguousarray(embeddings, dtype=np.float32).tobytes()

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (
                    cache_key,
                    file_hash,
                    file_id,
                    s3_index,
                    chunk_tokens,
                    embedding_model,
                    schema_version,
                    chunking_version,
                    embedding_split_version,
                    document_json,
                    chunks_json,
                    embedding_chunks_json,
                    embeddings_blob,
                    embedding_dim,
                    embedding_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM cache_entries WHERE cache_key = ?),
                    ?
                ), ?)
                """,
                (
                    cache_key,
                    file_hash,
                    file_id,
                    s3_index,
                    int(chunk_tokens),
                    embedding_model,
                    DOCUMENT_INDEX_CACHE_SCHEMA_VERSION,
                    DOCUMENT_CHUNKING_VERSION,
                    EMBEDDING_SPLIT_VERSION,
                    document_json,
                    chunks_json,
                    embedding_chunks_json,
                    sqlite3.Binary(blob),
                    int(embeddings.shape[1]),
                    int(embeddings.shape[0]),
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

    def _delete_entry(self, cache_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
            conn.commit()

    @staticmethod
    def _document_to_json(document: Document) -> dict:
        return {
            "id": document.id,
            "path": str(document.path),
            "title": document.title,
            "pages": document.pages,
        }

    @staticmethod
    def _document_from_json(raw: str, file_path: Path) -> Document:
        data = json.loads(raw)
        return Document(
            id=str(data["id"]),
            path=file_path,
            title=data.get("title"),
            pages=int(data.get("pages") or 0),
        )

    @staticmethod
    def _chunk_to_json(chunk: DocumentChunk) -> dict:
        return {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "heading_path": chunk.heading_path,
        }

    @staticmethod
    def _chunk_from_json(data: dict) -> DocumentChunk:
        return DocumentChunk(
            doc_id=str(data["doc_id"]),
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            heading_path=data.get("heading_path"),
        )
