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
from typing import Callable, Iterable

import numpy as np

from app.api.core.config import settings
from app.documents.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

DOCUMENT_INDEX_CACHE_SCHEMA_VERSION = 2
DOCUMENT_INDEX_CACHE_LEGACY_SCHEMA_VERSION = 1
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


@dataclass(frozen=True)
class DocumentIndexBuildRequest:
    file_path: Path
    file_id: str
    file_name: str | None
    s3_index: str
    chunk_tokens: int
    embedding_model: str
    builder: Callable[[], DocumentIndexData]


@dataclass(frozen=True)
class _ResolvedRequest:
    request: DocumentIndexBuildRequest
    file_hash: str
    cache_key: str
    legacy_cache_key: str


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={array.shape}")
    if array.shape[0] == 0:
        return array.copy()
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return np.ascontiguousarray(array / safe_norms, dtype=np.float32)


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
        request = DocumentIndexBuildRequest(
            file_path=Path(file_path),
            file_id=str(file_id),
            file_name=file_name,
            s3_index=s3_index,
            chunk_tokens=chunk_tokens,
            embedding_model=embedding_model,
            builder=builder,
        )
        return self.get_or_build_many([request])[0]

    def get_or_build_many(self, requests: Iterable[DocumentIndexBuildRequest]) -> list[CachedDocumentIndex]:
        resolved = [self._resolve_request(request) for request in requests]
        if not resolved:
            return []

        started = time.perf_counter()
        loaded = self._load_entries(resolved)
        results: dict[str, CachedDocumentIndex] = dict(loaded)
        logger.info(
            "Document index cache bulk load requested=%d hits=%d misses=%d seconds=%.3f",
            len(resolved),
            len(results),
            len(resolved) - len(results),
            time.perf_counter() - started,
        )

        for item in resolved:
            if item.cache_key in results:
                continue
            results[item.cache_key] = self._build_missing_entry(item)

        return [results[item.cache_key] for item in resolved]

    def _resolve_request(self, request: DocumentIndexBuildRequest) -> _ResolvedRequest:
        file_hash = self.compute_file_hash(request.file_path)
        cache_key = self.build_cache_key(
            file_hash=file_hash,
            file_id=request.file_id,
            s3_index=request.s3_index,
            chunk_tokens=request.chunk_tokens,
            embedding_model=request.embedding_model,
        )
        legacy_cache_key = self.build_cache_key(
            file_hash=file_hash,
            file_id=request.file_id,
            s3_index=request.s3_index,
            chunk_tokens=request.chunk_tokens,
            embedding_model=request.embedding_model,
            schema_version=DOCUMENT_INDEX_CACHE_LEGACY_SCHEMA_VERSION,
        )
        return _ResolvedRequest(
            request=request,
            file_hash=file_hash,
            cache_key=cache_key,
            legacy_cache_key=legacy_cache_key,
        )

    def _build_missing_entry(self, item: _ResolvedRequest) -> CachedDocumentIndex:
        request = item.request
        logger.info("Document index cache miss fileId=%s s3Index=%s", request.file_id, request.s3_index)
        lock = _FileLock(
            self._entry_lock_path(item.cache_key),
            timeout_seconds=self.lock_timeout_seconds,
            stale_seconds=self.stale_lock_seconds,
        )
        lock.acquire()
        try:
            cached = self._load_entries([item]).get(item.cache_key)
            if cached is not None:
                logger.info(
                    "Document index cache hit after wait fileId=%s s3Index=%s chunks=%d embedding_chunks=%d",
                    request.file_id,
                    request.s3_index,
                    len(cached.chunks),
                    len(cached.embedding_chunks),
                )
                return cached

            started = time.perf_counter()
            data = request.builder()
            self._save_entry(
                cache_key=item.cache_key,
                file_hash=item.file_hash,
                file_id=request.file_id,
                s3_index=request.s3_index,
                chunk_tokens=request.chunk_tokens,
                embedding_model=request.embedding_model,
                data=data,
            )
            embeddings = normalize_embeddings(np.asarray(data.embeddings, dtype=np.float32))
            logger.info(
                "Document index cache stored fileId=%s s3Index=%s chunks=%d embedding_chunks=%d seconds=%.3f",
                request.file_id,
                request.s3_index,
                len(data.chunks),
                len(data.embedding_chunks),
                time.perf_counter() - started,
            )
            return CachedDocumentIndex(
                document=data.document,
                chunks=data.chunks,
                embedding_chunks=data.embedding_chunks,
                embeddings=embeddings,
                cache_key=item.cache_key,
                file_hash=item.file_hash,
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
        conn.execute("PRAGMA temp_store=MEMORY")
        try:
            conn.execute("PRAGMA mmap_size=268435456")
        except sqlite3.DatabaseError:
            logger.debug("SQLite mmap_size pragma is not supported for document index cache")
        return conn

    def _prepare_database(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._prepare_legacy_table(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    cache_key TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    s3_index TEXT NOT NULL,
                    chunk_tokens INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    chunking_version TEXT NOT NULL,
                    embedding_split_version TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_count INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ready',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    cache_key TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    title TEXT,
                    pages INTEGER NOT NULL,
                    FOREIGN KEY(cache_key) REFERENCES cache_meta(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    cache_key TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    heading_path_json TEXT,
                    PRIMARY KEY(cache_key, chunk_order),
                    FOREIGN KEY(cache_key) REFERENCES cache_meta(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embedding_chunks (
                    cache_key TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    heading_path_json TEXT,
                    PRIMARY KEY(cache_key, chunk_order),
                    FOREIGN KEY(cache_key) REFERENCES cache_meta(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT NOT NULL,
                    embedding_order INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    PRIMARY KEY(cache_key, embedding_order),
                    FOREIGN KEY(cache_key) REFERENCES cache_meta(cache_key) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_cache_meta_file_hash ON cache_meta(file_hash);
                CREATE INDEX IF NOT EXISTS idx_cache_meta_file_id ON cache_meta(file_id);
                CREATE INDEX IF NOT EXISTS idx_cache_meta_s3_index ON cache_meta(s3_index);
                CREATE INDEX IF NOT EXISTS idx_chunks_cache_order ON chunks(cache_key, chunk_order);
                CREATE INDEX IF NOT EXISTS idx_embedding_chunks_cache_order ON embedding_chunks(cache_key, chunk_order);
                CREATE INDEX IF NOT EXISTS idx_embeddings_cache_order ON embeddings(cache_key, embedding_order);
                """
            )
            conn.commit()

    @staticmethod
    def _prepare_legacy_table(conn: sqlite3.Connection) -> None:
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

    def _load_entries(self, resolved: list[_ResolvedRequest]) -> dict[str, CachedDocumentIndex]:
        if not resolved:
            return {}

        started = time.perf_counter()
        path_by_key = {item.cache_key: item.request.file_path for item in resolved}
        legacy_to_current = {item.legacy_cache_key: item for item in resolved}
        keys = [item.cache_key for item in resolved]
        placeholders = ",".join("?" for _ in keys)
        result: dict[str, CachedDocumentIndex] = {}
        invalid_keys: set[str] = set()

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM cache_meta WHERE cache_key IN ({placeholders}) AND status = 'ready'",
                keys,
            ).fetchall()
            for row in rows:
                cache_key = str(row["cache_key"])
                try:
                    result[cache_key] = self._load_v2_entry_from_row(conn, row, path_by_key[cache_key])
                except Exception as e:
                    logger.warning("Document index cache v2 entry is invalid cacheKey=%s error=%s", cache_key, e)
                    invalid_keys.add(cache_key)

            missing = [item for item in resolved if item.cache_key not in result and item.cache_key not in invalid_keys]
            if missing:
                legacy_keys = [item.legacy_cache_key for item in missing]
                placeholders = ",".join("?" for _ in legacy_keys)
                legacy_rows = conn.execute(
                    f"SELECT * FROM cache_entries WHERE cache_key IN ({placeholders})",
                    legacy_keys,
                ).fetchall()
                for row in legacy_rows:
                    legacy_key = str(row["cache_key"])
                    item = legacy_to_current[legacy_key]
                    try:
                        legacy = self._load_v1_entry_from_row(row, item.request.file_path, item.cache_key)
                    except Exception as e:
                        logger.warning(
                            "Document index cache legacy entry is invalid cacheKey=%s error=%s",
                            legacy_key,
                            e,
                        )
                        invalid_keys.add(legacy_key)
                        continue
                    self._save_entry(
                        cache_key=item.cache_key,
                        file_hash=item.file_hash,
                        file_id=item.request.file_id,
                        s3_index=item.request.s3_index,
                        chunk_tokens=item.request.chunk_tokens,
                        embedding_model=item.request.embedding_model,
                        data=legacy,
                    )
                    result[item.cache_key] = CachedDocumentIndex(
                        document=legacy.document,
                        chunks=legacy.chunks,
                        embedding_chunks=legacy.embedding_chunks,
                        embeddings=normalize_embeddings(legacy.embeddings),
                        cache_key=item.cache_key,
                        file_hash=item.file_hash,
                        cache_hit=True,
                    )
                    logger.info(
                        "Document index cache migrated legacy entry legacyCacheKey=%s cacheKey=%s",
                        legacy_key,
                        item.cache_key,
                    )

        if invalid_keys:
            for cache_key in invalid_keys:
                self._delete_entry(cache_key)

        if result:
            chunks_count = sum(len(entry.chunks) for entry in result.values())
            embedding_count = sum(len(entry.embedding_chunks) for entry in result.values())
            embedding_bytes = sum(int(entry.embeddings.nbytes) for entry in result.values())
            logger.info(
                "Document index cache bulk loaded entries=%d chunks=%d embedding_chunks=%d embedding_bytes=%d seconds=%.3f",
                len(result),
                chunks_count,
                embedding_count,
                embedding_bytes,
                time.perf_counter() - started,
            )
        return result

    def _load_v2_entry_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        file_path: Path,
    ) -> CachedDocumentIndex:
        cache_key = str(row["cache_key"])
        doc_row = conn.execute("SELECT * FROM documents WHERE cache_key = ?", (cache_key,)).fetchone()
        if doc_row is None:
            raise ValueError("missing document row")

        chunks = [
            self._chunk_from_row(chunk_row)
            for chunk_row in conn.execute(
                "SELECT * FROM chunks WHERE cache_key = ? ORDER BY chunk_order",
                (cache_key,),
            )
        ]
        embedding_chunks = [
            self._chunk_from_row(chunk_row)
            for chunk_row in conn.execute(
                "SELECT * FROM embedding_chunks WHERE cache_key = ? ORDER BY chunk_order",
                (cache_key,),
            )
        ]
        embedding_dim = int(row["embedding_dim"])
        embedding_count = int(row["embedding_count"])
        if embedding_count != len(embedding_chunks):
            raise ValueError(f"embedding chunk count mismatch meta={embedding_count} rows={len(embedding_chunks)}")
        embeddings = self._load_embedding_matrix(conn, cache_key, embedding_count, embedding_dim)
        document = Document(
            id=str(doc_row["doc_id"]),
            path=file_path,
            title=doc_row["title"],
            pages=int(doc_row["pages"]),
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

    @staticmethod
    def _load_embedding_matrix(
        conn: sqlite3.Connection,
        cache_key: str,
        embedding_count: int,
        embedding_dim: int,
    ) -> np.ndarray:
        if embedding_count == 0:
            return np.empty((0, 0), dtype=np.float32)
        if embedding_dim <= 0:
            raise ValueError(f"invalid embedding_dim={embedding_dim}")
        rows = conn.execute(
            "SELECT vector_blob FROM embeddings WHERE cache_key = ? ORDER BY embedding_order",
            (cache_key,),
        ).fetchall()
        if len(rows) != embedding_count:
            raise ValueError(f"embedding row count mismatch rows={len(rows)} expected={embedding_count}")
        vectors = []
        for row in rows:
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
            if vector.size != embedding_dim:
                raise ValueError(f"embedding vector size mismatch size={vector.size} expected={embedding_dim}")
            vectors.append(vector)
        return normalize_embeddings(np.vstack(vectors))

    def _load_v1_entry_from_row(
        self,
        row: sqlite3.Row,
        file_path: Path,
        cache_key: str,
    ) -> DocumentIndexData:
        document = self._document_from_json(row["document_json"], file_path)
        chunks = [self._chunk_from_json(item) for item in json.loads(row["chunks_json"])]
        embedding_chunks = [self._chunk_from_json(item) for item in json.loads(row["embedding_chunks_json"])]
        embedding_dim = int(row["embedding_dim"])
        embedding_count = int(row["embedding_count"])
        if embedding_count > 0 and embedding_dim <= 0:
            raise ValueError(f"Invalid embedding_dim={embedding_dim}")
        if embedding_count != len(embedding_chunks):
            raise ValueError(f"Embedding count mismatch: stored={embedding_count} chunks={len(embedding_chunks)}")
        embeddings = np.frombuffer(row["embeddings_blob"], dtype=np.float32)
        expected_size = embedding_count * embedding_dim
        if embeddings.size != expected_size:
            raise ValueError(f"Embedding blob size mismatch: size={embeddings.size} expected={expected_size}")
        embeddings = embeddings.reshape((embedding_count, embedding_dim)).copy()
        logger.info(
            "Document index cache loaded legacy cacheKey=%s chunks=%d embedding_chunks=%d",
            cache_key,
            len(chunks),
            len(embedding_chunks),
        )
        return DocumentIndexData(
            document=document,
            chunks=chunks,
            embedding_chunks=embedding_chunks,
            embeddings=embeddings,
        )

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
        embeddings = normalize_embeddings(np.asarray(data.embeddings, dtype=np.float32))
        if embeddings.shape[0] != len(data.embedding_chunks):
            raise ValueError(
                f"Embeddings count mismatch: chunks={len(data.embedding_chunks)} embeddings={embeddings.shape[0]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM embedding_chunks WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM chunks WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM documents WHERE cache_key = ?", (cache_key,))
            conn.execute(
                """
                INSERT INTO cache_meta (
                    cache_key,
                    file_hash,
                    file_id,
                    s3_index,
                    chunk_tokens,
                    embedding_model,
                    schema_version,
                    chunking_version,
                    embedding_split_version,
                    embedding_dim,
                    embedding_count,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', COALESCE(
                    (SELECT created_at FROM cache_meta WHERE cache_key = ?),
                    ?
                ), ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    file_id = EXCLUDED.file_id,
                    s3_index = EXCLUDED.s3_index,
                    chunk_tokens = EXCLUDED.chunk_tokens,
                    embedding_model = EXCLUDED.embedding_model,
                    schema_version = EXCLUDED.schema_version,
                    chunking_version = EXCLUDED.chunking_version,
                    embedding_split_version = EXCLUDED.embedding_split_version,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_count = EXCLUDED.embedding_count,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
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
                    int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
                    int(embeddings.shape[0]),
                    cache_key,
                    now,
                    now,
                ),
            )
            doc = data.document
            conn.execute(
                """
                INSERT INTO documents(cache_key, doc_id, path, title, pages)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, doc.id, str(doc.path), doc.title, int(doc.pages)),
            )
            conn.executemany(
                """
                INSERT INTO chunks(
                    cache_key, chunk_order, doc_id, chunk_id, text, page_start, page_end, heading_path_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        cache_key,
                        index,
                        chunk.doc_id,
                        chunk.chunk_id,
                        chunk.text,
                        int(chunk.page_start),
                        int(chunk.page_end),
                        json.dumps(chunk.heading_path, ensure_ascii=False) if chunk.heading_path is not None else None,
                    )
                    for index, chunk in enumerate(data.chunks)
                ],
            )
            conn.executemany(
                """
                INSERT INTO embedding_chunks(
                    cache_key, chunk_order, doc_id, chunk_id, text, page_start, page_end, heading_path_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        cache_key,
                        index,
                        chunk.doc_id,
                        chunk.chunk_id,
                        chunk.text,
                        int(chunk.page_start),
                        int(chunk.page_end),
                        json.dumps(chunk.heading_path, ensure_ascii=False) if chunk.heading_path is not None else None,
                    )
                    for index, chunk in enumerate(data.embedding_chunks)
                ],
            )
            conn.executemany(
                """
                INSERT INTO embeddings(cache_key, embedding_order, vector_blob)
                VALUES (?, ?, ?)
                """,
                [
                    (cache_key, index, sqlite3.Binary(np.ascontiguousarray(vector, dtype=np.float32).tobytes()))
                    for index, vector in enumerate(embeddings)
                ],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _delete_entry(self, cache_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM embedding_chunks WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM chunks WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM documents WHERE cache_key = ?", (cache_key,))
            conn.execute("DELETE FROM cache_meta WHERE cache_key = ?", (cache_key,))
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

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> DocumentChunk:
        heading_path_raw = row["heading_path_json"]
        return DocumentChunk(
            doc_id=str(row["doc_id"]),
            chunk_id=str(row["chunk_id"]),
            text=str(row["text"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            heading_path=json.loads(heading_path_raw) if heading_path_raw else None,
        )
