from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from minio import Minio
from minio.error import S3Error

from app.api.core.config import settings

logger = logging.getLogger(__name__)

FILES_DIR = Path("files_materials")
FILES_DIR.mkdir(parents=True, exist_ok=True)


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
            logger.warning("Removed stale S3 cache lock path=%s age_seconds=%.1f", self.path, age)
            return True
        except OSError as e:
            logger.warning("Could not remove stale S3 cache lock path=%s error=%s", self.path, e)
            return False

    def acquire(self, *, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(self.timeout_seconds, 0.0)

        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n"
                os.write(self._fd, payload.encode("utf-8"))
                return True
            except FileExistsError:
                if self._remove_if_stale():
                    continue
                if not blocking:
                    return False
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for S3 cache lock: {self.path}")
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
            logger.warning("Could not remove S3 cache lock path=%s error=%s", self.path, e)


def _normalize_endpoint(raw_endpoint: str) -> tuple[str, bool]:
    endpoint = raw_endpoint.strip()
    secure = settings.minio_secure
    parsed = urlparse(endpoint)

    if parsed.scheme in ("http", "https"):
        if not parsed.netloc:
            raise ValueError(f"Invalid MINIO_ENDPOINT: {raw_endpoint!r}")

        endpoint = parsed.netloc
        secure = parsed.scheme == "https"

        if parsed.path and parsed.path != "/":
            logger.warning(
                "Ignoring path in MINIO_ENDPOINT path=%s. Put bucket name into MINIO_BUCKET instead.",
                parsed.path,
            )

    return endpoint, secure


class S3Client:
    """Small MinIO/S3 wrapper used by the worker pipeline."""

    def __init__(self, bucket_name: Optional[str] = None) -> None:
        endpoint, secure = _normalize_endpoint(settings.minio_endpoint)
        region = settings.minio_region.strip() or None
        self.bucket_name = bucket_name or settings.minio_bucket
        self.cache_enabled = settings.s3_download_cache_enabled
        self.cache_dir = Path(settings.s3_download_cache_dir)
        self.cache_index_path = self.cache_dir / "index.json"
        self.cache_ttl_tasks = settings.s3_download_cache_ttl_tasks
        self.cache_lock_timeout_seconds = settings.s3_download_cache_lock_timeout_seconds
        self.cache_stale_temp_seconds = settings.s3_download_cache_stale_temp_seconds
        self.cache_stale_lock_seconds = settings.s3_download_cache_stale_lock_seconds
        self.cache_io_retries = settings.s3_download_cache_io_retries
        self.cache_io_retry_delay_seconds = settings.s3_download_cache_io_retry_delay_seconds
        self._cache_lock = threading.RLock()
        self._object_locks: dict[str, threading.RLock] = {}
        self._held_usage_locks: dict[str, dict[str, Any]] = {}

        logger.info(
            "Initializing S3 client endpoint=%s secure=%s region=%s bucket=%s",
            endpoint,
            secure,
            region,
            self.bucket_name,
        )
        self.client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=secure,
            region=region,
        )

        if not settings.minio_check_bucket_on_startup:
            logger.info("S3 bucket startup validation skipped bucket=%s", self.bucket_name)
            self._prepare_cache()
            return

        try:
            found = self.client.bucket_exists(self.bucket_name)
        except S3Error as e:
            logger.exception(
                "S3 bucket validation failed bucket=%s endpoint=%s secure=%s region=%s code=%s error=%s",
                self.bucket_name,
                endpoint,
                secure,
                region,
                getattr(e, "code", None),
                e,
            )
            raise RuntimeError(
                "S3 bucket validation failed. For RustFS/S3-compatible storage check "
                "MINIO_ENDPOINT=https://..., MINIO_SECURE=true, MINIO_REGION=us-east-1 "
                "or the real bucket region. If the key has no bucket validation rights, "
                "set MINIO_CHECK_BUCKET_ON_STARTUP=false."
            ) from e

        if not found:
            if not settings.minio_auto_create_bucket:
                raise RuntimeError(
                    f"S3 bucket {self.bucket_name!r} does not exist and MINIO_AUTO_CREATE_BUCKET=false"
                )
            logger.info("S3 bucket missing, creating bucket=%s endpoint=%s", self.bucket_name, endpoint)
            self.client.make_bucket(self.bucket_name)
        else:
            logger.info("S3 bucket ready bucket=%s endpoint=%s", self.bucket_name, endpoint)

        self._prepare_cache()

    def _prepare_cache(self) -> None:
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_stale_cache_files()
            logger.info(
                "S3 download cache enabled dir=%s ttl_tasks=%d",
                self.cache_dir,
                self.cache_ttl_tasks,
            )
        else:
            logger.info("S3 download cache disabled")

    def _normalize_object_name(self, object_name: str) -> str:
        raw = str(object_name).strip().replace("\\", "/")
        parsed = urlparse(raw)

        if parsed.scheme in ("http", "https"):
            key = unquote(parsed.path).lstrip("/")
        else:
            key = raw.split("?", 1)[0].lstrip("/")

        bucket_prefix = f"{self.bucket_name}/"
        while key.startswith(bucket_prefix):
            key = key[len(bucket_prefix):]

        return key

    def _object_candidates(self, object_name: str) -> list[str]:
        base = self._normalize_object_name(object_name)
        candidates: list[str] = []

        configured_prefix = settings.minio_object_prefix.strip().strip("/")
        if configured_prefix and not base.startswith(f"{configured_prefix}/"):
            candidates.append(f"{configured_prefix}/{base}")

        candidates.append(base)

        bucket_prefixed = f"{self.bucket_name}/{base}"
        candidates.append(bucket_prefixed)

        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _cache_key(self, object_name: str) -> str:
        normalized = self._normalize_object_name(object_name)
        return hashlib.sha256(f"{self.bucket_name}/{normalized}".encode("utf-8")).hexdigest()

    def _cache_path(self, object_name: str) -> Path:
        normalized = self._normalize_object_name(object_name)
        suffix = Path(normalized).suffix
        return self.cache_dir / f"{self._cache_key(normalized)}{suffix}"

    def _get_object_lock(self, cache_key: str) -> threading.RLock:
        with self._cache_lock:
            lock = self._object_locks.get(cache_key)
            if lock is None:
                lock = threading.RLock()
                self._object_locks[cache_key] = lock
            return lock

    def _cache_lock_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.lock"

    def _index_lock_path(self) -> Path:
        return self.cache_dir / "index.lock"

    def _new_lock(self, path: Path) -> _FileLock:
        return _FileLock(
            path,
            self.cache_lock_timeout_seconds,
            stale_seconds=self.cache_stale_lock_seconds,
        )

    def _index_lock(self) -> _FileLock:
        return self._new_lock(self._index_lock_path())

    def _is_retryable_io_error(self, exc: OSError) -> bool:
        winerror = getattr(exc, "winerror", None)
        return winerror in (5, 32) or isinstance(exc, PermissionError)

    def _retry_io(self, operation, description: str):
        attempts = max(int(self.cache_io_retries), 0) + 1
        delay = max(float(self.cache_io_retry_delay_seconds), 0.0)

        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except OSError as e:
                if not self._is_retryable_io_error(e) or attempt >= attempts:
                    raise
                sleep_for = min(delay * (2 ** (attempt - 1)), 3.0)
                logger.warning(
                    "Retrying S3 cache IO operation description=%s attempt=%d/%d sleep=%.2f error=%s",
                    description,
                    attempt,
                    attempts,
                    sleep_for,
                    e,
                )
                time.sleep(sleep_for)
        return None

    def _replace_file_with_retry(self, source: Path, destination: Path) -> None:
        self._retry_io(lambda: os.replace(source, destination), f"replace {source} -> {destination}")

    def _unlink_with_retry(self, path: Path, description: str) -> bool:
        if not path.exists():
            return True

        try:
            self._retry_io(path.unlink, description)
            return True
        except OSError as e:
            logger.warning("S3 cache file could not be removed path=%s description=%s error=%s", path, description, e)
            return False

    def _cleanup_stale_cache_files(self) -> None:
        if not self.cache_dir.exists():
            return

        now = time.time()
        removed: list[str] = []
        patterns = ("*.download", "*.part.minio", "index.*.tmp")
        for pattern in patterns:
            for path in self.cache_dir.glob(pattern):
                try:
                    age = now - path.stat().st_mtime
                except OSError:
                    continue
                if age < self.cache_stale_temp_seconds:
                    continue
                if self._unlink_with_retry(path, f"stale temp cleanup age={age:.1f}"):
                    removed.append(str(path))

        for path in self.cache_dir.glob("*.lock"):
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age < self.cache_stale_lock_seconds:
                continue
            if self._unlink_with_retry(path, f"stale lock cleanup age={age:.1f}"):
                removed.append(str(path))

        if removed:
            logger.info("S3 cache stale files removed count=%d files=%s", len(removed), removed)

    @staticmethod
    def _held_lock_id(task_id: str, cache_key: str) -> str:
        return f"{task_id}:{cache_key}"

    def _acquire_cache_usage_lock(self, cache_key: str, task_id: str | None) -> _FileLock:
        if task_id:
            held_id = self._held_lock_id(task_id, cache_key)
            with self._cache_lock:
                held = self._held_usage_locks.get(held_id)
                if held is not None:
                    held["count"] = int(held.get("count", 1)) + 1
                    logger.info(
                        "S3 cache usage lock reused taskId=%s key=%s count=%d",
                        task_id,
                        cache_key,
                        held["count"],
                    )
                    return held["lock"]

        lock = self._new_lock(self._cache_lock_path(cache_key))
        lock.acquire(blocking=True)
        if task_id:
            with self._cache_lock:
                self._held_usage_locks[self._held_lock_id(task_id, cache_key)] = {
                    "lock": lock,
                    "count": 1,
                }
            logger.info("S3 cache usage lock acquired taskId=%s key=%s", task_id, cache_key)
        else:
            logger.info("S3 cache transient lock acquired key=%s", cache_key)
        return lock

    def _release_cache_usage_lock(
            self,
            cache_key: str,
            task_id: str | None,
            lock: _FileLock | None = None,
            *,
            release_all: bool = False,
    ) -> None:
        if task_id:
            held_id = self._held_lock_id(task_id, cache_key)
            with self._cache_lock:
                held = self._held_usage_locks.get(held_id)
                if held is None:
                    return
                count = int(held.get("count", 1))
                if count > 1 and not release_all:
                    held["count"] = count - 1
                    return
                lock = held["lock"]
                del self._held_usage_locks[held_id]

        if lock is not None:
            lock.release()
            logger.info("S3 cache usage lock released taskId=%s key=%s", task_id, cache_key)

    def _try_acquire_cache_cleanup_lock(self, cache_key: str) -> _FileLock | None:
        lock = self._new_lock(self._cache_lock_path(cache_key))
        if not lock.acquire(blocking=False):
            return None
        return lock

    def _stream_object_to_file(self, object_name: str, destination: Path) -> None:
        response = None
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            with destination.open("wb") as f:
                for chunk in response.stream(1024 * 1024):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _empty_cache_index(self) -> dict[str, Any]:
        return {"version": 1, "entries": {}}

    def _load_cache_index(self) -> dict[str, Any]:
        if not self.cache_index_path.exists():
            return self._empty_cache_index()

        try:
            with self.cache_index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception as e:
            logger.warning("S3 cache index could not be read path=%s error=%s", self.cache_index_path, e)
            return self._empty_cache_index()

        if not isinstance(index, dict) or not isinstance(index.get("entries"), dict):
            logger.warning("S3 cache index has invalid structure path=%s", self.cache_index_path)
            return self._empty_cache_index()

        return index

    def _save_cache_index(self, index: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_index_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            self._replace_file_with_retry(tmp_path, self.cache_index_path)
        finally:
            if tmp_path.exists():
                self._unlink_with_retry(tmp_path, "cache index temp cleanup")

    @staticmethod
    def _metadata_hash(metadata: Optional[dict[str, str]]) -> Optional[str]:
        if not metadata:
            return None

        for key, value in metadata.items():
            if "file_hash" in key.lower().replace("-", "_"):
                return str(value).lower()
        return None

    @staticmethod
    def _etag_md5(etag: Optional[str]) -> Optional[str]:
        if not etag:
            return None

        value = str(etag).strip('"').lower()
        if len(value) == 32 and all(ch in "0123456789abcdef" for ch in value):
            return value
        return None

    @staticmethod
    def _stat_signature(stat: Any) -> dict[str, Any]:
        last_modified = getattr(stat, "last_modified", None)
        if last_modified is not None and hasattr(last_modified, "isoformat"):
            last_modified_value = last_modified.isoformat()
        else:
            last_modified_value = str(last_modified) if last_modified else None

        metadata = getattr(stat, "metadata", None) or {}
        return {
            "size": int(getattr(stat, "size", 0) or 0),
            "etag": getattr(stat, "etag", None),
            "last_modified": last_modified_value,
            "metadata_file_hash": S3Client._metadata_hash(metadata),
        }

    def _cached_file_is_valid(
            self,
            cache_key: str,
            cache_path: Path,
            stat: Any,
            index: dict[str, Any],
    ) -> bool:
        if not cache_path.exists():
            return False

        signature = self._stat_signature(stat)
        local_size = cache_path.stat().st_size
        if signature["size"] and local_size != signature["size"]:
            logger.warning(
                "S3 cache size mismatch key=%s path=%s local_size=%d remote_size=%d",
                cache_key,
                cache_path,
                local_size,
                signature["size"],
            )
            return False

        remote_hash = signature.get("metadata_file_hash")
        if remote_hash:
            local_hash = self.compute_file_hash(str(cache_path)).lower()
            if local_hash != remote_hash:
                logger.warning(
                    "S3 cache hash mismatch key=%s path=%s local_hash=%s remote_hash=%s",
                    cache_key,
                    cache_path,
                    local_hash,
                    remote_hash,
                )
                return False
            logger.info("S3 cache hash validated key=%s path=%s hash=%s", cache_key, cache_path, local_hash)
            return True

        etag_md5 = self._etag_md5(signature.get("etag"))
        if etag_md5:
            local_md5 = self.compute_file_hash(str(cache_path), algorithm="md5").lower()
            if local_md5 != etag_md5:
                logger.warning(
                    "S3 cache ETag MD5 mismatch key=%s path=%s local_md5=%s etag=%s",
                    cache_key,
                    cache_path,
                    local_md5,
                    etag_md5,
                )
                return False
            logger.info("S3 cache ETag MD5 validated key=%s path=%s md5=%s", cache_key, cache_path, local_md5)
            return True

        entry = index.get("entries", {}).get(cache_key)
        if not entry:
            logger.info("S3 cache file exists without index entry key=%s path=%s", cache_key, cache_path)
            return False

        valid = (
                entry.get("size") == signature["size"]
                and entry.get("etag") == signature["etag"]
                and entry.get("last_modified") == signature["last_modified"]
        )
        if not valid:
            logger.warning("S3 cache signature mismatch key=%s path=%s", cache_key, cache_path)
        return valid

    def _downloaded_file_matches_stat(self, cache_path: Path, stat: Any) -> bool:
        signature = self._stat_signature(stat)
        local_size = cache_path.stat().st_size
        if signature["size"] and local_size != signature["size"]:
            logger.warning(
                "S3 downloaded cache size mismatch path=%s local_size=%d remote_size=%d",
                cache_path,
                local_size,
                signature["size"],
            )
            return False

        remote_hash = signature.get("metadata_file_hash")
        if remote_hash:
            local_hash = self.compute_file_hash(str(cache_path)).lower()
            if local_hash != remote_hash:
                logger.warning(
                    "S3 downloaded cache hash mismatch path=%s local_hash=%s remote_hash=%s",
                    cache_path,
                    local_hash,
                    remote_hash,
                )
                return False
            logger.info("S3 downloaded cache hash validated path=%s hash=%s", cache_path, local_hash)

        etag_md5 = self._etag_md5(signature.get("etag"))
        if etag_md5:
            local_md5 = self.compute_file_hash(str(cache_path), algorithm="md5").lower()
            if local_md5 != etag_md5:
                logger.warning(
                    "S3 downloaded cache ETag MD5 mismatch path=%s local_md5=%s etag=%s",
                    cache_path,
                    local_md5,
                    etag_md5,
                )
                return False
            logger.info("S3 downloaded cache ETag MD5 validated path=%s md5=%s", cache_path, local_md5)

        return True

    def _write_cache_entry(
            self,
            cache_key: str,
            cache_path: Path,
            object_name: str,
            stat: Any,
            index: dict[str, Any],
    ) -> None:
        signature = self._stat_signature(stat)
        index["entries"][cache_key] = {
            "bucket": self.bucket_name,
            "object": object_name,
            "path": str(cache_path),
            "size": signature["size"],
            "etag": signature["etag"],
            "last_modified": signature["last_modified"],
            "metadata_file_hash": signature["metadata_file_hash"],
            "local_sha256": self.compute_file_hash(str(cache_path)),
            "unused_tasks": 0,
            "last_used_at": datetime.now(timezone.utc).isoformat(),
        }

    def _with_index(self, callback):
        index_lock = self._index_lock()
        index_lock.acquire(blocking=True)
        try:
            with self._cache_lock:
                index = self._load_cache_index()
                result = callback(index)
                self._save_cache_index(index)
                return result
        finally:
            index_lock.release()

    def _download_to_cache(self, object_name: str, task_id: str | None = None) -> str:
        candidates = self._object_candidates(object_name)
        last_error: S3Error | None = None

        for candidate in candidates:
            cache_key = self._cache_key(candidate)
            thread_lock = self._get_object_lock(cache_key)

            with thread_lock:
                usage_lock: _FileLock | None = None
                keep_usage_lock = False
                try:
                    usage_lock = self._acquire_cache_usage_lock(cache_key, task_id)
                    try:
                        stat = self.client.stat_object(self.bucket_name, candidate)
                    except S3Error as e:
                        last_error = e
                        if getattr(e, "code", None) == "NoSuchKey":
                            logger.warning(
                                "S3 cache candidate missing bucket=%s object=%s original_object=%s",
                                self.bucket_name,
                                candidate,
                                object_name,
                            )
                            continue
                        logger.exception(
                            "S3 cache stat failed bucket=%s object=%s error=%s",
                            self.bucket_name,
                            candidate,
                            e,
                        )
                        raise

                    cache_path = self._cache_path(candidate)

                    def mark_cache_hit(index: dict[str, Any]) -> bool:
                        if self._cached_file_is_valid(cache_key, cache_path, stat, index):
                            entry = index["entries"].setdefault(cache_key, {})
                            entry["unused_tasks"] = 0
                            entry["last_used_at"] = datetime.now(timezone.utc).isoformat()
                            entry["path"] = str(cache_path)
                            return True
                        return False

                    if self._with_index(mark_cache_hit):
                        logger.info(
                            "S3 cache hit bucket=%s object=%s path=%s taskId=%s",
                            self.bucket_name,
                            candidate,
                            cache_path,
                            task_id,
                        )
                        keep_usage_lock = task_id is not None
                        return str(cache_path)

                    tmp_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.download")
                    try:
                        self._cleanup_stale_cache_files()
                        logger.info(
                            "S3 cache miss, downloading bucket=%s object=%s cache_path=%s taskId=%s",
                            self.bucket_name,
                            candidate,
                            cache_path,
                            task_id,
                        )
                        self._stream_object_to_file(candidate, tmp_path)
                        self._replace_file_with_retry(tmp_path, cache_path)
                    finally:
                        if tmp_path.exists():
                            self._unlink_with_retry(tmp_path, "cache download temp cleanup")

                    def write_downloaded_entry(index: dict[str, Any]) -> None:
                        if not self._downloaded_file_matches_stat(cache_path, stat):
                            raise RuntimeError(f"S3 cache validation failed for object {candidate!r}")
                        self._write_cache_entry(cache_key, cache_path, candidate, stat, index)

                    self._with_index(write_downloaded_entry)

                    logger.info(
                        "S3 cache stored bucket=%s object=%s path=%s taskId=%s",
                        self.bucket_name,
                        candidate,
                        cache_path,
                        task_id,
                    )
                    keep_usage_lock = task_id is not None
                    return str(cache_path)
                finally:
                    if not keep_usage_lock:
                        self._release_cache_usage_lock(cache_key, task_id, usage_lock)

        self._log_missing_object_context(object_name, candidates)
        assert last_error is not None
        raise last_error

    def finish_task_cache_usage(self, used_object_names: List[str], task_id: str | None = None) -> None:
        if not self.cache_enabled:
            return

        task_id_str = str(task_id) if task_id is not None else None
        used_keys = {
            self._cache_key(candidate)
            for object_name in used_object_names
            for candidate in self._object_candidates(object_name)
        }

        if task_id_str:
            for cache_key in used_keys:
                self._release_cache_usage_lock(cache_key, task_id_str, release_all=True)

        def update_index(index: dict[str, Any]) -> list[str]:
            entries = index.get("entries", {})
            deleted: list[str] = []

            for cache_key, entry in list(entries.items()):
                if cache_key in used_keys:
                    entry["unused_tasks"] = 0
                    entry["last_used_at"] = datetime.now(timezone.utc).isoformat()
                    continue

                entry["unused_tasks"] = int(entry.get("unused_tasks", 0)) + 1
                if self.cache_ttl_tasks >= 0 and entry["unused_tasks"] >= self.cache_ttl_tasks:
                    cache_path = Path(entry.get("path", ""))
                    cleanup_lock = self._try_acquire_cache_cleanup_lock(cache_key)
                    if cleanup_lock is None:
                        logger.info("S3 cache TTL skipped locked file key=%s path=%s", cache_key, cache_path)
                        continue
                    try:
                        if cache_path.exists() and not self._unlink_with_retry(cache_path, "cache TTL cleanup"):
                            continue
                    finally:
                        cleanup_lock.release()
                    deleted.append(str(cache_path))
                    del entries[cache_key]

            return deleted

        deleted = self._with_index(update_index)

        if deleted:
            logger.info("S3 cache TTL removed files count=%d files=%s", len(deleted), deleted)

    def _log_missing_object_context(self, object_name: str, candidates: list[str]) -> None:
        prefixes: list[str] = []
        for candidate in candidates:
            parts = candidate.split("/")
            if len(parts) > 1:
                prefixes.append("/".join(parts[:-1]) + "/")
            if parts:
                prefixes.append(parts[0] + "/")

        seen: list[str] = []
        for prefix in prefixes:
            if prefix not in seen:
                seen.append(prefix)

        for prefix in seen[:4]:
            try:
                objects = self.client.list_objects(
                    bucket_name=self.bucket_name,
                    prefix=prefix,
                    recursive=True,
                )
                nearby = []
                for obj in objects:
                    nearby.append(obj.object_name)
                    if len(nearby) >= 20:
                        break

                logger.warning(
                    "S3 object missing context requested=%s prefix=%s nearby_count=%d nearby=%s",
                    object_name,
                    prefix,
                    len(nearby),
                    nearby,
                )
            except S3Error as e:
                logger.warning(
                    "S3 object missing context unavailable requested=%s prefix=%s code=%s error=%s",
                    object_name,
                    prefix,
                    getattr(e, "code", None),
                    e,
                )

    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = "sha256", chunk_size: int = 4096) -> str:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def upload_file(self, file_path: str, object_name: Optional[str] = None) -> str:
        p = Path(file_path)
        if object_name is None:
            object_name = p.name
        object_name = self._object_candidates(object_name)[0]

        try:
            logger.info("S3 upload started bucket=%s object=%s path=%s", self.bucket_name, object_name, p)
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(p),
            )
            logger.info("S3 upload finished bucket=%s object=%s", self.bucket_name, object_name)
        except S3Error as e:
            logger.exception(
                "S3 upload failed bucket=%s object=%s path=%s error=%s",
                self.bucket_name,
                object_name,
                p,
                e,
            )
            raise

        return object_name

    def download_file(self, object_name: str, destination_path: str) -> str:
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        candidates = self._object_candidates(object_name)
        last_error: S3Error | None = None

        for candidate in candidates:
            try:
                logger.info(
                    "S3 download started bucket=%s object=%s destination=%s original_object=%s",
                    self.bucket_name,
                    candidate,
                    dest,
                    object_name,
                )
                self.client.fget_object(
                    bucket_name=self.bucket_name,
                    object_name=candidate,
                    file_path=str(dest),
                )
                logger.info(
                    "S3 download finished bucket=%s object=%s destination=%s",
                    self.bucket_name,
                    candidate,
                    dest,
                )
                return str(dest)
            except S3Error as e:
                last_error = e
                if getattr(e, "code", None) == "NoSuchKey":
                    logger.warning(
                        "S3 download candidate missing bucket=%s object=%s original_object=%s",
                        self.bucket_name,
                        candidate,
                        object_name,
                    )
                    continue

                logger.exception(
                    "S3 download failed bucket=%s object=%s destination=%s error=%s",
                    self.bucket_name,
                    candidate,
                    dest,
                    e,
                )
                raise

        self._log_missing_object_context(object_name, candidates)
        assert last_error is not None
        logger.error(
            "S3 download failed: object not found bucket=%s requested=%s tried=%s",
            self.bucket_name,
            object_name,
            candidates,
        )
        raise last_error

    def download_to_materials(
            self,
            object_name: str,
            materials_dir: Path | str = FILES_DIR,
            task_id: str | None = None,
    ) -> str:
        if self.cache_enabled:
            return self._download_to_cache(object_name, task_id=task_id)

        filename = self._normalize_object_name(object_name).split("/")[-1]
        dest = Path(materials_dir) / filename
        return self.download_file(object_name, str(dest))

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            logger.info("S3 list started bucket=%s prefix=%s", self.bucket_name, prefix)
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True,
            )
            result = [obj.object_name for obj in objects]
            logger.info("S3 list finished bucket=%s prefix=%s count=%d", self.bucket_name, prefix, len(result))
            return result
        except S3Error as e:
            logger.exception("S3 list failed bucket=%s prefix=%s error=%s", self.bucket_name, prefix, e)
            return []

    def get_metadata(self, s3_key: str) -> Optional[Dict]:
        candidates = self._object_candidates(s3_key)
        for candidate in candidates:
            try:
                logger.info("S3 metadata request bucket=%s object=%s original_object=%s", self.bucket_name, candidate,
                            s3_key)
                stat = self.client.stat_object(self.bucket_name, candidate)
                logger.info(
                    "S3 metadata loaded bucket=%s object=%s metadata_keys=%s",
                    self.bucket_name,
                    candidate,
                    sorted(stat.metadata.keys()),
                )
                return stat.metadata
            except S3Error as e:
                if getattr(e, "code", None) == "NoSuchKey":
                    continue
                logger.exception("S3 metadata failed bucket=%s object=%s error=%s", self.bucket_name, candidate, e)
                return None

        self._log_missing_object_context(s3_key, candidates)
        logger.warning("S3 metadata not found bucket=%s requested=%s tried=%s", self.bucket_name, s3_key, candidates)
        return None

    def upload_file_to_bucket(
            self,
            local_path: str,
            bucket: str,
            original_name: Optional[str] = None,
            user_id: Optional[str] = None,
    ) -> str:
        local_path = str(local_path)
        if original_name is None:
            original_name = os.path.basename(local_path)

        safe_name = original_name.encode("ascii", "ignore").decode("ascii") or "file"
        file_id = str(uuid.uuid4())

        if user_id:
            if bucket:
                object_name = f"{bucket}/{file_id}_{safe_name}"
            else:
                object_name = f"{user_id}/{file_id}_{safe_name}"
        else:
            if bucket:
                object_name = f"{bucket}{file_id}_{safe_name}"
            else:
                object_name = f"{file_id}_{safe_name}"
        object_name = self._object_candidates(object_name)[0]

        file_hash = self.compute_file_hash(local_path)
        metadata = {
            "file_id": file_id,
            "original_name": safe_name,
            "file_hash": file_hash,
        }

        try:
            logger.info(
                "S3 bucket upload started bucket=%s object=%s path=%s original_name=%s user_id=%s hash=%s",
                self.bucket_name,
                object_name,
                local_path,
                safe_name,
                user_id,
                file_hash,
            )
            self.client.fput_object(
                self.bucket_name,
                object_name,
                local_path,
                metadata=metadata,
            )
            logger.info("S3 bucket upload finished bucket=%s object=%s", self.bucket_name, object_name)
            return object_name
        except S3Error as e:
            logger.exception(
                "S3 bucket upload failed bucket=%s object=%s path=%s error=%s",
                self.bucket_name,
                object_name,
                local_path,
                e,
            )
            raise

    def download_file_from_bucket(
            self,
            object_name: str,
            local_filename: Optional[str] = None,
    ) -> Path:
        if local_filename is None:
            local_filename = os.path.basename(self._normalize_object_name(object_name))

        dest_path = FILES_DIR / local_filename
        candidates = self._object_candidates(object_name)
        last_error: S3Error | None = None

        for candidate in candidates:
            try:
                logger.info(
                    "S3 bucket download started bucket=%s object=%s destination=%s original_object=%s",
                    self.bucket_name,
                    candidate,
                    dest_path,
                    object_name,
                )
                self.client.fget_object(self.bucket_name, candidate, str(dest_path))
                logger.info(
                    "S3 bucket download finished bucket=%s object=%s destination=%s",
                    self.bucket_name,
                    candidate,
                    dest_path,
                )
                return dest_path
            except S3Error as e:
                last_error = e
                if getattr(e, "code", None) == "NoSuchKey":
                    logger.warning(
                        "S3 bucket download candidate missing bucket=%s object=%s original_object=%s",
                        self.bucket_name,
                        candidate,
                        object_name,
                    )
                    continue
                logger.exception(
                    "S3 bucket download failed bucket=%s object=%s destination=%s error=%s",
                    self.bucket_name,
                    candidate,
                    dest_path,
                    e,
                )
                raise

        self._log_missing_object_context(object_name, candidates)
        assert last_error is not None
        logger.error(
            "S3 bucket download failed: object not found bucket=%s requested=%s tried=%s",
            self.bucket_name,
            object_name,
            candidates,
        )
        raise last_error


_s3_client: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client()
    return _s3_client
