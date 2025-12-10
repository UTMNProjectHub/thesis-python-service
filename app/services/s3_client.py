# app/services/s3_client.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import os
import uuid
import hashlib

from minio import Minio
from minio.error import S3Error

from app.api.core.config import settings


FILES_DIR = Path("files_materials")
FILES_DIR.mkdir(parents=True, exist_ok=True)



class S3Client:
    """
    Обёртка над MinIO/S3.
    Умеет:
    - загружать файл в конкретный бакет (с возвратом ключа);
    - скачивать файл по ключу в локальную папку;
    - упрощённый метод скачивания в каталог files_materials.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        ) -> None:
        endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")

        self.bucket_name = bucket_name or settings.minio_bucket

        self.client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

        # создаём бакет, если его нет
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)

    # ===================== upload =====================

    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = "sha256", chunk_size: int = 4096) -> str:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def upload_file(
            self,
            file_path: str,
            object_name: Optional[str] = None,
    ) -> str:
        """
        Загрузить файл в бакет.
        :param file_path: локальный путь к файлу
        :param object_name: ключ в бакете (если None — берём имя файла)
        :return: object_name (ключ S3)
        """
        p = Path(file_path)
        if object_name is None:
            object_name = p.name  # можно добавить префиксы по желанию

        try:
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(p),
            )
        except S3Error as e:
            print(f"[S3] Ошибка загрузки {file_path}: {e}")
            raise

        return object_name

    # ===================== download =====================

    def download_file(self, object_name: str, destination_path: str) -> str:
        """
        Скачать файл из бакета в destination_path.
        """
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(dest),
            )
        except S3Error as e:
            print(f"[S3] Ошибка скачивания {object_name}: {e}")
            raise

        return str(dest)

    def download_to_materials(self, object_name: str) -> str:
        """
        Скачивает объект в каталог files_materials, имя файла = последний сегмент ключа.
        """
        filename = object_name.split("/")[-1]
        dest = FILES_DIR / filename
        return self.download_file(object_name, str(dest))

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"[S3] Ошибка list_files: {e}")
            return []

    def get_metadata(self, s3_key: str) -> Optional[Dict]:
        try:
            stat = self.client.stat_object(self.bucket_name, s3_key)
            return stat.metadata
        except S3Error as e:
            print(f"[S3] Ошибка получения metadata '{s3_key}': {e}")
            return None

    def upload_file_to_bucket(
            self,
            local_path: str,
            bucket:str,
            original_name: Optional[str] = None,
            user_id: Optional[str] = None,
    ) -> str:
        """
        Загружает файл в бакет и возвращает object_name (s3_key).

        :param local_path: путь к локальному файлу
        :param original_name: исходное имя файла (для читаемости ключа)
        :param user_id: опционально — можно включить в префикс
        :return: s3_key (строка) — ключ объекта в бакете
        """
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
                object_name = f"{bucket}_{safe_name}"
            else:
                object_name = f"{file_id}_{safe_name}"

        file_hash = self.compute_file_hash(local_path)

        metadata = {
            "file_id": file_id,
            "original_name": safe_name,
            "file_hash": file_hash,
        }

        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                local_path,
                metadata=metadata,
            )
            print(f"[S3] Uploaded {local_path} → {self.bucket_name}/{object_name}")
            return object_name
        except S3Error as e:
            print(f"[S3] Ошибка загрузки файла: {e}")
            raise

    def download_file_from_bucket(
            self,
            object_name: str,
            local_filename: Optional[str] = None,
    ) -> Path:
        """
        Скачивает объект из S3/MinIO в директорию files_materials.

        :param object_name: ключ объекта в бакете (s3_key)
        :param local_filename: имя локального файла (по умолчанию = последний сегмент ключа)
        :return: Path до локального файла
        """
        if local_filename is None:
            local_filename = os.path.basename(object_name)

        dest_path = FILES_DIR / local_filename

        try:
            self.client.fget_object(self.bucket_name, object_name, str(dest_path))
            print(f"[S3] Downloaded {self.bucket_name}/{object_name} → {dest_path}")
            return dest_path
        except S3Error as e:
            print(f"[S3] Ошибка скачивания файла: {e}")
            raise


_s3_client: Optional[S3Client] = None

# удобная фабрика
def get_s3_client() -> S3Client:
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client()
    return _s3_client