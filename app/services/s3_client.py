# app/services/s3_client.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

from minio import Minio
from minio.error import S3Error

from app.api.core.config import settings


FILES_MATERIALS_DIR = Path("files_materials")


class S3Client:
    """
    Обёртка над MinIO/S3.
    Умеет:
    - загружать файл в конкретный бакет (с возвратом ключа);
    - скачивать файл по ключу в локальную папку;
    - упрощённый метод скачивания в каталог files_materials.
    """

    def __init__(self) -> None:
        endpoint = settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        self.bucket_name = settings.MINIO_BUCKET
        self.client = Minio(
            endpoint,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_ENDPOINT.startswith("https://"),
        )
        # создаём бакет, если его нет
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

        FILES_MATERIALS_DIR.mkdir(parents=True, exist_ok=True)

    # ===================== upload =====================

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
        dest = FILES_MATERIALS_DIR / filename
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


# удобная фабрика
def get_s3_client() -> S3Client:
    return S3Client()
