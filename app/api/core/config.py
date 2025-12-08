from __future__ import annotations
from pydantic import BaseModel, Field, AliasChoices
from dotenv import load_dotenv
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseModel):
    proxyapi_key: str = os.getenv("PROXYAPI_KEY", "")
    base_url: str = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    model: str = os.getenv("PROXYAPI_MODEL", "gpt-5-nano")
    embedding_model: str = os.getenv("PROXYAPI_EMBEDDING_MODEL", "text-embedding-3-small")


class Settings(BaseSettings):
    """
    Глобальные настройки сервиса.

    Все значения подтягиваются из .env / переменных окружения.
    Поля специально привязаны к уже существующим переменным:

      PROXYAPI_BASE_URL  -> base_url
      PROXYAPI_MODEL     -> model
      PROXYAPI_KEY       -> proxyapi_key

      DATABASE_URL       -> database_url
      REDIS_URL          -> redis_url
      AMQP_URL           -> amqp_url

      MINIO_ENDPOINT     -> minio_endpoint
      MINIO_ROOT_USER    -> minio_access_key
      MINIO_ROOT_PASSWORD-> minio_secret_key

      ELYSIA_PORT        -> elysia_port
      RPC_PORT           -> rpc_port
    """

    # -------- ProxyAPI / OpenAI-совместимый endpoint --------
    # базовый URL прокси (используется в proxy_client и embeddings_client)
    base_url: str = Field(
        ...,
        validation_alias=AliasChoices(
            "PROXYAPI_BASE_URL",  # как в .env
            "proxyapi_base_url",
        ),
        description="Базовый URL ProxyAPI / OpenAI совместимого сервиса",
    )

    # API-ключ для ProxyAPI
    proxyapi_key: str = Field(
        ...,
        validation_alias=AliasChoices(
            "PROXYAPI_KEY",
            "proxyapi_key",
        ),
        description="API-ключ ProxyAPI",
    )

    # Модель чата (используется в proxy_client)
    model: str = Field(
        default="gpt-4.1-mini",
        validation_alias=AliasChoices(
            "PROXYAPI_MODEL",
            "proxyapi_model",
        ),
        description="Имя модели для chat.completions",
    )

    # Модель эмбеддингов (используется в embeddings_client)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "EMBEDDING_MODEL",
            "embedding_model",
        ),
        description="Имя модели для эмбеддингов",
    )

    # -------- Базы данных / брокеры --------
    database_url: str = Field(
        ...,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
        description="Строка подключения к PostgreSQL",
    )

    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
        description="Строка подключения к Redis (если используется)",
    )

    amqp_url: str = Field(
        ...,
        validation_alias=AliasChoices("AMQP_URL", "amqp_url"),
        description="Строка подключения к RabbitMQ (amqp://...)",
    )

    # -------- MinIO / S3 --------
    minio_endpoint: str = Field(
        ...,
        validation_alias=AliasChoices("MINIO_ENDPOINT", "minio_endpoint"),
        description="Endpoint MinIO/S3 (например http://host:9000)",
    )

    minio_access_key: str = Field(
        ...,
        validation_alias=AliasChoices("MINIO_ROOT_USER", "MINIO_ACCESS_KEY", "minio_access_key"),
        description="Access key для MinIO/S3",
    )

    minio_secret_key: str = Field(
        ...,
        validation_alias=AliasChoices("MINIO_ROOT_PASSWORD", "MINIO_SECRET_KEY", "minio_secret_key"),
        description="Secret key для MinIO/S3",
    )

    minio_secure: bool = Field(
        default=False,
        validation_alias=AliasChoices("MINIO_SECURE", "minio_secure"),
        description="Использовать ли HTTPS при подключении к MinIO",
    )

    # -------- Порты вспомогательных сервисов (если нужно) --------
    elysia_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("ELYSIA_PORT", "elysia_port"),
        description="Порт Elysia (если нужен для интеграций)",
    )

    rpc_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("RPC_PORT", "rpc_port"),
        description="Порт RPC (если нужен)",
    )

    # -------- Конфиг pydantic-settings --------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # лишние переменные окружения НЕ вызывают ошибку
    )

    # ---------- MINIO / S3 ----------
    minio_endpoint: str = Field(
        ...,
        description="MinIO/S3 endpoint, например http://sagrefve.ru:9000",
        validation_alias=AliasChoices("MINIO_ENDPOINT", "minio_endpoint"),
    )

    minio_access_key: str = Field(
        ...,
        description="MinIO access key (MINIO_ROOT_USER или MINIO_ACCESS_KEY)",
        validation_alias=AliasChoices("MINIO_ROOT_USER", "MINIO_ACCESS_KEY", "minio_access_key"),
    )

    minio_secret_key: str = Field(
        ...,
        description="MinIO secret key (MINIO_ROOT_PASSWORD или MINIO_SECRET_KEY)",
        validation_alias=AliasChoices("MINIO_ROOT_PASSWORD", "MINIO_SECRET_KEY", "minio_secret_key"),
    )

    minio_secure: bool = Field(
        default=False,
        description="Использовать HTTPS для MinIO/S3",
        validation_alias=AliasChoices("MINIO_SECURE", "minio_secure"),
    )

    # имя бакета, куда будем складывать материалы
    minio_bucket: str = Field(
        ...,
        description="Имя S3/MinIO бакета для учебных материалов",
        validation_alias=AliasChoices("MINIO_BUCKET", "S3_BUCKET_NAME", "minio_bucket"),
    )


settings = Settings()
