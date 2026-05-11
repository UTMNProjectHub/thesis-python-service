from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env and process environment."""

    proxyapi_key: str = Field(
        default="",
        validation_alias=AliasChoices("PROXYAPI_KEY", "proxyapi_key"),
    )
    base_url: str = Field(
        default="https://api.proxyapi.ru/openai/v1",
        validation_alias=AliasChoices("PROXYAPI_BASE_URL", "proxyapi_base_url"),
    )
    model: str = Field(
        default="gpt-4.1-mini",
        validation_alias=AliasChoices("PROXYAPI_MODEL", "proxyapi_model"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "PROXYAPI_EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
            "embedding_model",
        ),
    )

    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )
    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
    )
    amqp_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        validation_alias=AliasChoices("AMQP_URL", "amqp_url"),
    )
    amqp_heartbeat: int = Field(
        default=600,
        validation_alias=AliasChoices("AMQP_HEARTBEAT", "amqp_heartbeat"),
    )

    minio_endpoint: str = Field(
        default="localhost:9000",
        validation_alias=AliasChoices("MINIO_ENDPOINT", "minio_endpoint"),
    )
    minio_access_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MINIO_ROOT_USER",
            "MINIO_ACCESS_KEY",
            "minio_access_key",
        ),
    )
    minio_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MINIO_ROOT_PASSWORD",
            "MINIO_SECRET_KEY",
            "minio_secret_key",
        ),
    )
    minio_secure: bool = Field(
        default=False,
        validation_alias=AliasChoices("MINIO_SECURE", "minio_secure"),
    )
    minio_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices(
            "MINIO_REGION",
            "S3_REGION",
            "AWS_REGION",
            "RUSTFS_REGION",
            "minio_region",
        ),
    )
    minio_bucket: str = Field(
        default="materials",
        validation_alias=AliasChoices("MINIO_BUCKET", "S3_BUCKET_NAME", "minio_bucket"),
    )
    minio_object_prefix: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MINIO_OBJECT_PREFIX",
            "S3_OBJECT_PREFIX",
            "RUSTFS_OBJECT_PREFIX",
            "minio_object_prefix",
        ),
    )
    minio_check_bucket_on_startup: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "MINIO_CHECK_BUCKET_ON_STARTUP",
            "S3_CHECK_BUCKET_ON_STARTUP",
            "minio_check_bucket_on_startup",
        ),
    )
    minio_auto_create_bucket: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "MINIO_AUTO_CREATE_BUCKET",
            "S3_AUTO_CREATE_BUCKET",
            "minio_auto_create_bucket",
        ),
    )
    quiz_source_max_chars: int = Field(
        default=120_000,
        validation_alias=AliasChoices("QUIZ_SOURCE_MAX_CHARS", "quiz_source_max_chars"),
    )
    quiz_source_max_chars_per_file: int = Field(
        default=30_000,
        validation_alias=AliasChoices("QUIZ_SOURCE_MAX_CHARS_PER_FILE", "quiz_source_max_chars_per_file"),
    )
    s3_download_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("S3_DOWNLOAD_CACHE_ENABLED", "s3_download_cache_enabled"),
    )
    s3_download_cache_dir: str = Field(
        default="files_materials/_s3_cache",
        validation_alias=AliasChoices("S3_DOWNLOAD_CACHE_DIR", "s3_download_cache_dir"),
    )
    s3_download_cache_ttl_tasks: int = Field(
        default=3,
        validation_alias=AliasChoices("S3_DOWNLOAD_CACHE_TTL_TASKS", "s3_download_cache_ttl_tasks"),
    )

    elysia_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("ELYSIA_PORT", "elysia_port"),
    )
    rpc_port: int | None = Field(
        default=None,
        validation_alias=AliasChoices("RPC_PORT", "rpc_port"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
