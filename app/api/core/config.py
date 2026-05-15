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
    quiz_generation_use_source_documents: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "QUIZ_GENERATION_USE_SOURCE_DOCUMENTS",
            "quiz_generation_use_source_documents",
        ),
    )
    faq_generation_use_source_documents: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "FAQ_GENERATION_USE_SOURCE_DOCUMENTS",
            "faq_generation_use_source_documents",
        ),
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
    s3_download_cache_lock_timeout_seconds: float = Field(
        default=900.0,
        validation_alias=AliasChoices(
            "S3_DOWNLOAD_CACHE_LOCK_TIMEOUT_SECONDS",
            "s3_download_cache_lock_timeout_seconds",
        ),
    )
    s3_download_cache_stale_temp_seconds: float = Field(
        default=3600.0,
        validation_alias=AliasChoices(
            "S3_DOWNLOAD_CACHE_STALE_TEMP_SECONDS",
            "s3_download_cache_stale_temp_seconds",
        ),
    )
    s3_download_cache_stale_lock_seconds: float = Field(
        default=21600.0,
        validation_alias=AliasChoices(
            "S3_DOWNLOAD_CACHE_STALE_LOCK_SECONDS",
            "s3_download_cache_stale_lock_seconds",
        ),
    )
    s3_download_cache_io_retries: int = Field(
        default=8,
        validation_alias=AliasChoices("S3_DOWNLOAD_CACHE_IO_RETRIES", "s3_download_cache_io_retries"),
    )
    s3_download_cache_io_retry_delay_seconds: float = Field(
        default=0.25,
        validation_alias=AliasChoices(
            "S3_DOWNLOAD_CACHE_IO_RETRY_DELAY_SECONDS",
            "s3_download_cache_io_retry_delay_seconds",
        ),
    )
    faq_batch_size: int = Field(
        default=20,
        validation_alias=AliasChoices("FAQ_BATCH_SIZE", "faq_batch_size"),
    )
    faq_completion_tokens_per_question: int = Field(
        default=220,
        validation_alias=AliasChoices(
            "FAQ_COMPLETION_TOKENS_PER_QUESTION",
            "faq_completion_tokens_per_question",
        ),
    )
    faq_max_completion_tokens: int = Field(
        default=8000,
        validation_alias=AliasChoices("FAQ_MAX_COMPLETION_TOKENS", "faq_max_completion_tokens"),
    )
    lecture_target_words: int = Field(
        default=5000,
        validation_alias=AliasChoices("LECTURE_TARGET_WORDS", "lecture_target_words"),
    )
    lecture_words_per_section: int = Field(
        default=850,
        validation_alias=AliasChoices("LECTURE_WORDS_PER_SECTION", "lecture_words_per_section"),
    )
    lecture_min_sections: int = Field(
        default=3,
        validation_alias=AliasChoices("LECTURE_MIN_SECTIONS", "lecture_min_sections"),
    )
    lecture_max_sections: int = Field(
        default=9,
        validation_alias=AliasChoices("LECTURE_MAX_SECTIONS", "lecture_max_sections"),
    )
    lecture_chunk_tokens: int = Field(
        default=700,
        validation_alias=AliasChoices("LECTURE_CHUNK_TOKENS", "lecture_chunk_tokens"),
    )
    lecture_base_plan_context_chunks: int = Field(
        default=16,
        validation_alias=AliasChoices("LECTURE_BASE_PLAN_CONTEXT_CHUNKS", "lecture_base_plan_context_chunks"),
    )
    lecture_base_section_context_chunks: int = Field(
        default=10,
        validation_alias=AliasChoices("LECTURE_BASE_SECTION_CONTEXT_CHUNKS", "lecture_base_section_context_chunks"),
    )
    lecture_max_plan_context_chunks: int = Field(
        default=64,
        validation_alias=AliasChoices("LECTURE_MAX_PLAN_CONTEXT_CHUNKS", "lecture_max_plan_context_chunks"),
    )
    lecture_max_section_context_chunks: int = Field(
        default=32,
        validation_alias=AliasChoices("LECTURE_MAX_SECTION_CONTEXT_CHUNKS", "lecture_max_section_context_chunks"),
    )
    lecture_retrieval_pool_k: int = Field(
        default=40,
        validation_alias=AliasChoices("LECTURE_RETRIEVAL_POOL_K", "lecture_retrieval_pool_k"),
    )
    lecture_max_retrieval_pool_k: int = Field(
        default=180,
        validation_alias=AliasChoices("LECTURE_MAX_RETRIEVAL_POOL_K", "lecture_max_retrieval_pool_k"),
    )
    lecture_plan_context_token_budget: int = Field(
        default=24_000,
        validation_alias=AliasChoices("LECTURE_PLAN_CONTEXT_TOKEN_BUDGET", "lecture_plan_context_token_budget"),
    )
    lecture_section_context_token_budget: int = Field(
        default=16_000,
        validation_alias=AliasChoices("LECTURE_SECTION_CONTEXT_TOKEN_BUDGET", "lecture_section_context_token_budget"),
    )
    lecture_document_profiles_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LECTURE_DOCUMENT_PROFILES_ENABLED", "lecture_document_profiles_enabled"),
    )
    lecture_doc_profile_chunks: int = Field(
        default=8,
        validation_alias=AliasChoices("LECTURE_DOC_PROFILE_CHUNKS", "lecture_doc_profile_chunks"),
    )
    lecture_doc_profile_max_chunks: int = Field(
        default=16,
        validation_alias=AliasChoices("LECTURE_DOC_PROFILE_MAX_CHUNKS", "lecture_doc_profile_max_chunks"),
    )
    lecture_doc_profile_max_tokens: int = Field(
        default=700,
        validation_alias=AliasChoices("LECTURE_DOC_PROFILE_MAX_TOKENS", "lecture_doc_profile_max_tokens"),
    )
    lecture_final_edit_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LECTURE_FINAL_EDIT_ENABLED", "lecture_final_edit_enabled"),
    )
    lecture_final_edit_input_token_budget: int = Field(
        default=50_000,
        validation_alias=AliasChoices("LECTURE_FINAL_EDIT_INPUT_TOKEN_BUDGET", "lecture_final_edit_input_token_budget"),
    )
    document_index_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("DOCUMENT_INDEX_CACHE_ENABLED", "document_index_cache_enabled"),
    )
    document_index_cache_db_path: str = Field(
        default="files_materials/_document_index_cache/index.sqlite3",
        validation_alias=AliasChoices("DOCUMENT_INDEX_CACHE_DB_PATH", "document_index_cache_db_path"),
    )
    document_index_cache_lock_timeout_seconds: float = Field(
        default=900.0,
        validation_alias=AliasChoices(
            "DOCUMENT_INDEX_CACHE_LOCK_TIMEOUT_SECONDS",
            "document_index_cache_lock_timeout_seconds",
        ),
    )
    document_index_cache_stale_lock_seconds: float = Field(
        default=21600.0,
        validation_alias=AliasChoices(
            "DOCUMENT_INDEX_CACHE_STALE_LOCK_SECONDS",
            "document_index_cache_stale_lock_seconds",
        ),
    )
    document_index_cache_busy_timeout_ms: int = Field(
        default=30_000,
        validation_alias=AliasChoices("DOCUMENT_INDEX_CACHE_BUSY_TIMEOUT_MS", "document_index_cache_busy_timeout_ms"),
    )
    quiz_answer_dialog_summary_message_limit: int = Field(
        default=30,
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_SUMMARY_MESSAGE_LIMIT",
            "quiz_answer_dialog_summary_message_limit",
        ),
    )
    quiz_answer_dialog_top_k_chunks: int = Field(
        default=6,
        validation_alias=AliasChoices("QUIZ_ANSWER_DIALOG_TOP_K_CHUNKS", "quiz_answer_dialog_top_k_chunks"),
    )
    quiz_answer_dialog_context_token_budget: int = Field(
        default=6000,
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_CONTEXT_TOKEN_BUDGET",
            "quiz_answer_dialog_context_token_budget",
        ),
    )
    quiz_answer_dialog_max_response_tokens: int = Field(
        default=700,
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_MAX_RESPONSE_TOKENS",
            "quiz_answer_dialog_max_response_tokens",
        ),
    )
    quiz_answer_dialog_lock_timeout_seconds: float = Field(
        default=300.0,
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_LOCK_TIMEOUT_SECONDS",
            "quiz_answer_dialog_lock_timeout_seconds",
        ),
    )
    quiz_answer_dialog_summary_cache_db_path: str = Field(
        default="files_materials/_quiz_answer_dialog_cache/summaries.sqlite3",
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_SUMMARY_CACHE_DB_PATH",
            "quiz_answer_dialog_summary_cache_db_path",
        ),
    )
    quiz_answer_dialog_summary_cache_busy_timeout_ms: int = Field(
        default=30_000,
        validation_alias=AliasChoices(
            "QUIZ_ANSWER_DIALOG_SUMMARY_CACHE_BUSY_TIMEOUT_MS",
            "quiz_answer_dialog_summary_cache_busy_timeout_ms",
        ),
    )
    pdf_font_regular: str = Field(
        default="",
        validation_alias=AliasChoices("PDF_FONT_REGULAR", "pdf_font_regular"),
    )
    pdf_font_bold: str = Field(
        default="",
        validation_alias=AliasChoices("PDF_FONT_BOLD", "pdf_font_bold"),
    )
    pdf_font_italic: str = Field(
        default="",
        validation_alias=AliasChoices("PDF_FONT_ITALIC", "pdf_font_italic"),
    )
    pdf_font_bold_italic: str = Field(
        default="",
        validation_alias=AliasChoices("PDF_FONT_BOLD_ITALIC", "pdf_font_bold_italic"),
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
