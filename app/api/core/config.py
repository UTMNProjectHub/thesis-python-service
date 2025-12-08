from __future__ import annotations
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseModel):
    proxyapi_key: str = os.getenv("PROXYAPI_KEY", "")
    base_url: str = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    model: str = os.getenv("PROXYAPI_MODEL", "gpt-5-nano")
    embedding_model: str = os.getenv("PROXYAPI_EMBEDDING_MODEL", "text-embedding-3-small")


class Settings(BaseSettings):
    # уже были
    proxyapi_key: str
    base_url: str
    model: str = "gpt-4.1-mini"

    # добавляем
    DATABASE_URL: str

    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_BUCKET: str = "thesis-materials"

    AMQP_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
