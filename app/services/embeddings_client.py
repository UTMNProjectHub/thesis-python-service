# app/services/embeddings_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openai import OpenAI, AsyncOpenAI

from app.api.core.config import settings


@dataclass
class OpenAIEmbeddingsClient:
    """
    Клиент для получения эмбеддингов через ProxyAPI (OpenAI-совместимый API).

    Поддерживает:
      • синхронный режим (по умолчанию)
      • асинхронный режим (async_mode=True)
    """
    model: str | None = None
    async_mode: bool = False  # по умолчанию — синхронный

    def __post_init__(self) -> None:
        # Всегда создаём синхронный клиент
        self.sync_client = OpenAI(
            api_key=settings.proxyapi_key,
            base_url=settings.base_url,
        )

        # Асинхронный клиент — только если явно включён
        if self.async_mode:
            self.async_client = AsyncOpenAI(
                api_key=settings.proxyapi_key,
                base_url=settings.base_url,
            )

        if self.model is None:
            self.model = settings.embedding_model

    # --------------------- СИНХРОННЫЙ МЕТОД ---------------------
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Синхронная версия — использует self.sync_client"""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.sync_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    # --------------------- АСИНХРОННЫЙ МЕТОД ---------------------
    async def embed_texts_async(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Асинхронная версия — требует async_mode=True"""
        if not self.async_mode:
            raise RuntimeError(
                "Асинхронный клиент не инициализирован. "
                "Создайте экземпляр с async_mode=True"
            )

        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await self.async_client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# --------------------- ГЛОБАЛЬНЫЕ ЭКЗЕМПЛЯРЫ ---------------------

# Синхронный клиент — для всех обычных скриптов и тестов
embeddings_client = OpenAIEmbeddingsClient(async_mode=False)

# Асинхронный клиент — для асинхронных модулей (rag, explainer, checker)
embeddings_client_async = OpenAIEmbeddingsClient(async_mode=True)


# --------------------- УДОБНЫЕ ФУНКЦИИ ---------------------

def get_embeddings_sync(texts: List[str]) -> List[List[float]]:
    """Синхронная обёртка — используйте в main_explainer_and_quiz_generate.py"""
    return embeddings_client.embed_texts(texts)


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Асинхронная обёртка — используйте в асинхронных частях проекта"""
    return await embeddings_client_async.embed_texts_async(texts)