from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.embeddings_client import (
    get_embeddings_sync,
    get_embeddings,
)


class SimpleVectorStore:
    """
    Простое векторное хранилище для RAG.
    Поддерживает как синхронный, так и асинхронный режим.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] | None = None

    async def add_document(self, text: str, chunk_size: int = 500) -> None:
        """Асинхронно добавляет документ (чанкит + эмбеддинги)"""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 100)]
        self.chunks = chunks

        embeddings = await get_embeddings(chunks)
        self.embeddings = embeddings

    async def search(self, query: str, top_k: int = 4) -> List[str]:
        """Асинхронный поиск по косинусному сходству"""
        if not self.embeddings:
            return []

        query_emb = (await get_embeddings([query]))[0]
        query_vec = np.array(query_emb).reshape(1, -1)
        chunk_vecs = np.array(self.embeddings)

        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]

    def add_document_sync(self, text: str, chunk_size: int = 500) -> None:
        """Синхронно добавляет документ (для тестовых скриптов)"""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 100)]
        self.chunks = chunks

        embeddings = get_embeddings_sync(chunks)
        self.embeddings = embeddings

    def search_sync(self, query: str, top_k: int = 4) -> List[str]:
        """Синхронный поиск — без await"""
        if not self.embeddings:
            return []

        query_emb = get_embeddings_sync([query])[0]
        query_vec = np.array(query_emb).reshape(1, -1)
        chunk_vecs = np.array(self.embeddings)

        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]
