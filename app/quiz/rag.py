# app/quiz/rag.py
from __future__ import annotations

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Правильные импорты из исправленного embeddings_client.py
from app.services.embeddings_client import (
    get_embeddings_sync,      # Синхронная функция
    get_embeddings,           # Асинхронная функция
)


class SimpleVectorStore:
    """
    Простое векторное хранилище для RAG.
    Поддерживает как синхронный, так и асинхронный режим.
    """
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] | None = None

    # ====================== АСИНХРОННЫЕ МЕТОДЫ ======================
    async def add_document(self, text: str, chunk_size: int = 500) -> None:
        """Асинхронно добавляет документ (чанкит + эмбеддинги)"""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 100)]
        self.chunks = chunks

        embeddings = await get_embeddings(chunks)  # ← асинхронная функция
        self.embeddings = embeddings  # теперь возвращает List[List[float]], а не dict

    async def search(self, query: str, top_k: int = 4) -> List[str]:
        """Асинхронный поиск по косинусному сходству"""
        if not self.embeddings:
            return []

        query_emb = (await get_embeddings([query]))[0]  # ← уже List[float]
        query_vec = np.array(query_emb).reshape(1, -1)
        chunk_vecs = np.array(self.embeddings)

        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]

    # ====================== СИНХРОННЫЕ МЕТОДЫ ======================
    def add_document_sync(self, text: str, chunk_size: int = 500) -> None:
        """Синхронно добавляет документ (для тестовых скриптов)"""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 100)]
        self.chunks = chunks

        embeddings = get_embeddings_sync(chunks)  # ← синхронная функция
        self.embeddings = embeddings  # List[List[float]]

    def search_sync(self, query: str, top_k: int = 4) -> List[str]:
        """Синхронный поиск — без await"""
        if not self.embeddings:
            return []

        query_emb = get_embeddings_sync([query])[0]  # List[float]
        query_vec = np.array(query_emb).reshape(1, -1)
        chunk_vecs = np.array(self.embeddings)

        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]