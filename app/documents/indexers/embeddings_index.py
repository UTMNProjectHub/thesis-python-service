from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.documents.models import DocumentChunk
from app.documents.indexers.base import BaseRetriever
from app.services.embeddings_client import OpenAIEmbeddingsClient


class EmbeddingsRetriever(BaseRetriever):
    """
    Ретривер на базе семантических эмбеддингов.

    Сейчас использует OpenAIEmbeddingsClient (ProxyAPI).
    Позже можно подменить backend на SBERT, не меняя интерфейс.
    """

    def __init__(self, backend: OpenAIEmbeddingsClient | None = None) -> None:
        self.backend = backend or OpenAIEmbeddingsClient()
        self._chunks: List[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None  # shape: (N, dim)

    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Строим эмбеддинги для всех чанков и сохраняем их в памяти.
        """
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return

        texts = [c.text for c in chunks]
        embeddings = self.backend.embed_texts(texts)

        if not embeddings:
            self._chunks = []
            self._embeddings = None
            return

        self._chunks = chunks
        self._embeddings = np.asarray(embeddings, dtype="float32")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Возвращает top_k чанков по косинусному сходству эмбеддингов.
        """
        if not self._chunks or self._embeddings is None:
            return []

        query_embeddings = self.backend.embed_texts([query])
        if not query_embeddings:
            return []

        q_vec = np.asarray(query_embeddings[0], dtype="float32").reshape(1, -1)
        sims = cosine_similarity(q_vec, self._embeddings)[0]  # 1D: N

        indexed_scores = list(enumerate(sims))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in indexed_scores[:top_k]:
            results.append((self._chunks[idx], float(score)))

        return results
