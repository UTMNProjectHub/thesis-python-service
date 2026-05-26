from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.documents.chunking import estimate_tokens, split_long_text
from app.documents.indexers.base import BaseRetriever
from app.documents.models import DocumentChunk
from app.services.embeddings_client import OpenAIEmbeddingsClient
import requests


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

    @staticmethod
    def _split_oversized_chunks(chunks: List[DocumentChunk], max_tokens: int = 700) -> List[DocumentChunk]:
        result: List[DocumentChunk] = []
        for chunk in chunks:
            if estimate_tokens(chunk.text) < max_tokens:
                result.append(chunk)
                continue

            for index, piece in enumerate(split_long_text(chunk.text, max_tokens=max_tokens)):
                result.append(
                    DocumentChunk(
                        doc_id=chunk.doc_id,
                        chunk_id=f"{chunk.chunk_id}-emb-{index}",
                        text=piece,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        heading_path=chunk.heading_path,
                    )
                )
        return result

    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Строим эмбеддинги для всех чанков и сохраняем их в памяти.
        """
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return

        safe_chunks = self._split_oversized_chunks(chunks)
        # texts = [c.text for c in safe_chunks]  # больше не нужно

        # Вместо одного вызова backend.embed_texts – поочерёдно через кэш-сервис
        embeddings = []

        for idx, chunk in enumerate(safe_chunks):
            resp = requests.post(
                "http://localhost:3000/embedding",
                json={
                    "text": chunk.text,
                    "file_id": chunk.doc_id,           # UUID или строка
                    "chunk_index": idx,                # порядковый номер
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "model_name": "intfloat/e5-large-v2",  # модель, которую использует Python-сервис
                },
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])

        if not embeddings:
            self._chunks = []
            self._embeddings = None
            return

        self._chunks = safe_chunks
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
