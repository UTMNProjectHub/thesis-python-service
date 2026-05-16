from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np

from app.documents.chunking import estimate_tokens, split_long_text
from app.documents.index_cache import normalize_embeddings
from app.documents.indexers.base import BaseRetriever
from app.documents.models import DocumentChunk
from app.services.embeddings_client import OpenAIEmbeddingsClient

logger = logging.getLogger(__name__)


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
    def prepare_embedding_chunks(chunks: List[DocumentChunk], max_tokens: int = 700) -> List[DocumentChunk]:
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

    _split_oversized_chunks = prepare_embedding_chunks

    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Строим эмбеддинги для всех чанков и сохраняем их в памяти.
        """
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return

        safe_chunks = self._split_oversized_chunks(chunks)
        texts = [c.text for c in safe_chunks]
        embeddings = self.backend.embed_texts(texts)

        if not embeddings:
            self._chunks = []
            self._embeddings = None
            return

        self._chunks = safe_chunks
        self._embeddings = normalize_embeddings(np.asarray(embeddings, dtype="float32"))

    def index_precomputed(self, chunks: List[DocumentChunk], embeddings: np.ndarray | List[List[float]]) -> None:
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return

        array = np.asarray(embeddings, dtype="float32")
        if array.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape={array.shape}")
        if array.shape[0] != len(chunks):
            raise ValueError(
                f"Embeddings count mismatch: chunks={len(chunks)} embeddings={array.shape[0]}"
            )

        self._chunks = list(chunks)
        self._embeddings = normalize_embeddings(array)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Возвращает top_k чанков по косинусному сходству эмбеддингов.
        """
        if not self._chunks or self._embeddings is None:
            return []

        query_embeddings = self.backend.embed_texts([query])
        if not query_embeddings:
            return []

        started = time.perf_counter()
        q_vec = normalize_embeddings(np.asarray(query_embeddings[0], dtype="float32").reshape(1, -1))[0]
        sims = np.dot(self._embeddings, q_vec)
        limit = min(max(int(top_k), 0), sims.shape[0])
        if limit <= 0:
            return []

        if limit == sims.shape[0]:
            top_indexes = np.argsort(sims)[::-1]
        else:
            candidate_indexes = np.argpartition(sims, -limit)[-limit:]
            top_indexes = candidate_indexes[np.argsort(sims[candidate_indexes])[::-1]]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indexes:
            score = sims[idx]
            results.append((self._chunks[idx], float(score)))
        logger.debug(
            "Embeddings search finished chunks=%d top_k=%d seconds=%.4f",
            len(self._chunks),
            top_k,
            time.perf_counter() - started,
        )

        return results
