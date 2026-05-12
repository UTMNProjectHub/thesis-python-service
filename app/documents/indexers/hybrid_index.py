from __future__ import annotations

from typing import List, Tuple, Dict

from app.documents.indexers.base import BaseRetriever
from app.documents.indexers.embeddings_index import EmbeddingsRetriever
from app.documents.indexers.tfidf_index import TfidfRetriever
from app.documents.models import DocumentChunk


class HybridRetriever(BaseRetriever):
    """
    Гибридный ретривер: TF-IDF + эмбеддинги.

    Идея:
      - TF-IDF хорошо ловит точные совпадения терминов.
      - Эмбеддинги хорошо ловят семантику, синонимы и перефразы.
      - Мы берём оба скора и считаем взвешенную сумму:
          score = alpha * emb_score + (1 - alpha) * tfidf_score

    На выходе — тот же интерфейс BaseRetriever.
    """

    def __init__(
            self,
            tfidf_retriever: TfidfRetriever | None = None,
            embeddings_retriever: EmbeddingsRetriever | None = None,
            alpha: float = 0.7,
    ) -> None:
        """
        alpha ∈ [0; 1] — вес семантического скора:
          - 0.7 → сильнее доверяем эмбеддингам,
          - 0.3 → сильнее доверяем TF-IDF.
        """
        self.tfidf = tfidf_retriever or TfidfRetriever(stop_words=None)
        self.emb = embeddings_retriever or EmbeddingsRetriever()
        self.alpha = alpha

    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Индексируем один и тот же набор чанков двумя способами.
        """
        self.tfidf.index(chunks)
        self.emb.index(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Делаем два поиска (TF-IDF и эмбеддинги), объединяем результаты.
        """
        base_k = max(top_k * 3, 10)

        tfidf_results = self.tfidf.search(query, top_k=base_k)
        emb_results = self.emb.search(query, top_k=base_k)

        if not tfidf_results and not emb_results:
            return []
        if not tfidf_results:
            return emb_results[:top_k]
        if not emb_results:
            return tfidf_results[:top_k]

        combined_scores: Dict[str, float] = {}
        chunks_map: Dict[str, DocumentChunk] = {}

        for chunk, score in tfidf_results:
            cid = chunk.chunk_id
            chunks_map[cid] = chunk
            combined_scores.setdefault(cid, 0.0)
            combined_scores[cid] += (1.0 - self.alpha) * score

        for chunk, score in emb_results:
            cid = chunk.chunk_id
            chunks_map[cid] = chunk
            combined_scores.setdefault(cid, 0.0)
            combined_scores[cid] += self.alpha * score

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float]] = []
        for cid, score in ranked[:top_k]:
            chunk = chunks_map[cid]
            results.append((chunk, float(score)))

        return results
