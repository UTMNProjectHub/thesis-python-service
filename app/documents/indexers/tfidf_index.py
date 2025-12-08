from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.documents.models import DocumentChunk
from app.documents.indexers.base import BaseRetriever


class TfidfRetriever(BaseRetriever):
    """
    Простой индексатор на базе TF-IDF + косинусного сходства.

    Это хороший старт: работает быстро, не требует эмбеддингов.
    Позже можно добавить EmbeddingsRetriever с тем же интерфейсом.
    """

    def __init__(self, stop_words: str | None = None):
        """
        stop_words:
          - None        — не использовать стоп-слова;
          - "english"   — английские;
          - "russian"   — русские (если есть в sklearn).
        """
        self.stop_words = stop_words
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunks: List[DocumentChunk] = []

    def index(self, chunks: List[DocumentChunk]) -> None:
        texts = [c.text for c in chunks]
        self._vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self._matrix = self._vectorizer.fit_transform(texts)
        self._chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self._vectorizer or self._matrix is None or not self._chunks:
            return []

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]  # 1d array

        # соберём (index, score), отсортируем по score убыванию
        indexed_scores = list(enumerate(sims))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in indexed_scores[:top_k]:
            results.append((self._chunks[idx], float(score)))

        return results
