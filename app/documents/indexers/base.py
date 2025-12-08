from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from app.documents.models import DocumentChunk


class BaseRetriever(ABC):
    """
    Абстрактный интерфейс для поискового индексатора по чанкам.

    Реализации (TF-IDF, эмбеддинги и т.п.) должны:
      - уметь проиндексировать список чанков;
      - по текстовому запросу возвращать top_k чанков с оценкой релевантности.
    """

    @abstractmethod
    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Строит внутренний индекс по списку чанков.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Возвращает top_k чанков, наиболее релевантных запросу, вместе с score.
        """
        raise NotImplementedError
