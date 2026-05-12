from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """
    Абстракция учебного документа (книга, РПД, МОК и т.д.).

    id    — произвольный идентификатор (можно генерировать из имени файла).
    path  — путь к файлу на диске.
    title — человеко-читаемое имя (обычно — имя файла без расширения).
    pages — количество страниц (для PDF; для DOCX можно считать как 1 "логическую").
    """
    id: str
    path: Path
    title: Optional[str] = None
    pages: int = 0


@dataclass
class DocumentChunk:
    """
    Чанк текста документа — минимальная единица для RAG/поиска.

    doc_id      — идентификатор документа (Document.id).
    chunk_id    — уникальный id чанка внутри документа.
    text        — текст чанка.
    page_start  — номер первой страницы, попавшей в чанк (1-based).
    page_end    — номер последней страницы, попавшей в чанк (1-based).
    heading_path — опциональный "путь" заголовков (Глава 1 → 1.2 → ...),
                   пригодится позже, когда будем парсить структуру книги.
    """
    doc_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    heading_path: Optional[List[str]] = None
