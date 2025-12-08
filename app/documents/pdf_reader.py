from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyPDF2 import PdfReader

from app.documents.models import Document


def extract_pdf_pages(path: str | Path) -> List[str]:
    """
    Возвращает список строк — текст по каждой странице PDF.
    Если текст на странице не извлечён (скан), возвращаем пустую строку для этой страницы.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    reader = PdfReader(str(p))
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return pages_text


def load_pdf_document(path: str | Path, doc_id: str | None = None, title: str | None = None) -> Tuple[Document, List[str]]:
    """
    Загружает PDF как Document + список текстов страниц.

    Возвращает:
      - Document (id, path, title, pages)
      - список строк pages_text (по одной строке на страницу)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    pages_text = extract_pdf_pages(p)

    if doc_id is None:
        # Простой вариант: id = имя файла без расширения
        doc_id = p.stem

    if title is None:
        title = p.stem

    doc = Document(
        id=doc_id,
        path=p,
        title=title,
        pages=len(pages_text),
    )

    return doc, pages_text
