from __future__ import annotations

from typing import List

from app.documents.models import Document, DocumentChunk


def estimate_tokens(text: str) -> int:
    """
    Примерная оценка кол-ва токенов в строке.
    Грубая эвристика: 1 токен ≈ 4 символа.
    Для диплома этого достаточно, при желании можно заменить на tiktoken.
    """
    return max(1, len(text) // 4)


def split_into_paragraphs(page_text: str) -> List[str]:
    """
    Делим текст страницы на абзацы по пустым строкам.
    Если пустых строк нет — возвращаем один абзац.
    """
    normalized = page_text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    # разбиваем по двум и более переносам строки
    raw_parts = [part.strip() for part in normalized.split("\n\n")]
    paragraphs = [p for p in raw_parts if p]
    return paragraphs or [normalized]


def split_long_text(text: str, max_tokens: int) -> List[str]:
    max_chars = max(500, max_tokens * 3)
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    pieces: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            split_at = max(
                normalized.rfind("\n", start, end),
                normalized.rfind(". ", start, end),
                normalized.rfind(" ", start, end),
            )
            if split_at > start + max_chars // 2:
                end = split_at + 1

        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    return pieces


def chunk_document_pages(
        doc: Document,
        pages_text: List[str],
        max_tokens: int = 700,
) -> List[DocumentChunk]:
    """
    Превращает текст страниц документа в список чанков.

    Логика:
      - проходим по страницам;
      - каждую страницу делим на абзацы;
      - накапливаем абзацы в текущем чанке, пока не превысили max_tokens;
      - если превышаем — начинаем новый чанк.
    """
    chunks: List[DocumentChunk] = []

    current_buffer: List[str] = []
    current_tokens = 0
    current_page_start = 1  # 1-based
    current_page_end = 1
    chunk_index = 0

    def flush_chunk():
        nonlocal chunk_index, current_buffer, current_tokens, current_page_start, current_page_end
        if not current_buffer:
            return

        chunk_text = "\n\n".join(current_buffer).strip()
        if not chunk_text:
            return

        chunk_id = f"{doc.id}-chunk-{chunk_index}"
        chunk = DocumentChunk(
            doc_id=doc.id,
            chunk_id=chunk_id,
            text=chunk_text,
            page_start=current_page_start,
            page_end=current_page_end,
            heading_path=None,
        )
        chunks.append(chunk)
        chunk_index += 1

        current_buffer = []
        current_tokens = 0

    for page_idx, page_text in enumerate(pages_text):
        page_num = page_idx + 1  # 1-based
        paragraphs = split_into_paragraphs(page_text)

        if not paragraphs:
            continue

        if not current_buffer:
            current_page_start = page_num

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            if para_tokens >= max_tokens:
                flush_chunk()
                for piece in split_long_text(para, max_tokens=max_tokens):
                    current_page_start = page_num
                    current_page_end = page_num
                    chunk_id = f"{doc.id}-chunk-{chunk_index}"
                    chunks.append(
                        DocumentChunk(
                            doc_id=doc.id,
                            chunk_id=chunk_id,
                            text=piece,
                            page_start=page_num,
                            page_end=page_num,
                            heading_path=None,
                        )
                    )
                    chunk_index += 1
                continue

            if current_tokens + para_tokens > max_tokens and current_buffer:
                current_page_end = page_num
                flush_chunk()
                current_page_start = page_num

            current_buffer.append(para)
            current_tokens += para_tokens
            current_page_end = page_num

    flush_chunk()

    return chunks
