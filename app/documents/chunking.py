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

        # сбрасываем буферы
        current_buffer = []
        current_tokens = 0

    # основной цикл по страницам
    for page_idx, page_text in enumerate(pages_text):
        page_num = page_idx + 1  # 1-based
        paragraphs = split_into_paragraphs(page_text)

        if not paragraphs:
            # пустая страница — просто продолжаем
            continue

        # если в буфере ничего нет, стартовая страница нового чанка = текущая
        if not current_buffer:
            current_page_start = page_num

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            # если абзац сам по себе больше max_tokens — сохраняем как отдельный чанк
            # чтобы не зависнуть в бесконечном объединении
            if para_tokens >= max_tokens:
                # сначала сбрасываем текущий чанк, если что-то накопилось
                flush_chunk()
                current_page_start = page_num
                current_page_end = page_num
                chunk_id = f"{doc.id}-chunk-{chunk_index}"
                chunks.append(
                    DocumentChunk(
                        doc_id=doc.id,
                        chunk_id=chunk_id,
                        text=para,
                        page_start=page_num,
                        page_end=page_num,
                        heading_path=None,
                    )
                )
                chunk_index += 1
                continue

            # если добавление абзаца переполнит чанк — сначала сбрасываем текущий
            if current_tokens + para_tokens > max_tokens and current_buffer:
                current_page_end = page_num
                flush_chunk()
                current_page_start = page_num  # новый чанк начинается с текущей страницы

            # добавляем абзац в текущий чанк
            current_buffer.append(para)
            current_tokens += para_tokens
            current_page_end = page_num

    # после прохода по всем страницам сбрасываем всё, что осталось
    flush_chunk()

    return chunks
