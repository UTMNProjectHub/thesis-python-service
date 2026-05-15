from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.curriculum.models import LectureTopic
from app.documents.chunking import estimate_tokens
from app.documents.indexers.base import BaseRetriever
from app.documents.models import Document, DocumentChunk
from app.services.proxy_client import proxy_completion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentProfile:
    doc_id: str
    title: str
    pages: int
    chunk_count: int
    summary: str
    coverage: str


def _chunk_sort_key(item: tuple[DocumentChunk, float]) -> tuple[str, int, int, str]:
    chunk, _ = item
    return chunk.doc_id, chunk.page_start, chunk.page_end, chunk.chunk_id


def _fits_budget(
        selected: list[tuple[DocumentChunk, float]],
        candidate: DocumentChunk,
        token_budget: int | None,
) -> bool:
    if token_budget is None:
        return True
    used = sum(estimate_tokens(chunk.text) for chunk, _ in selected)
    return used + estimate_tokens(candidate.text) <= token_budget


def _is_near_duplicate(
        selected: list[tuple[DocumentChunk, float]],
        candidate: DocumentChunk,
        page_window: int = 1,
) -> bool:
    for chunk, _ in selected:
        if chunk.chunk_id == candidate.chunk_id:
            return True
        if chunk.doc_id != candidate.doc_id:
            continue
        pages_overlap = candidate.page_start <= chunk.page_end and candidate.page_end >= chunk.page_start
        pages_near = abs(candidate.page_start - chunk.page_start) <= page_window
        if pages_overlap or pages_near:
            return True
    return False


def select_balanced_chunks(
        retriever: BaseRetriever,
        query: str,
        *,
        pool_k: int,
        limit: int,
        token_budget: int | None = None,
        doc_ids: Sequence[str] | None = None,
) -> list[tuple[DocumentChunk, float]]:
    if limit <= 0:
        return []

    raw_results = retriever.search(query, top_k=max(pool_k, limit))
    if not raw_results:
        return []

    groups: dict[str, list[tuple[DocumentChunk, float]]] = {}
    seen_chunk_ids: set[str] = set()
    for chunk, score in sorted(raw_results, key=lambda item: item[1], reverse=True):
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        groups.setdefault(chunk.doc_id, []).append((chunk, score))

    ordered_doc_ids = list(doc_ids or [])
    for doc_id in sorted(groups):
        if doc_id not in ordered_doc_ids:
            ordered_doc_ids.append(doc_id)

    selected: list[tuple[DocumentChunk, float]] = []
    while len(selected) < limit:
        added = False
        for doc_id in ordered_doc_ids:
            group = groups.get(doc_id) or []
            while group:
                candidate = group.pop(0)
                chunk = candidate[0]
                if _is_near_duplicate(selected, chunk) and len(group) > 0:
                    continue
                if not _fits_budget(selected, chunk, token_budget):
                    continue
                selected.append(candidate)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break

    if len(selected) < limit:
        selected_ids = {chunk.chunk_id for chunk, _ in selected}
        leftovers = [
            item
            for group in groups.values()
            for item in group
            if item[0].chunk_id not in selected_ids
        ]
        leftovers.sort(key=lambda item: item[1], reverse=True)
        for chunk, score in leftovers:
            if len(selected) >= limit:
                break
            if not _fits_budget(selected, chunk, token_budget):
                continue
            selected.append((chunk, score))

    return sorted(selected, key=_chunk_sort_key)


def _representative_chunks(chunks: Sequence[DocumentChunk], limit: int) -> list[DocumentChunk]:
    if not chunks or limit <= 0:
        return []
    ordered = sorted(chunks, key=lambda c: (c.page_start, c.page_end, c.chunk_id))
    if len(ordered) <= limit:
        return list(ordered)

    indexes = {0, len(ordered) - 1}
    step = (len(ordered) - 1) / max(1, limit - 1)
    for i in range(limit):
        indexes.add(round(i * step))

    result = [ordered[i] for i in sorted(indexes)[:limit]]
    return result


def _profile_context(chunks: Iterable[DocumentChunk], token_budget: int) -> str:
    lines: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        chunk_tokens = estimate_tokens(chunk.text)
        if used + chunk_tokens > token_budget and lines:
            break
        used += chunk_tokens
        lines.append(
            f"### Fragment D{index} pages={chunk.page_start}-{chunk.page_end}\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(lines)


def _fallback_profile(doc: Document, chunks: Sequence[DocumentChunk], reason: str) -> DocumentProfile:
    sample = " ".join(chunk.text.strip().replace("\n", " ") for chunk in chunks[:3])
    summary = sample[:1200].strip() or "Text content was not extracted."
    coverage = f"Fallback extractive profile. Reason: {reason}"
    return DocumentProfile(
        doc_id=doc.id,
        title=doc.title or doc.id,
        pages=doc.pages,
        chunk_count=len(chunks),
        summary=summary,
        coverage=coverage,
    )


async def build_document_profiles(
        documents: Sequence[Document],
        chunks_by_doc: dict[str, Sequence[DocumentChunk]],
        *,
        topic: LectureTopic,
        additional_requirements: str = "",
        chunks_per_doc: int,
        max_tokens: int,
) -> list[DocumentProfile]:
    profiles: list[DocumentProfile] = []
    for doc in documents:
        chunks = list(chunks_by_doc.get(doc.id, []))
        if not chunks:
            profiles.append(_fallback_profile(doc, [], "document has no chunks"))
            continue

        selected = _representative_chunks(chunks, chunks_per_doc)
        context = _profile_context(selected, token_budget=max(500, chunks_per_doc * 700))

        system_prompt = (
            "Ты составляешь компактный профиль учебного материала для последующей генерации лекции. "
            "Пиши на русском языке. Выделяй только факты и темы, полезные для лекции. "
            "Не добавляй citations, URLs, source ids, file ids, page references или технические ссылки. "
            "Верни только обычный текст без Markdown-таблиц."
        )
        user_prompt = "\n".join(
            part
            for part in [
                f"Тема лекции: {topic.title}",
                f"Название документа: {doc.title or doc.id}",
                f"Страниц: {doc.pages}, фрагментов: {len(chunks)}",
                f"Дополнительные требования: {additional_requirements}" if additional_requirements else "",
                "Создай два раздела:",
                "1. Summary: 5-8 предложений о содержании документа, полезном для лекции.",
                "2. Coverage: основные темы, разделы и понятия, которые покрывает документ.",
                "",
                context,
            ]
            if part
        )

        try:
            raw, _ = await proxy_completion(
                text="",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            profile_text = raw.strip()
            if not profile_text:
                raise RuntimeError("empty document profile response")
            profiles.append(
                DocumentProfile(
                    doc_id=doc.id,
                    title=doc.title or doc.id,
                    pages=doc.pages,
                    chunk_count=len(chunks),
                    summary=profile_text,
                    coverage=profile_text,
                )
            )
            logger.info("Document profile built doc_id=%s chunks=%d", doc.id, len(chunks))
        except Exception as exc:
            logger.warning("Document profile fallback doc_id=%s error=%s", doc.id, exc)
            profiles.append(_fallback_profile(doc, chunks, str(exc)))

    return profiles
