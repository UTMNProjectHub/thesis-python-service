from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from app.api.core.config import settings
from app.documents.chunking import chunk_document_pages
from app.documents.docx_reader import load_docx_document
from app.documents.pdf_reader import load_pdf_document
from app.services.proxy_client import proxy_completion
from .config import FAQGenerationConfig, normalize_faq_detail_level
from .models import FAQ, FAQItem

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _json_candidates(raw: str) -> List[str]:
    cleaned = _strip_code_fence(raw)
    candidates = [cleaned]

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(cleaned[object_start:object_end + 1])

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(cleaned[array_start:array_end + 1])

    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _items_from_payload(data) -> list:
    if isinstance(data, dict):
        for key in ("items", "questions", "faqs", "faq"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    if isinstance(data, list):
        return data
    return []


def parse_faq_from_json(raw: str) -> List[FAQItem]:
    items = []
    for candidate in _json_candidates(raw):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = _items_from_payload(data)
        if items:
            break

    faq_items: List[FAQItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("q") or item.get("prompt") or "").strip()
        answer = str(item.get("answer") or item.get("a") or item.get("response") or "").strip()
        if not question or not answer:
            continue
        category = item.get("category") or item.get("section") or item.get("topic")
        faq_items.append(FAQItem(question=question, answer=answer, category=category))
    return faq_items


def build_faq_prompt(cfg: FAQGenerationConfig) -> tuple[str, str]:
    detail_level = normalize_faq_detail_level(cfg.detail_level)
    detail_hint = {
        "low": "Short answers, no extra detail, but preserve the meaning.",
        "medium": "Medium detail: explain key ideas and add useful examples.",
        "high": "Detailed answers with context, explanations, and educational examples.",
    }[detail_level]

    system_prompt = """
You are an expert at creating FAQ materials for educational lecture notes.
Write all questions and answers in Russian.
Use only the provided lecture text.
Return strict JSON only, without Markdown or explanations.
""".strip()

    parts = [
        f"Generate {cfg.num_questions} FAQ questions from the lecture text below.",
        "Questions must cover key concepts, definitions, practical conclusions, and common misunderstandings.",
        "Group questions by meaningful categories.",
        f"Detail level: {detail_hint}",
    ]

    additional_requirements = (cfg.additional_requirements or "").strip()
    if additional_requirements:
        parts.extend(["", "Additional requirements:", additional_requirements])

    if cfg.batch_count > 1:
        parts.extend(
            [
                "",
                f"FAQ batch: {cfg.batch_index} of {cfg.batch_count}.",
                "Generate only this batch. Avoid repeating questions already generated in previous batches.",
            ]
        )
        if cfg.avoid_questions:
            parts.append("Previously generated questions to avoid:")
            parts.extend(f"- {question}" for question in cfg.avoid_questions[:30])

    parts.extend(
        [
            "",
            "JSON format:",
            """
{
  "items": [
    {
      "question": "Question?",
      "answer": "Answer.",
      "category": "Category"
    }
  ]
}
""".strip(),
            "",
            "Lecture text:",
        ]
    )

    return system_prompt, "\n".join(parts)


def _faq_max_tokens(num_questions: int) -> int:
    per_question = max(settings.faq_completion_tokens_per_question, 80)
    max_tokens = max(1200, num_questions * per_question)
    return min(max_tokens, settings.faq_max_completion_tokens)


async def _generate_faq_batch(
        text: str,
        title: str,
        cfg: FAQGenerationConfig,
) -> FAQ:
    system_prompt, user_prompt = build_faq_prompt(cfg)
    full_user_prompt = f"{user_prompt}\n{text}"
    max_tokens = _faq_max_tokens(cfg.num_questions)

    raw_answer, _ = await proxy_completion(
        text="",
        user_prompt=full_user_prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=max_tokens,
    )

    items = parse_faq_from_json(raw_answer)
    if not items:
        preview = raw_answer[:1000].replace("\n", "\\n")
        logger.warning(
            "FAQ generation returned empty parse title=%s questions=%d raw_len=%d raw_preview=%s",
            title,
            cfg.num_questions,
            len(raw_answer),
            preview,
        )
    else:
        logger.info(
            "FAQ batch generated title=%s batch=%d/%d requested=%d parsed=%d max_tokens=%d",
            title,
            cfg.batch_index,
            cfg.batch_count,
            cfg.num_questions,
            len(items),
            max_tokens,
        )

    return FAQ(title=title, items=items)


def _dedupe_question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().rstrip("?!.").lower())


async def generate_faq_from_text(
        text: str,
        title: str,
        cfg: Optional[FAQGenerationConfig] = None,
) -> FAQ:
    if cfg is None:
        cfg = FAQGenerationConfig()

    total_questions = max(int(cfg.num_questions), 1)
    batch_size = max(int(settings.faq_batch_size), 1)
    if total_questions <= batch_size:
        return await _generate_faq_batch(text=text, title=title, cfg=cfg)

    batch_count = math.ceil(total_questions / batch_size)
    items: List[FAQItem] = []
    seen_questions: set[str] = set()

    for batch_index in range(1, batch_count + 1):
        remaining = total_questions - len(items)
        if remaining <= 0:
            break

        batch_cfg = replace(
            cfg,
            num_questions=min(batch_size, remaining),
            batch_index=batch_index,
            batch_count=batch_count,
            avoid_questions=tuple(item.question for item in items[-30:]),
        )
        batch = await _generate_faq_batch(text=text, title=title, cfg=batch_cfg)
        for item in batch.items:
            key = _dedupe_question_key(item.question)
            if not key or key in seen_questions:
                continue
            seen_questions.add(key)
            items.append(item)
            if len(items) >= total_questions:
                break

    logger.info("FAQ generation finished title=%s requested=%d produced=%d", title, total_questions, len(items))
    return FAQ(title=title, items=items)


async def generate_faq_from_file(
        file_path: str,
        title: Optional[str] = None,
        cfg: Optional[FAQGenerationConfig] = None,
) -> FAQ:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if title is None:
        title = p.stem

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        doc, pages_text = load_pdf_document(file_path, doc_id=title)
        chunks = chunk_document_pages(doc, pages_text)
        text = "\n\n".join(chunk.text for chunk in chunks)
    elif suffix == ".docx":
        _, pages_text = load_docx_document(file_path, doc_id=title)
        text = "\n\n".join(pages_text)
    elif suffix in {".md", ".markdown", ".txt"}:
        text = p.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return await generate_faq_from_text(text=text, title=title, cfg=cfg)


def format_faq_as_markdown(faq: FAQ) -> str:
    lines = [f"# {faq.title}", ""]

    categories: dict[str, List[FAQItem]] = {}
    for item in faq.items:
        category = str(item.category or "General").strip() or "General"
        categories.setdefault(category, []).append(item)

    for category, items in categories.items():
        lines.extend([f"## {category}", ""])
        for index, item in enumerate(items, start=1):
            question = item.question.rstrip("?")
            lines.extend(
                [
                    f"### {index}. {question}?",
                    "",
                    item.answer.strip(),
                    "",
                ]
            )

    return "\n".join(lines).strip()
