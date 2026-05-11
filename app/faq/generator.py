from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.documents.chunking import chunk_document_pages
from app.documents.docx_reader import load_docx_document
from app.documents.pdf_reader import load_pdf_document
from app.services.proxy_client import proxy_completion
from .config import FAQGenerationConfig
from .models import FAQ, FAQItem


def _strip_code_fence(text: str) -> str:
    """
    Убираем ```json ... ``` вокруг ответа LLM.
    """
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_faq_from_json(raw: str) -> List[FAQItem]:
    """
    Парсит JSON от LLM в список FAQItem.
    Ожидает: {"items": [{"question": "...", "answer": "...", "category": "..."}, ...]}
    """
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        return []

    faq_items: List[FAQItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        cat = item.get("category")
        faq_items.append(FAQItem(question=q, answer=a, category=cat))
    return faq_items


def build_faq_prompt(cfg: FAQGenerationConfig) -> str:
    """
    Системный и user-промпт для генерации FAQ.
    """
    system_prompt = """
Ты — эксперт по составлению FAQ для учебных конспектов.
Твоя задача — по тексту конспекта создать детальный, понятный FAQ.
Всегда используй РУССКИЙ язык.
Возвращай результат СТРОГО в формате JSON без пояснений.
""".strip()

    detail_hint = {
        "low": "Краткие ответы, без примеров.",
        "medium": "Средняя детальность, с примерами где нужно.",
        "high": "Подробные ответы, с объяснениями и примерами.",
    }[cfg.detail_level]

    user_prompt = f"""
По тексту конспекта ниже сгенерируй {cfg.num_questions} вопросов для FAQ.
Вопросы должны быть релевантными, охватывать ключевые аспекты.
Группируй по категориям (например, "Общие", "Технические").
Уровень детальности: {detail_hint}.

Формат JSON:
{{
  "items": [
    {{
      "question": "Вопрос?",
      "answer": "Ответ.",
      "category": "Категория"  // опционально
    }},
    ...
  ]
}}

Текст конспекта:
""".strip()
    return system_prompt, user_prompt


async def generate_faq_from_text(
        text: str,
        title: str,
        cfg: Optional[FAQGenerationConfig] = None,
) -> FAQ:
    """
    Генерирует FAQ из чистого текста.
    """
    if cfg is None:
        cfg = FAQGenerationConfig()

    system_prompt, user_prompt = build_faq_prompt(cfg)
    full_user_prompt = f"{user_prompt}\n{text}"

    raw_answer, _ = await proxy_completion(
        text="",
        user_prompt=full_user_prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=2000,
    )

    items = parse_faq_from_json(raw_answer)
    return FAQ(title=title, items=items)


async def generate_faq_from_file(
        file_path: str,
        title: Optional[str] = None,
        cfg: Optional[FAQGenerationConfig] = None,
) -> FAQ:
    """
    Генерирует FAQ из файла (MD, PDF, DOCX).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if title is None:
        title = p.stem

    if p.suffix.lower() == ".pdf":
        doc, pages_text = load_pdf_document(file_path, doc_id=title)
        chunks = chunk_document_pages(doc, pages_text)
        text = "\n\n".join([c.text for c in chunks])  # Склеиваем чанки
    elif p.suffix.lower() == ".docx":
        doc, pages_text = load_docx_document(file_path, doc_id=title)
        text = "\n\n".join(pages_text)
    elif p.suffix.lower() in {".md", ".markdown"}:
        text = p.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return await generate_faq_from_text(text=text, title=title, cfg=cfg)


def format_faq_as_markdown(faq: FAQ) -> str:
    """
    Форматирует FAQ в красивый Markdown.
    """
    lines = [f"# {faq.title}"]
    lines.append("")

    # Группировка по категориям
    categories: dict[str, List[FAQItem]] = {}
    for item in faq.items:
        cat = item.category or "Общие вопросы"
        categories.setdefault(cat, []).append(item)

    for cat, items in categories.items():
        lines.append(f"## {cat}")
        lines.append("")
        for item in items:
            lines.append(f"**{item.question}**")
            lines.append(item.answer)
            lines.append("")

    return "\n".join(lines).strip()
