from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.documents.chunking import chunk_document_pages
from app.documents.docx_reader import load_docx_document
from app.documents.pdf_reader import load_pdf_document
from app.services.proxy_client import proxy_completion
from .config import FAQGenerationConfig, normalize_faq_detail_level
from .models import FAQ, FAQItem


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


def parse_faq_from_json(raw: str) -> List[FAQItem]:
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
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        category = item.get("category")
        faq_items.append(FAQItem(question=question, answer=answer, category=category))
    return faq_items


def build_faq_prompt(cfg: FAQGenerationConfig) -> tuple[str, str]:
    detail_level = normalize_faq_detail_level(cfg.detail_level)
    detail_hint = {
        "low": "Краткие ответы без лишних деталей, но с сохранением смысла.",
        "medium": "Средняя детализация: объясняй ключевые идеи и добавляй примеры там, где они полезны.",
        "high": "Подробные ответы с пояснениями, контекстом и учебными примерами.",
    }[detail_level]

    system_prompt = """
Ты - эксперт по составлению FAQ для учебных конспектов.
Составляй вопросы и ответы на русском языке.
Опирайся только на переданный текст лекции или конспекта.
Возвращай результат строго в JSON без пояснений и без Markdown.
""".strip()

    parts = [
        f"По тексту ниже сгенерируй {cfg.num_questions} вопросов для FAQ.",
        "Вопросы должны покрывать ключевые понятия, определения, практические выводы и типичные ошибки понимания.",
        "Группируй вопросы по осмысленным категориям.",
        f"Уровень детализации: {detail_hint}",
    ]

    additional_requirements = (cfg.additional_requirements or "").strip()
    if additional_requirements:
        parts.extend(["", "Дополнительные требования:", additional_requirements])

    parts.extend(
        [
            "",
            "Формат JSON:",
            """
{
  "items": [
    {
      "question": "Вопрос?",
      "answer": "Ответ.",
      "category": "Категория"
    }
  ]
}
""".strip(),
            "",
            "Текст лекции:",
        ]
    )

    return system_prompt, "\n".join(parts)


async def generate_faq_from_text(
        text: str,
        title: str,
        cfg: Optional[FAQGenerationConfig] = None,
) -> FAQ:
    if cfg is None:
        cfg = FAQGenerationConfig()

    system_prompt, user_prompt = build_faq_prompt(cfg)
    full_user_prompt = f"{user_prompt}\n{text}"

    raw_answer, _ = await proxy_completion(
        text="",
        user_prompt=full_user_prompt,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=2500,
    )

    items = parse_faq_from_json(raw_answer)
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
        category = str(item.category or "Общие вопросы").strip() or "Общие вопросы"
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
