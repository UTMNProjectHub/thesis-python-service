from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FAQItem:
    """
    Одна пара вопрос-ответ в FAQ.
    """
    question: str
    answer: str
    category: Optional[str] = None  # Опциональная категория для группировки


@dataclass
class FAQ:
    """
    Полный FAQ-документ.
    title — заголовок (например, "FAQ по теме X")
    items — список пар Q&A
    """
    title: str
    items: List[FAQItem]
