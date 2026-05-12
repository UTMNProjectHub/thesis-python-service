from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from app.curriculum.models import Curriculum

PathLike = Union[str, Path]


def load_curriculum_from_json(path: PathLike) -> Curriculum:
    """
    Загружает Curriculum из JSON-файла.

    Формат JSON:
    {
      "course_id": "...",
      "course_name": "...",
      "description": "...",
      "topics": [
        {
          "id": "intro_pgvector",
          "title": "Введение в PGVector",
          "description": "Основы векторного поиска...",
          "difficulty": "easy",
          "keywords": ["pgvector", "эмбеддинги", "семантический поиск"],
          "duration_min": 90,
          "source_docs": ["pgvector_book"],
          "order": 1
        },
        ...
      ]
    }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Curriculum JSON not found: {p}")

    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw)
    return Curriculum.model_validate(data)


def save_curriculum_to_json(curriculum: Curriculum, path: PathLike) -> None:
    """
    Сохраняет Curriculum в JSON с красивым форматированием.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = curriculum.model_dump(mode="python")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
