from __future__ import annotations

import math
import random
from typing import List, Sequence, TypeVar

T = TypeVar("T")


def shuffle_list(items: Sequence[T]) -> List[T]:
    """
    Перемешать элементы, не меняя исходную коллекцию.
    """
    result = list(items)
    random.shuffle(result)
    return result


def count_note_tokens(note_contents: str) -> int:
    return round(len(note_contents) / 4)


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    Косинусное сходство двух векторов.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)
