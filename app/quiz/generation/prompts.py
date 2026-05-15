from __future__ import annotations

from typing import List, Optional

from .config import QuizGenerationConfig

GENERATION_SYSTEM_PROMPT = """
Ты — преподаватель, который составляет вопросы для проверки понимания учебного текста.

Твоя задача:
- создавать вопросы строго по предоставленному материалу;
- делать вопросы однозначными и проверяемыми;
- соблюдать запрошенные типы вопросов и количество;
- формулировать варианты ответов так, чтобы неправильные варианты были правдоподобными, но явно неверными при знании материала;
- для вопросов с несколькими правильными ответами не смешивать взаимоисключающие варианты;
- для matching-вопросов делать пары логически связанными и недвусмысленными;
- не добавлять факты, которых нет в учебном тексте;
- возвращать только валидный JSON по требуемой схеме;
- писать на русском языке.
""".strip()


def build_requirements(cfg: QuizGenerationConfig) -> str:
    """
    Текст, описывающий, какие типы и сколько вопросов нужно сгенерировать.
    Это отражение флагов/количеств из QuizGenerationConfig.
    """
    parts: List[str] = []

    if cfg.generate_true_false and cfg.num_true_false > 0:
        parts.append(
            f"- {cfg.num_true_false} вопросов формата True/False "
            f"(ответ 'true' или 'false' в JSON)."
        )

    if cfg.generate_multiple_choice and cfg.num_multiple_choice > 0:
        parts.append(
            f"- {cfg.num_multiple_choice} вопросов Multiple Choice "
            f"с 4–6 вариантами ответа. В JSON поле 'options', "
            f"a 'answer' — индекс правильного варианта (0-based)."
        )

    if cfg.generate_select_all_that_apply and cfg.num_select_all_that_apply > 0:
        parts.append(
            f"- {cfg.num_select_all_that_apply} вопросов Select All That Apply "
            f"с 4–6 вариантами. 'options' — список вариантов, "
            f"'answer' — массив индексов всех правильных вариантов."
        )

    if cfg.generate_fill_in_the_blank and cfg.num_fill_in_the_blank > 0:
        parts.append(
            f"- {cfg.num_fill_in_the_blank} вопросов Fill in the Blank. "
            f"В тексте вопроса явно помечай пропуски (например, подчёркиванием). "
            f"В JSON 'answer' — массив возможных правильных ответов (строки)."
        )

    if cfg.generate_matching and cfg.num_matching > 0:
        parts.append(
            f"- {cfg.num_matching} вопросов Matching. В JSON 'answer' — массив объектов "
            f"{{'leftOption': '...', 'rightOption': '...'}}."
        )

    if cfg.generate_short_answer and cfg.num_short_answer > 0:
        parts.append(
            f"- {cfg.num_short_answer} вопросов Short Answer "
            f"(краткий текстовый ответ до ~250 символов)."
        )

    if cfg.generate_long_answer and cfg.num_long_answer > 0:
        parts.append(
            f"- {cfg.num_long_answer} вопросов Long Answer "
            f"(развёрнутый текстовый ответ)."
        )

    return "\n".join(parts)


def build_user_prompt(cfg: QuizGenerationConfig, theme_name: Optional[str] = None) -> str:
    """
    Финальный user-prompt, который мы отправляем в LLM вместе с текстом конспекта.
    Здесь описываем формат JSON и правила для поля answer.
    """
    requirements = build_requirements(cfg)

    theme_block = ""
    if theme_name:
        theme_block = (
            f"Тема квиза: «{theme_name}».\n"
            f"Все вопросы должны относиться именно к этой теме и опираться на предоставленный учебный текст.\n\n"
        )

    format_description = """
Используй следующий JSON-формат:

{
  "questions": [
    {
      "question": "строка с текстом вопроса",
      "options": ["вариант 1", "вариант 2", "..."],   // только для типов с вариантами
      "answer": ...                                   // см. правила ниже
    },
    ...
  ]
}

Правила поля "answer" для разных типов:

- True/False:
    "answer": true  или  "answer": false

- Multiple Choice:
    "answer": целое число — индекс правильного варианта из массива "options",
    нумерация с 0.

- Select All That Apply:
    "answer": [индекс1, индекс2, ...] — массив индексов всех правильных вариантов.

- Fill in the Blank:
    "answer": ["правильный ответ 1", "правильный ответ 2", ...]

- Matching:
    "answer": [
      { "leftOption": "элемент из левого столбца", "rightOption": "соответствующий элемент справа" },
      ...
    ]

- Short Answer / Long Answer:
    "answer": "строка с правильным ответом".
""".strip()

    return f"""
Выше приведён учебный текст (конспект).
{theme_block}Сгенерируй экзаменационные вопросы по этому тексту.

Требования к количеству и типам вопросов:
{requirements}

Все формулировки вопросов и ответов делай на языке: {cfg.language}.

Качество вопросов:
- вопрос должен проверять понимание материала, а не общую эрудицию;
- правильный ответ должен однозначно следовать из предоставленного текста;
- неправильные варианты должны быть тематически близкими, но не совпадать по смыслу с правильным ответом;
- не используй внутренние идентификаторы источников, fileId, page references или source anchors.

{format_description}

ВЕРНИ ТОЛЬКО ОДИН JSON-ОБЪЕКТ, без пояснений, комментариев, Markdown-блоков и т.п.
""".strip()
