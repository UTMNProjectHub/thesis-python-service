import asyncio
from pathlib import Path
from app.faq.generator import generate_faq_from_file, format_faq_as_markdown
from app.faq.config import FAQGenerationConfig


async def main():
    FILE_PATH = "субд1.md"
    TITLE = "основные определения системы управления базами данных"
    NUM_QUESTIONS = 15
    DETAIL_LEVEL = "low"

    path = Path(FILE_PATH)
    if not path.exists():
        print(f"Файл не найден: {path}")
        return

    title = TITLE or path.stem.replace("_", " ").replace("-", " ").title()
    title = f"Часто задаваемые вопросы (FAQ): {title}"

    print(f"Генерация FAQ из: {path.name}")
    print(f"Вопросов: {NUM_QUESTIONS} | Детальность: {DETAIL_LEVEL}\n")

    cfg = FAQGenerationConfig(
        language="ru",
        num_questions=NUM_QUESTIONS,
        detail_level=DETAIL_LEVEL,
    )

    faq = await generate_faq_from_file(
        file_path=str(path),
        title=title,
        cfg=cfg,
    )

    markdown = format_faq_as_markdown(faq)

    output_file = path.parent / f"FAQ_{path.stem}.md"
    output_file.write_text(markdown, encoding="utf-8")

    print(f"Готово! FAQ сохранён → {output_file.name}")
    print(f"Создано вопросов: {len(faq.items)}")


if __name__ == "__main__":
    asyncio.run(main())