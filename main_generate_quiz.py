import asyncio
from pathlib import Path

from app.quiz.generation import QuizGenerationConfig, generate_quiz_from_text
from app.quiz.saver import format_quiz_as_markdown
from app.quiz.config import SaveFormat

async def main():
    note_path = Path("субд1.md")
    note_text = note_path.read_text(encoding="utf-8")

    cfg = QuizGenerationConfig(
        language="Русский",
        generate_true_false=True,
        num_true_false=2,
        generate_multiple_choice=True,
        num_multiple_choice=3,
        generate_select_all_that_apply=True,
        num_select_all_that_apply=2,
        generate_fill_in_the_blank=True,
        num_fill_in_the_blank=2,
        generate_matching=True,
        num_matching=1,
        generate_short_answer=True,
        num_short_answer=2,
        generate_long_answer=True,
        num_long_answer=1,
    )

    questions = await generate_quiz_from_text(note_text, cfg)

    quiz_md = format_quiz_as_markdown(
        questions,
        save_format=SaveFormat.SPACED_REPETITION,
        inline_separator="::",
        multiline_separator="?",
    )

    print(quiz_md)

if __name__ == "__main__":
    asyncio.run(main())
