import asyncio

from app.documents.pdf_reader import load_pdf_document
from app.documents.chunking import chunk_document_pages
from app.documents.indexers import HybridRetriever
from app.curriculum.models import LectureTopic, DifficultyLevel
from app.lectures import build_lecture_plan_for_topic, generate_lecture_markdown


async def main():
    doc, pages = load_pdf_document("Л1_Основные определения ССУБД.pdf")
    print(f"Документ: {doc.title}, страниц: {doc.pages}")

    chunks = chunk_document_pages(doc, pages, max_tokens=700)
    print(f"Чанков: {len(chunks)}")

    retriever = HybridRetriever(alpha=0.7)
    retriever.index(chunks)

    topic = LectureTopic(
        id="intro_исбд",
        title="Информационная система базы данных",
        description="основные определения системы управления базами данных",
        difficulty=DifficultyLevel.MEDIUM,
        keywords=["база данных", "информационная система", "субд", "PostgreSQL"],
        duration_min=90,
        source_docs=[doc.id],
        order=1,
    )

    plan = await build_lecture_plan_for_topic(
        topic,
        retriever=retriever,
        top_k_chunks=8,
        min_sections=3,
        max_sections=7,
    )

    lecture_md = await generate_lecture_markdown(
        plan=plan,
        retriever=retriever,
        topic_description=topic.description,
        max_tokens_per_section=1500,
        top_k_chunks_per_section=5,
    )

    out_path = "субд1.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(lecture_md)

    print(f"\nЛекция сгенерирована и сохранена в {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
