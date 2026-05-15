import asyncio
import logging

from app.logging_config import configure_logging
from app.services.postgres import get_postgres_client
from app.services.rabbitmq_async import get_async_rabbit_client
from app.services.s3_client import get_s3_client
from app.services.tasks_processor import TaskProcessor

logger = logging.getLogger(__name__)


async def main():
    configure_logging()
    logger.info("Starting Rabbit worker")

    logger.info("Initializing Rabbit/S3/Postgres clients")
    rabbit = get_async_rabbit_client()
    s3 = get_s3_client()
    db = get_postgres_client()
    logger.info("Clients initialized")

    processor = TaskProcessor(rabbit=rabbit, s3=s3, db=db)

    await rabbit.connect()
    logger.info("Rabbit worker connected to broker")

    async def quiz_handler(payload: dict):
        logger.info("Received quiz generation task quizId=%s", payload.get("quizId"))
        await processor.handle_quiz_gen(payload)

    async def summary_handler(payload: dict):
        logger.info("Received summary generation task summaryId=%s", payload.get("summaryId"))
        await processor.handle_summary_gen(payload)

    async def faq_handler(payload: dict):
        logger.info("Received FAQ generation task faqId=%s", payload.get("faqId"))
        await processor.handle_faq_gen(payload)

    await rabbit.listen(rabbit.queue_quiz_gen, quiz_handler)
    await rabbit.listen(rabbit.queue_summary_gen, summary_handler)
    await rabbit.listen(rabbit.queue_faq_gen, faq_handler)

    logger.info(
        "Rabbit listeners started queues=%s,%s,%s",
        rabbit.queue_quiz_gen,
        rabbit.queue_summary_gen,
        rabbit.queue_faq_gen,
    )

    while True:
        await asyncio.sleep(3600)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
