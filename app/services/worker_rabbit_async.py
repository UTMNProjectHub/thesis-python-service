import asyncio

from app.services.rabbitmq_async import get_async_rabbit_client
from app.services.s3_client import get_s3_client
from app.services.postgres import get_postgres_client
from app.services.tasks_processor import TaskProcessor


async def main():
    rabbit = get_async_rabbit_client()
    s3 = get_s3_client()
    db = get_postgres_client()

    processor = TaskProcessor(rabbit=rabbit, s3=s3, db=db)

    await rabbit.connect()

    async def quiz_handler(payload: dict):
        await processor.handle_quiz_gen(payload)

    async def summary_handler(payload: dict):
        await processor.handle_summary_gen(payload)

    await rabbit.listen(rabbit.queue_quiz_gen, quiz_handler)
    await rabbit.listen(rabbit.queue_summary_gen, summary_handler)

    print("[Worker] Async Rabbit listeners started")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
