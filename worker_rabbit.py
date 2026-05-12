# worker_rabbit.py
import asyncio
import time

from app.services.rabbitmq import get_rabbit_client
from app.services.s3_client import get_s3_client
from app.services.postgres import get_postgres_client
from app.services.tasks_processor import TaskProcessor


def main():
    rabbit = get_rabbit_client()
    s3 = get_s3_client()
    db = get_postgres_client()
    processor = TaskProcessor(rabbit=rabbit, s3=s3, db=db)

    # Обработчик задач QuizGen
    def quiz_handler(payload: dict):
        asyncio.run(processor.handle_quiz_gen(payload))

    # Обработчик задач SummaryGen
    def summary_handler(payload: dict):
        asyncio.run(processor.handle_summary_gen(payload))

    # Слушаем очереди задач (в отдельных потоках, см. реализацию listen)
    rabbit.listen(rabbit.queue_quiz_gen, quiz_handler)
    rabbit.listen(rabbit.queue_summary_gen, summary_handler)

    print("[Worker] Rabbit listeners started")

    # Просто не даём процессу упасть, пока живут daemon-потоки
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
