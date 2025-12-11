# app/services/rabbitmq.py
from __future__ import annotations

import json
import threading
from typing import Callable

import pika
from urllib.parse import urlparse

from app.api.core.config import settings
import asyncio


class RabbitClient:
    """
    Клиент RabbitMQ для проекта Андрей Диплом.

    Позволяет:
    - отправлять задачи (QuizGen, SummaryGen)
    - слушать очереди результатов (QuizGenComplete, SummaryGenComplete)
    """

    def __init__(
            self,
            host: str,
            port: int,
            username: str,
            password: str,
            vhost: str = "/",
    ):
        credentials = pika.PlainCredentials(username, password)

        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            virtual_host=vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=None,
        )

        # Названия очередей
        self.queue_quiz_gen = "quiz.generation.request"
        self.queue_summary_gen = "summary.generation.request"
        self.queue_quiz_gen_complete = "quiz.generation.complete"
        self.queue_summary_gen_complete = "summary.generation.complete"

    # ======================================================================
    # 1. ПУБЛИКАЦИЯ СООБЩЕНИЙ
    # ======================================================================
    def publish(self, queue: str, payload: dict) -> None:
        """
        Универсальная публикация JSON-сообщения в очередь.
        """
        connection = pika.BlockingConnection(self.connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)

        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(payload).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2  # persistent
            ),
        )

        connection.close()

    # ------------------------------------------------------------------
    # QuizGen → помещаем задачу на генерацию квиза
    # ------------------------------------------------------------------
    def enqueue_quiz_generation(self, payload: dict):
        """
        payload (QuizGen):
        {
            quizId: uuid,
            files: uuid[],
            difficulty: "easy"|"medium"|"hard",
            question_count: number,
            question_types: string[],
            additional_requirements: text
        }
        """
        self.publish(self.queue_quiz_gen, payload)

    # ------------------------------------------------------------------
    # SummaryGen → помещаем задачу генерации конспекта
    # ------------------------------------------------------------------
    def enqueue_summary_generation(self, payload: dict):
        """
        payload (SummaryGen):
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            files: uuid[],
            additional_requirements: text
        }
        """
        self.publish(self.queue_summary_gen, payload)

    # ======================================================================
    # 2. СЛУШАТЕЛЬ ОЧЕРЕДЕЙ
    # ======================================================================
    def listen(self, queue: str, callback: Callable[[dict], None]):
        """
        Универсальный слушатель очереди.
        Работает в отдельном потоке, чтобы не блокировать FastAPI
        (или основной поток приложения).
        """

        def _thread_worker():
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)

            def _on_message(ch, method, properties, body):
                try:
                    data = json.loads(body.decode("utf-8"))

                    result = callback(data)
                    if asyncio.iscoroutine(result):
                        loop = asyncio.get_event_loop()
                        loop.create_task(result)

                except Exception as e:
                    print(f"[Rabbit] Ошибка обработки сообщения: {e}")
                finally:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=queue, on_message_callback=_on_message)

            print(f"[Rabbit] Listening on queue: {queue}")
            channel.start_consuming()

        thread = threading.Thread(target=_thread_worker, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Подписка на QuizGenComplete
    # ------------------------------------------------------------------
    def on_quiz_generation_complete(self, callback: Callable[[dict], None]):
        """
        callback принимает payload (QuizGenComplete):
        {
            quizId: uuid,
            status: "SUCCESS"|"FAILED",
            error: string
        }
        """
        self.listen(self.queue_quiz_gen_complete, callback)

    # ------------------------------------------------------------------
    # Подписка на SummaryGenComplete
    # ------------------------------------------------------------------
    def on_summary_generation_complete(self, callback: Callable[[dict], None]):
        """
        callback принимает payload (SummaryGenComplete):
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            status: "SUCCESS"|"FAILED",
            error: string
        }
        """
        self.listen(self.queue_summary_gen_complete, callback)


def get_rabbit_client() -> RabbitClient:
    """
    Фабрика RabbitClient, берёт строку подключения из settings.amqp_url
    (переменная окружения / .env: AMQP_URL=amqp://user:pass@host:5672/vhost).
    """
    # было: settings.AMQP_URL
    url = urlparse(settings.amqp_url)

    username = url.username or "guest"
    password = url.password or "guest"
    host = url.hostname or "localhost"
    port = url.port or 5672
    vhost = url.path[1:] if url.path.startswith("/") else "/"

    return RabbitClient(
        host=host,
        port=port,
        username=username,
        password=password,
        vhost=vhost,
    )