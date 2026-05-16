from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable
from urllib.parse import urlparse

import pika

from app.api.core.config import settings

logger = logging.getLogger(__name__)


class RabbitClient:
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

        self.queue_quiz_gen = "quiz.generation.request"
        self.queue_summary_gen = "summary.generation.request"
        self.queue_faq_gen = "faq.generation.request"
        self.queue_quiz_answer_dialog_message = "quiz.answer.dialog.message"
        self.queue_quiz_gen_complete = "quiz.generation.complete"
        self.queue_summary_gen_complete = "summary.generation.complete"
        self.queue_faq_gen_complete = "faq.generation.complete"
        self.queue_quiz_answer_dialog_response = "quiz.answer.dialog.response"

    def publish(self, queue: str, payload: dict) -> None:
        """
        Универсальная публикация JSON-сообщения в очередь.
        """
        logger.info("Publishing RabbitMQ message queue=%s payload_keys=%s", queue, sorted(payload.keys()))
        logger.info("Opening RabbitMQ blocking connection for publish queue=%s", queue)
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
        logger.info("RabbitMQ publish completed queue=%s", queue)

    def enqueue_quiz_generation(self, payload: dict):
        """
        payload (QuizGen):
        {
            quizId: uuid,
            userId: uuid,
            files: uuid[],
            summaryId: number,
            difficulty: "easy"|"medium"|"hard",
            question_count: number,
            question_types: string[],
            additional_requirements: text
        }
        """
        self.publish(self.queue_quiz_gen, payload)

    def enqueue_summary_generation(self, payload: dict):
        """
        payload (SummaryGen):
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            userId: uuid,
            files: uuid[],
            additional_requirements: text
        }
        """
        self.publish(self.queue_summary_gen, payload)

    def enqueue_faq_generation(self, payload: dict):
        """
        payload (FAQGen):
        {
            summaryId: int,
            faqId: uuid,
            userId: uuid,
            title: str,
            numQuestions: int,
            detailLevel: "easy"|"medium"|"hard",
            additionalRequirements: text
        }
        """
        self.publish(self.queue_faq_gen, payload)

    def listen(self, queue: str, callback: Callable[[dict], None]):
        """
        Универсальный слушатель очереди.
        Работает в отдельном потоке, чтобы не блокировать FastAPI
        """

        def _thread_worker():
            logger.info("Opening RabbitMQ blocking connection for listener queue=%s", queue)
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)

            def _on_message(ch, method, properties, body):
                try:
                    data = json.loads(body.decode("utf-8"))
                    logger.info(
                        "RabbitMQ message received queue=%s payload_keys=%s",
                        queue,
                        sorted(data.keys()),
                    )

                    result = callback(data)
                    if asyncio.iscoroutine(result):
                        asyncio.run(result)
                    logger.info("RabbitMQ message processed queue=%s", queue)

                except Exception as e:
                    logger.exception("RabbitMQ handler failed queue=%s error=%s", queue, e)
                finally:
                    ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=queue, on_message_callback=_on_message)

            logger.info("Listening RabbitMQ queue=%s", queue)
            channel.start_consuming()

        thread = threading.Thread(target=_thread_worker, daemon=True)
        thread.start()

    def on_quiz_generation_complete(self, callback: Callable[[dict], None]):
        """
        callback принимает payload (QuizGenComplete):
        {
            quizId: uuid,
            userId: uuid,
            status: "SUCCESS"|"FAILED",
            error: string
        }
        """
        self.listen(self.queue_quiz_gen_complete, callback)

    def on_summary_generation_complete(self, callback: Callable[[dict], None]):
        """
        callback принимает payload (SummaryGenComplete):
        {
            summaryId: uuid,
            subjectId: number,
            themeId: number,
            userId: uuid,
            status: "SUCCESS"|"FAILED",
            error: string
        }
        """
        self.listen(self.queue_summary_gen_complete, callback)

    def on_faq_generation_complete(self, callback: Callable[[dict], None]):
        """
        callback receives payload (FAQGenComplete):
        {
            faqId: uuid,
            userId: uuid,
            status: "SUCCESS"|"FAILED",
            error: string
        }
        """
        self.listen(self.queue_faq_gen_complete, callback)


def get_rabbit_client() -> RabbitClient:
    url = urlparse(settings.amqp_url)

    username = url.username or "guest"
    password = url.password or "guest"
    host = url.hostname or "localhost"
    port = url.port or 5672
    vhost = (url.path[1:] if url.path.startswith("/") else url.path) or "/"

    return RabbitClient(
        host=host,
        port=port,
        username=username,
        password=password,
        vhost=vhost,
    )
