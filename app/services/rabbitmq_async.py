from __future__ import annotations

import json
import logging
from typing import Callable, Awaitable

import aio_pika
from aio_pika import IncomingMessage

from app.api.core.config import settings

logger = logging.getLogger(__name__)


class AsyncRabbitClient:
    """
    Полностью асинхронный клиент RabbitMQ для проекта Андрей Диплом.
    """

    def __init__(self, amqp_url: str):
        self.amqp_url = amqp_url
        self.connection: aio_pika.RobustConnection | None = None
        self.channel: aio_pika.RobustChannel | None = None

        self.queue_quiz_gen = "quiz.generation.request"
        self.queue_summary_gen = "summary.generation.request"
        self.queue_faq_gen = "faq.generation.request"
        self.queue_quiz_answer_dialog_message = "quiz.answer.dialog.message"
        self.queue_quiz_gen_complete = "quiz.generation.complete"
        self.queue_summary_gen_complete = "summary.generation.complete"
        self.queue_faq_gen_complete = "faq.generation.complete"
        self.queue_quiz_answer_dialog_response = "quiz.answer.dialog.response"

    async def connect(self):
        """
        Создаём асинхронное соединение и канал.
        """
        if self.connection:
            logger.info("RabbitMQ connection already initialized")
            return

        logger.info("Connecting to RabbitMQ heartbeat=%s", settings.amqp_heartbeat)
        self.connection = await aio_pika.connect_robust(
            self.amqp_url,
            client_properties={"connection_name": "AndreyDiplomaWorker"},
            heartbeat=settings.amqp_heartbeat,
        )
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)
        logger.info("Connected to RabbitMQ with prefetch_count=1")

    async def publish(self, queue_name: str, payload: dict):
        """
        Публикация сообщений в очередь.
        """
        assert self.channel, "RabbitMQ channel not initialized"

        queue = await self.channel.declare_queue(queue_name, durable=True)
        logger.info("Publishing RabbitMQ message queue=%s payload_keys=%s", queue.name, sorted(payload.keys()))

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue_name,
        )

    async def listen(self, queue_name: str, handler: Callable[[dict], Awaitable[None]]):
        """
        Асинхронный слушатель очереди задач.
        """

        assert self.channel, "RabbitMQ channel not initialized"

        queue = await self.channel.declare_queue(queue_name, durable=True)
        logger.info("Listening RabbitMQ queue=%s", queue_name)

        async def _on_message(message: IncomingMessage):
            async with message.process():
                try:
                    data = json.loads(message.body.decode("utf-8"))
                    logger.info(
                        "RabbitMQ message received queue=%s payload_keys=%s",
                        queue_name,
                        sorted(data.keys()),
                    )
                    await handler(data)
                    logger.info("RabbitMQ message processed queue=%s", queue_name)
                except Exception as exc:
                    logger.exception("RabbitMQ handler failed queue=%s error=%s", queue_name, exc)

        await queue.consume(_on_message, no_ack=False)


def get_async_rabbit_client() -> AsyncRabbitClient:
    url = settings.amqp_url
    return AsyncRabbitClient(url)
