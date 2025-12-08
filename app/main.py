from fastapi import FastAPI
from app.v1.router import router as v1

from app.services.rabbitmq import RabbitClient
from app.api.core.config import settings

app = FastAPI(title="квизи API")
app.include_router(v1)

@app.get("/")
async def root():
    return {"ok": True}


rabbit = RabbitClient(
    host=settings.RABBIT_HOST,
    port=settings.RABBIT_PORT,
    username=settings.RABBIT_USERNAME,
    password=settings.RABBIT_PASSWORD,
    vhost=settings.RABBIT_VHOST,
)

# app/main.py

from app.services.rabbitmq import RabbitClient
from app.services.s3_client import S3Client
from app.services.postgres import PostgresClient
from app.services.tasks_processor import TaskProcessor

from app.api.core.config import settings
import asyncio

# Instances
rabbit = RabbitClient(
    host=settings.RABBIT_HOST,
    port=settings.RABBIT_PORT,
    username=settings.RABBIT_USER,
    password=settings.RABBIT_PASSWORD,
    vhost="/"
)

s3 = S3Client(
    bucket_name=settings.MINIO_BUCKET,
    endpoint=settings.MINIO_ENDPOINT.replace("http://", ""),
    access_key=settings.MINIO_ROOT_USER,
    secret_key=settings.MINIO_ROOT_PASSWORD,
    secure=False
)

pg = PostgresClient()
processor = TaskProcessor(rabbit, s3, pg)

# Run consumers
rabbit.on_quiz_generation_complete(lambda payload: print("Quiz done:", payload))
rabbit.on_summary_generation_complete(lambda payload: print("Summary done:", payload))

# Listening queues
rabbit.listen("quiz.gen", lambda payload: asyncio.create_task(processor.process_quiz_task(payload)))
rabbit.listen("summary.gen", lambda payload: asyncio.create_task(processor.process_summary_task(payload)))

def on_quiz_done(payload: dict):
    print("Quiz generation completed:", payload)

def on_summary_done(payload: dict):
    print("Summary generation completed:", payload)

# запуск слушателей в фоне
rabbit.on_quiz_generation_complete(on_quiz_done)
rabbit.on_summary_generation_complete(on_summary_done)
