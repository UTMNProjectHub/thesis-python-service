import logging

from fastapi import FastAPI

from app.logging_config import configure_logging
from app.v1.router import router as v1_router

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Quizy Python API")
app.include_router(v1_router)
logger.info("FastAPI app initialized")


@app.get("/")
async def root():
    logger.info("Health check requested")
    return {"ok": True}
