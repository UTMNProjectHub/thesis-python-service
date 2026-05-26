# app/services/cached_embedding_client.py
import httpx
from typing import List, Optional

async def get_cached_embedding(
    text: str,
    file_id: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    model_name: str = "intfloat/e5-large-v2",
    subject_id: Optional[str] = None,
    theme_id: Optional[str] = None,
) -> List[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:3000/embedding",
            json={
                "text": text,
                "file_id": file_id,
                "chunk_index": chunk_index,
                "page_start": page_start,
                "page_end": page_end,
                "model_name": model_name,
                "subject_id": subject_id,
                "theme_id": theme_id,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]