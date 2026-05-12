from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.api.core.config import settings

client = AsyncOpenAI(api_key=settings.proxyapi_key, base_url=settings.base_url)


async def proxy_completion(
        text: str,
        user_prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
) -> tuple[str, str]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{text}\n\n{user_prompt}".strip()})

    try:
        response = await client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens or 1500,
        )
        data = response.model_dump()
        text_out = ""

        if data.get("choices"):
            text_out = data["choices"][0].get("message", {}).get("content") or ""
        elif data.get("output"):
            content = data["output"][0].get("content", [])
            if content and isinstance(content, list):
                text_out = content[0].get("text", "")

        return text_out.strip(), data.get("model", settings.model)

    except Exception as exc:
        print(f"ProxyAPI error: {exc}")
        return "", settings.model
