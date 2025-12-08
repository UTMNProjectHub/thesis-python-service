from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.api.core.config import settings

client = OpenAI(api_key=settings.proxyapi_key, base_url=settings.base_url)


async def proxy_completion(
        text: str,
        user_prompt: str,
        system_prompt: str = "",
        *,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
) -> tuple[str, str]:
    """
    Универсальная обёртка над chat.completions.create.

    text          — дополнительный текст (можно передавать исходный контент),
    user_prompt   — основной запрос (инструкции + данные),
    system_prompt — роль/контекст модели,
    temperature   — креативность (для JSON лучше ставить 0–0.3),
    max_tokens    — лимит токенов для ОТВЕТА (completion). Если None — используем дефолт 800.
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Склеиваем text + user_prompt, как и раньше
    messages.append({"role": "user", "content": f"{text}\n\n{user_prompt}"})

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=temperature,
            # ключевой момент: даём возможность управлять лимитом
            max_completion_tokens=max_tokens or 1500,
        )

        # Отладочный вывод
        print("\n=== RAW RESPONSE ===")
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        print("====================\n")

        data = response.model_dump()
        text_out = ""

        if "choices" in data and data["choices"]:
            text_out = data["choices"][0]["message"]["content"]
        elif "output" in data and data["output"]:
            content = data["output"][0]["content"]
            if content and isinstance(content, list) and "text" in content[0]:
                text_out = content[0]["text"]

        return text_out.strip(), data.get("model", settings.model)

    except Exception as e:
        print(f"Ошибка ProxyAPI: {e}")
        return "", settings.model
