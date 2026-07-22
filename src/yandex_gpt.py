"""Обёртка над YandexGPT API (REST) для генерации ответа по найденным чанкам.

Ключи и настройки читаются ИСКЛЮЧИТЕЛЬНО из переменных окружения. Сервисного
аккаунта Yandex Cloud пока нет — код готов к работе, но при отсутствии ключей
приложение должно явно упасть на старте (см. validate_config), а не в рантайме
на первом сообщении пользователя.

Docs: https://yandex.cloud/ru/docs/foundation-models/text-generation/api-ref/
Endpoint: POST https://llm.api.cloud.yandex.net/foundationModels/v1/completion
"""

from __future__ import annotations

import logging
import os
from typing import List

import httpx

from .pdf_processor import Chunk

logger = logging.getLogger(__name__)

YANDEX_COMPLETION_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)

# Имя модели настраивается через переменную окружения. Дефолт — lite-версия
# (дешевле). Формат URI модели: gpt://<folder_id>/<model>/latest.
# Актуальные имена моделей см. в документации Yandex Cloud Foundation Models.
DEFAULT_MODEL = "yandexgpt-lite"

# Жёсткая системная инструкция: отвечать только по контексту, не выдумывать.
SYSTEM_PROMPT = (
    "Ты — ассистент для сотрудников службы доставки БК компании "
    "«ТД Нефтьмагистраль». Отвечай на вопрос СТРОГО и ТОЛЬКО на основе "
    "приведённых ниже фрагментов инструкции. Запрещено использовать общие знания "
    "и что-либо додумывать.\n"
    "Правила:\n"
    "1. Отвечай на часть вопроса ТОЛЬКО если фрагмент прямо описывает именно эту "
    "ситуацию и даёт на неё ответ. Простого совпадения слов недостаточно: если "
    "фрагмент лишь упоминает те же слова (например, «напиток»), но не отвечает на "
    "суть вопроса, — считай, что ответа на эту часть нет. Не притягивай соседние "
    "правила, которые про другое.\n"
    "2. Если вопрос состоит из нескольких частей, ответь на те части, ответ на "
    "которые реально ЕСТЬ во фрагментах, а по остальным коротко напиши, что в "
    "инструкции этого нет. Не отказывайся отвечать на весь вопрос из-за одной "
    "части, которой нет.\n"
    "3. Если НИ НА ОДНУ часть вопроса во фрагментах нет прямого ответа — напиши "
    "ровно: «В инструкции нет точного ответа на этот вопрос».\n"
    "4. Не придумывай номера пунктов, цифры, сроки, адреса, телефоны и контакты, "
    "которых нет в тексте фрагментов.\n"
    "5. Отвечай кратко, по-русски, простым языком.\n"
    "6. Не ссылайся на страницы, разделы или номера фрагментов — давай только "
    "суть ответа."
)


class YandexConfigError(RuntimeError):
    """Не задана обязательная переменная окружения для YandexGPT."""


def validate_config() -> None:
    """Проверяет наличие ключей при старте. Кидает YandexConfigError с понятным текстом."""
    missing = []
    if not os.environ.get("YANDEX_API_KEY"):
        missing.append("YANDEX_API_KEY")
    if not os.environ.get("YANDEX_FOLDER_ID"):
        missing.append("YANDEX_FOLDER_ID")
    if missing:
        raise YandexConfigError(
            "Не заданы переменные окружения для YandexGPT: "
            + ", ".join(missing)
            + ". Задайте их в окружении (локально — в .env, на Railway — в Variables)."
        )


def _model_uri() -> str:
    folder_id = os.environ["YANDEX_FOLDER_ID"]
    model = os.environ.get("YANDEX_GPT_MODEL", DEFAULT_MODEL)
    # Разрешаем задавать как короткое имя (yandexgpt-lite), так и полный gpt://...
    if model.startswith("gpt://"):
        return model
    return f"gpt://{folder_id}/{model}/latest"


def _build_context(chunks: List[Chunk]) -> str:
    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[Фрагмент {i}]\n{ch.text}")
    return "\n\n".join(parts)


def build_user_message(question: str, chunks: List[Chunk]) -> str:
    """Собирает пользовательскую часть промпта: контекст-фрагменты + вопрос."""
    context = _build_context(chunks)
    return (
        "Фрагменты инструкции (используй только их):\n"
        f"{context}\n\n"
        f"Вопрос сотрудника: {question}"
    )


# Общий HTTP-клиент с пулом соединений: под нагрузкой не создаём новое
# TLS-соединение на каждый запрос. Лимиты пула — потолок одновременных
# соединений к YandexGPT (дополнительно к семафору в bot.py).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
    return _client


async def generate_answer(question: str, chunks: List[Chunk], timeout: float = 30.0) -> str:
    """Вызывает YandexGPT и возвращает текст ответа.

    Предполагается, что validate_config() уже вызван при старте, ключи есть.
    """
    api_key = os.environ["YANDEX_API_KEY"]
    folder_id = os.environ["YANDEX_FOLDER_ID"]

    payload = {
        "modelUri": _model_uri(),
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,  # ниже температура → меньше «фантазий»
            "maxTokens": "1000",
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": build_user_message(question, chunks)},
        ],
    }

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }

    client = _get_client()
    resp = await client.post(
        YANDEX_COMPLETION_URL, json=payload, headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()

    # Формат ответа: result.alternatives[0].message.text
    try:
        alternatives = data["result"]["alternatives"]
        text = alternatives[0]["message"]["text"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("Неожиданный формат ответа YandexGPT: %s", data)
        raise RuntimeError("Неожиданный формат ответа YandexGPT") from exc

    return text
