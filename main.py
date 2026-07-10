"""Точка входа.

Порядок работы:
  1. Настроить логирование в stdout (для просмотра в Railway).
  2. Проверить конфигурацию YandexGPT (ключи из окружения) — если чего-то нет,
     явно упасть на старте с понятным сообщением, а не в рантайме.
  3. Загрузить локальную модель эмбеддингов один раз.
  4. Построить или загрузить из кеша индекс базы знаний.
  5. Запустить aiogram long polling.

ПРИМЕЧАНИЕ про Railway: контейнер при каждом передеплое стартует с нуля. Если
папка knowledge_base/.cache не лежит на персистентном volume — индекс
пересчитается заново при старте. Для 84-страничного PDF это ожидаемо и занимает
секунды/десятки секунд на CPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from src import yandex_gpt
from src.bot import run_bot
from src.indexer import build_or_load_index, load_embedding_model

# Локально удобно держать ключи в .env. На Railway переменные задаются в Variables,
# поэтому .env там просто отсутствует — это нормально. Источник истины — os.environ.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Папка базы знаний (можно переопределить переменной окружения).
KNOWLEDGE_BASE_DIR = os.environ.get(
    "KNOWLEDGE_BASE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base"),
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # aiohttp/httpx бывают слишком болтливы на INFO — приглушаем.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)


async def _amain() -> None:
    logger = logging.getLogger("main")

    # 2. Валидация ключей YandexGPT и Telegram — падаем на старте, если чего-то нет.
    yandex_gpt.validate_config()
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        raise RuntimeError(
            "Не задан TELEGRAM_BOT_TOKEN. Задайте переменную окружения "
            "(локально — в .env, на Railway — в Variables)."
        )

    # 3. Модель эмбеддингов — грузим один раз и переиспользуем в индексе и рантайме.
    embedding_model = load_embedding_model()

    # 4. Индекс: пересчитываем только при изменении/добавлении PDF, иначе — из кеша.
    logger.info("Инициализация индекса базы знаний ...")
    index = build_or_load_index(KNOWLEDGE_BASE_DIR, model=embedding_model)
    logger.info("Индекс готов: %d чанков.", len(index.chunks))

    # 5. Запуск бота.
    await run_bot(index, embedding_model)


def main() -> None:
    _configure_logging()
    logger = logging.getLogger("main")
    try:
        asyncio.run(_amain())
    except (yandex_gpt.YandexConfigError, RuntimeError) as exc:
        # Понятное сообщение об ошибке конфигурации/старта — в stdout, exit code 1.
        logger.error("Старт невозможен: %s", exc)
        sys.exit(1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Необработанная ошибка при старте")
        sys.exit(1)


if __name__ == "__main__":
    main()
