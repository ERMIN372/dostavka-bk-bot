"""aiogram v3 бот (long polling). Обработчики сообщений + RAG-логика на сообщение.

Память диалога НЕ ведётся: каждое сообщение обрабатывается независимо.
Логи вопросов пользователей НЕ пишутся ни в файл, ни в БД. В stdout идут только
служебные логи (старт, факт получения сообщения, ошибки).
"""

from __future__ import annotations

import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from .indexer import Index
from .retriever import SIMILARITY_THRESHOLD, retrieve
from . import yandex_gpt

logger = logging.getLogger(__name__)

# Фиксированный ответ, когда в базе знаний ничего релевантного не нашлось.
NO_ANSWER_MESSAGE = (
    "Не нашёл ответа на этот вопрос в базе знаний. "
    "Уточните формулировку или обратитесь к руководителю точки."
)

WELCOME_MESSAGE = (
    "Здравствуйте! Я бот-помощник службы доставки БК «ТД Нефтьмагистраль».\n"
    "Задайте вопрос по инструкции доставки обычным текстом — я найду ответ в базе знаний."
)


def create_dispatcher(index: Index, embedding_model) -> Dispatcher:
    """Создаёт Dispatcher с внедрёнными индексом и моделью эмбеддингов."""
    dp = Dispatcher()

    @dp.message(Command("start", "help"))
    async def on_start(message: Message) -> None:
        await message.answer(WELCOME_MESSAGE)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        question = (message.text or "").strip()
        logger.info("Получено сообщение (len=%d)", len(question))  # без содержимого вопроса
        if not question:
            return

        try:
            result = retrieve(index, embedding_model, question, threshold=SIMILARITY_THRESHOLD)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка на этапе retrieval")
            await message.answer("Произошла внутренняя ошибка при поиске. Попробуйте позже.")
            return

        # Порог не пройден → не тратим деньги на YandexGPT, отвечаем фиксированно.
        if not result.passed_threshold:
            await message.answer(NO_ANSWER_MESSAGE)
            return

        try:
            answer = await yandex_gpt.generate_answer(question, result.chunks)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка вызова YandexGPT")
            await message.answer("Сервис ответов временно недоступен. Попробуйте позже.")
            return

        await message.answer(answer or NO_ANSWER_MESSAGE)

    @dp.message()
    async def on_other(message: Message) -> None:
        # Не текст (фото, стикеры и т.п.) — вежливо просим текст.
        await message.answer("Пожалуйста, задайте вопрос текстом.")

    return dp


async def run_bot(index: Index, embedding_model) -> None:
    """Запускает long polling. Токен читается из TELEGRAM_BOT_TOKEN."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Не задан TELEGRAM_BOT_TOKEN. Задайте переменную окружения "
            "(локально — в .env, на Railway — в Variables)."
        )

    bot = Bot(token=token)
    dp = create_dispatcher(index, embedding_model)

    logger.info("Запуск long polling ...")
    # drop_pending_updates: не отвечаем на сообщения, накопившиеся, пока бот лежал.
    await dp.start_polling(bot, drop_pending_updates=True)
