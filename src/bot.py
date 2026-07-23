"""aiogram v3 бот (long polling). Обработчики сообщений + RAG-логика на сообщение.

Память диалога НЕ ведётся: каждое сообщение обрабатывается независимо.
Логи вопросов пользователей НЕ пишутся ни в файл, ни в БД. В stdout идут только
служебные логи (старт, факт получения сообщения, ошибки).

Защита от флуда/перегрузки (см. throttling.py):
  * per-user rate limit — от спама одним пользователем;
  * глобальный лимит вызовов YandexGPT в минуту — предохранитель бюджета;
  * CPU-bound эмбеддинг вынесен из event loop (to_thread) и ограничен семафором,
    иначе один тяжёлый запрос замораживает бот для ВСЕХ пользователей;
  * ограничение длины вопроса.
"""

from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from .indexer import Index
from .retriever import SIMILARITY_THRESHOLD, retrieve
from .throttling import GlobalRateLimiter, UserRateLimiter
from . import yandex_gpt

logger = logging.getLogger(__name__)

# Telegram-ID администраторов, которым доступна команда /debug (диагностика
# ретривера прямо в чате). Задаётся через переменную окружения DEBUG_ADMIN_IDS
# (список ID через запятую). Пусто → команда недоступна никому и невидима.
DEBUG_ADMIN_IDS = {
    int(x)
    for x in os.environ.get("DEBUG_ADMIN_IDS", "").replace(" ", "").split(",")
    if x.isdigit()
}

# Временное сообщение-«раздумье»: показывается сразу после вопроса и
# самоудаляется, когда готов настоящий ответ.
THINKING_MESSAGE = "🔎 Ищу ответ в базе знаний…"

# Фиксированный ответ, когда в базе знаний ничего релевантного не нашлось.
NO_ANSWER_MESSAGE = (
    "Не нашёл ответа на этот вопрос в базе знаний. "
    "Уточните формулировку или обратитесь к руководителю точки."
)

WELCOME_MESSAGE = (
    "Здравствуйте! Я бот-помощник службы доставки БК «ТД Нефтьмагистраль».\n"
    "Задайте вопрос по инструкции доставки обычным текстом — я найду ответ в базе знаний.\n\n"
    "/shablon — шаблон сообщения о проблемном заказе для чата поддержки."
)

# Шаблон обращения по проблемному заказу. Введён по итогам анализа чата
# поддержки: 58% живых обращений — свободная переписка вокруг заказов без
# единой структуры, из-за чего диспетчеры тратят время на уточняющие вопросы.
TEMPLATE_MESSAGE = (
    "Шаблон сообщения о проблемном заказе — скопируйте, заполните и отправьте "
    "в чат поддержки:\n\n"
    "Точка: М__\n"
    "Номер заказа: \n"
    "Источник: МП / Яндекс.Еда / Купер / касса\n"
    "Телефон гостя: \n"
    "Проблема (одной фразой): \n"
    "Что уже сделали: \n"
    "Какая помощь нужна: \n\n"
    "Для предзаказа дополнительно укажите дату/время выдачи и статус предоплаты."
)

RATE_LIMIT_MESSAGE = (
    "Слишком много запросов подряд. Подождите немного и задайте вопрос снова."
)

OVERLOAD_MESSAGE = (
    "Сейчас слишком много обращений к боту. Попробуйте через пару минут."
)

# Максимальная длина вопроса. Реальные вопросы сотрудников короткие; всё длиннее —
# либо вставленный текст, либо флуд. Экономит CPU (эмбеддинг) и токены YandexGPT.
MAX_QUESTION_LEN = 500

# Параллелизм тяжёлых операций:
#   эмбеддинг — CPU-bound, на Railway обычно 1-2 vCPU, больше 2 потоков смысла нет;
#   YandexGPT — сетевые вызовы, ограничиваем, чтобы не копить сотни висящих задач.
EMBED_CONCURRENCY = 2
GPT_CONCURRENCY = 4


def create_dispatcher(index: Index, embedding_model) -> Dispatcher:
    """Создаёт Dispatcher с внедрёнными индексом и моделью эмбеддингов."""
    dp = Dispatcher()

    user_limiter = UserRateLimiter()
    gpt_limiter = GlobalRateLimiter()
    embed_sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    gpt_sem = asyncio.Semaphore(GPT_CONCURRENCY)

    @dp.message(Command("start", "help"))
    async def on_start(message: Message) -> None:
        await message.answer(WELCOME_MESSAGE)

    @dp.message(Command("shablon"))
    async def on_template(message: Message) -> None:
        await message.answer(TEMPLATE_MESSAGE)

    @dp.message(Command("debug"))
    async def on_debug(message: Message, command: CommandObject) -> None:
        """Диагностика ретривера (только для админов из DEBUG_ADMIN_IDS).

        Показывает top-k чанков со score, порог и решение «идти ли в GPT» — чтобы
        понять, почему на вопрос пришёл (или не пришёл) ответ. Обычным
        пользователям команда не отвечает вовсе (как будто её нет).
        """
        user_id = message.from_user.id if message.from_user else 0
        if user_id not in DEBUG_ADMIN_IDS:
            return
        query = (command.args or "").strip()
        if not query:
            await message.answer(
                f"Ваш Telegram-ID: {user_id}\nИспользование: /debug <вопрос>"
            )
            return

        result = await asyncio.to_thread(
            retrieve, index, embedding_model, query, threshold=SIMILARITY_THRESHOLD,
        )
        lines = [
            f"max_cos={result.max_score:.3f} | порог={SIMILARITY_THRESHOLD} "
            f"| к GPT: {'ДА' if result.passed_threshold else 'НЕТ'}",
            "гибрид = 0.7·cos + 0.3·bm25",
            "",
        ]
        for i, (c, s, cs, ls) in enumerate(
            zip(result.chunks, result.scores, result.cos_scores, result.lex_scores), 1
        ):
            head = " ".join(c.text.split())[:110]
            lines.append(
                f"{i}. {s:.3f} (cos {cs:.3f} | lex {ls:.2f}) "
                f"(стр.{c.page_start}-{c.page_end}) {head}"
            )
        await message.answer("\n".join(lines)[:4000])

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        question = (message.text or "").strip()
        logger.info("Получено сообщение (len=%d)", len(question))  # без содержимого вопроса
        if not question:
            return

        # --- Анти-флуд: per-user лимит. Срабатывает ДО любых тяжёлых операций. ---
        user_id = message.from_user.id if message.from_user else 0
        allowed, warn = user_limiter.check(user_id)
        if not allowed:
            if warn:
                await message.answer(RATE_LIMIT_MESSAGE)
            # Повторные нарушения в том же окне игнорируем молча,
            # чтобы не отвечать флудом на флуд.
            return

        if len(question) > MAX_QUESTION_LEN:
            await message.answer(
                f"Вопрос слишком длинный (максимум {MAX_QUESTION_LEN} символов). "
                "Сформулируйте короче."
            )
            return

        # Мгновенная обратная связь: сообщение-«раздумье». Отправляем сразу, чтобы
        # пользователь видел, что бот принял вопрос и работает (поиск + генерация
        # YandexGPT занимают несколько секунд), а не решил, что бот завис.
        # Удалим его, как только будет готов настоящий ответ.
        thinking = await message.answer(THINKING_MESSAGE)

        async def reply(text: str) -> None:
            """Убрать «раздумье» и прислать финальный ответ."""
            # Удаление в try: сообщение могло быть уже удалено/устарело — не критично.
            try:
                await thinking.delete()
            except Exception:  # noqa: BLE001
                pass
            await message.answer(text)

        try:
            # Эмбеддинг — синхронный CPU-bound вызов. Уводим его в поток, чтобы не
            # блокировать event loop (иначе на время encode бот виснет для всех),
            # и ограничиваем параллелизм семафором, чтобы наплыв запросов
            # не устроил CPU-давку на 1-2 ядрах контейнера.
            async with embed_sem:
                result = await asyncio.to_thread(
                    retrieve, index, embedding_model, question,
                    threshold=SIMILARITY_THRESHOLD,
                )
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка на этапе retrieval")
            await reply("Произошла внутренняя ошибка при поиске. Попробуйте позже.")
            return

        # Порог не пройден → не тратим деньги на YandexGPT, отвечаем фиксированно.
        if not result.passed_threshold:
            await reply(NO_ANSWER_MESSAGE)
            return

        # --- Предохранитель бюджета: глобальный лимит вызовов YandexGPT в минуту. ---
        if not gpt_limiter.allow():
            logger.warning("Глобальный лимит YandexGPT исчерпан — отвечаем отказом.")
            await reply(OVERLOAD_MESSAGE)
            return

        try:
            async with gpt_sem:
                answer = await yandex_gpt.generate_answer(question, result.chunks)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка вызова YandexGPT")
            await reply("Сервис ответов временно недоступен. Попробуйте позже.")
            return

        await reply(answer or NO_ANSWER_MESSAGE)

    @dp.message()
    async def on_other(message: Message) -> None:
        # Не текст (фото, стикеры и т.п.) — вежливо просим текст.
        # Тот же rate limit: флуд стикерами не должен получать ответ на каждый.
        user_id = message.from_user.id if message.from_user else 0
        allowed, _ = user_limiter.check(user_id)
        if allowed:
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
