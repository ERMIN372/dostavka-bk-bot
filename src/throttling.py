"""Защита бота от флуда и перегрузки. Всё в памяти, без БД (по требованиям проекта).

Три уровня защиты:
  1. Пер-пользовательский rate limit (sliding window) — от флуда одним человеком.
  2. Глобальный лимит вызовов YandexGPT в минуту — предохранитель бюджета,
     если флудят с многих аккаунтов сразу (бот открыт всем).
  3. Ограничение параллелизма тяжёлых операций (семафоры в bot.py) — чтобы
     наплыв запросов не съел CPU/память контейнера.

Память ограничена: старые записи о пользователях вычищаются, словарь не растёт
бесконечно даже при флуде с тысяч уникальных аккаунтов.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict

# --- Пер-пользовательские лимиты ---
USER_MIN_INTERVAL = 2.0     # сек между сообщениями одного пользователя
USER_WINDOW = 60.0          # окно, сек
USER_MAX_PER_WINDOW = 8     # максимум сообщений пользователя за окно

# --- Глобальный предохранитель бюджета YandexGPT ---
GPT_WINDOW = 60.0           # окно, сек
GPT_MAX_PER_WINDOW = 30     # максимум вызовов YandexGPT в минуту на всех

# --- Ограничение памяти трекинга ---
MAX_TRACKED_USERS = 10_000  # при превышении вычищаем самых давних


class UserRateLimiter:
    """Sliding-window лимитер per-user. Не потокобезопасен, но и не нужно:
    aiogram-обработчики крутятся в одном event loop."""

    def __init__(self) -> None:
        self._events: Dict[int, Deque[float]] = {}
        # Флаг «предупреждение уже отправлено в этом окне» — чтобы не отвечать
        # на каждый заспамленный месседж (иначе сами становимся источником флуда).
        self._warned_at: Dict[int, float] = {}

    def check(self, user_id: int) -> tuple[bool, bool]:
        """Возвращает (разрешено, стоит_ли_предупредить).

        allowed=False, warn=True  → превышение, отправить одно предупреждение;
        allowed=False, warn=False → превышение, молча игнорировать.
        """
        now = time.monotonic()
        q = self._events.get(user_id)
        if q is None:
            q = deque()
            self._events[user_id] = q
            self._maybe_evict(now)

        # Чистим события старше окна.
        while q and now - q[0] > USER_WINDOW:
            q.popleft()

        too_fast = bool(q) and (now - q[-1]) < USER_MIN_INTERVAL
        too_many = len(q) >= USER_MAX_PER_WINDOW

        if too_fast or too_many:
            warned = self._warned_at.get(user_id, 0.0)
            if now - warned > USER_WINDOW:
                self._warned_at[user_id] = now
                return False, True
            return False, False

        q.append(now)
        return True, False

    def _maybe_evict(self, now: float) -> None:
        """Не даём словарю расти бесконечно при флуде уникальными user_id."""
        if len(self._events) <= MAX_TRACKED_USERS:
            return
        # Выкидываем пользователей без активности в последнем окне.
        stale = [
            uid for uid, q in self._events.items()
            if not q or now - q[-1] > USER_WINDOW
        ]
        for uid in stale:
            self._events.pop(uid, None)
            self._warned_at.pop(uid, None)


class GlobalRateLimiter:
    """Общий лимит на дорогие вызовы (YandexGPT) в скользящем окне."""

    def __init__(self, max_per_window: int = GPT_MAX_PER_WINDOW, window: float = GPT_WINDOW) -> None:
        self._events: Deque[float] = deque()
        self._max = max_per_window
        self._window = window

    def allow(self) -> bool:
        now = time.monotonic()
        while self._events and now - self._events[0] > self._window:
            self._events.popleft()
        if len(self._events) >= self._max:
            return False
        self._events.append(now)
        return True
