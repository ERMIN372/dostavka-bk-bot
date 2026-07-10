"""Поиск релевантных чанков по косинусному сходству (numpy, в памяти).

Никаких внешних/векторных БД: эмбеддинги уже L2-нормированы, поэтому косинусное
сходство = обычное скалярное произведение (матрично-векторное умножение numpy).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from .indexer import Index
from .pdf_processor import Chunk

logger = logging.getLogger(__name__)

# Кол-во возвращаемых чанков. 8 (а не 5) даёт лучший recall: на составных или
# нечётко сформулированных вопросах нужный фрагмент реже выпадает из контекста.
# Строгий системный промпт YandexGPT игнорирует нерелевантные фрагменты, поэтому
# больший top-k не повышает риск выдумок.
TOP_K = 8

# Порог косинусного сходства. Если лучший чанк ниже порога — считаем, что ответа
# в базе нет, и НЕ дёргаем YandexGPT (экономия денег), отвечаем фиксированной фразой.
#
# ВНИМАНИЕ: значение 0.5 — стартовое, его НУЖНО откалибровать на реальных вопросах
# сотрудников. Для e5-моделей типичные «релевантные» сходства часто лежат в районе
# 0.75-0.85, поэтому порог, возможно, стоит поднять. См. раздел калибровки в README.
SIMILARITY_THRESHOLD = 0.5


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    max_score: float
    passed_threshold: bool


def embed_query(model, query: str) -> np.ndarray:
    """Эмбеддинг вопроса пользователя (префикс 'query:' для e5), L2-нормированный."""
    emb = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb[0].astype(np.float32)


def retrieve(
    index: Index,
    model,
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> RetrievalResult:
    """Находит top-k чанков и решает, прошёл ли лучший из них порог сходства."""
    q = embed_query(model, query)

    # Косинусное сходство = скалярное произведение (всё уже нормировано).
    sims = index.embeddings @ q  # shape (N,)

    k = min(top_k, sims.shape[0])
    # argpartition для top-k, затем сортировка по убыванию сходства.
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    chunks = [index.chunks[i] for i in top_idx]
    scores = [float(sims[i]) for i in top_idx]
    max_score = scores[0] if scores else 0.0

    logger.info(
        "Retrieval: max_score=%.4f, threshold=%.2f, top_k=%d",
        max_score,
        threshold,
        k,
    )

    return RetrievalResult(
        chunks=chunks,
        scores=scores,
        max_score=max_score,
        passed_threshold=max_score >= threshold,
    )
