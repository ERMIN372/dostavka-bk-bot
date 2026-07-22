"""Гибридный поиск релевантных чанков: эмбеддинги (косинус) + лексический BM25.

Зачем гибрид: у e5-эмбеддингов на близких темах score «слипаются» (десятки
курьерских Q&A дают 0.85-0.87 без явного лидера), и нужный чанк может проиграть
соседям микроскопическую разницу. Лексическая компонента (BM25) вытягивает чанки,
где буквально встречаются слова вопроса («не забирает», «термосумка», «ПИН»), —
вместе они ранжируют значительно надёжнее, чем каждый по отдельности.

Всё в памяти, без внешних сервисов и зависимостей (numpy + стандартная библиотека).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .indexer import Index
from .pdf_processor import Chunk

logger = logging.getLogger(__name__)

# Кол-во возвращаемых чанков. Индекс небольшой (сотни чанков), поэтому берём с
# запасом: строгий системный промпт YandexGPT игнорирует нерелевантные фрагменты,
# так что больший top-k не повышает риск выдумок.
TOP_K = 10

# Порог косинусного сходства (гейт «есть ли ответ в базе вообще»). Если лучший
# чанк ниже порога — НЕ дёргаем YandexGPT (экономия денег), отвечаем фиксированной
# фразой. Гейт работает по чистому косинусу (семантическое присутствие темы),
# а ранжирование внутри топа — по гибридному score.
#
# ВНИМАНИЕ: значение 0.5 — стартовое, его НУЖНО откалибровать на реальных вопросах
# сотрудников. Для e5-моделей «релевантные» сходства обычно 0.8+, «не по теме» —
# заметно ниже. См. раздел калибровки в README.
SIMILARITY_THRESHOLD = 0.5

# Вес семантики в гибридном score: final = alpha*cos + (1-alpha)*bm25_norm.
# Семантика остаётся главной (перефразировки, опечатки), лексика — уточняющей.
HYBRID_ALPHA = 0.7

# Параметры BM25 (классические значения).
_BM25_K1 = 1.5
_BM25_B = 0.75

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

# Частые русские слова, не несущие смысла для поиска, — не даём им влиять на BM25.
_STOPWORDS = {
    "и", "в", "на", "не", "что", "как", "если", "то", "а", "с", "по", "у", "за",
    "к", "для", "из", "от", "до", "или", "же", "ли", "бы", "это", "его", "её",
    "их", "он", "она", "они", "мы", "вы", "я", "надо", "нужно", "делать", "быть",
    "есть", "при", "об", "про", "так", "уже", "ещё", "еще", "когда", "где", "чем",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class _LexicalIndex:
    """Лёгкий BM25-индекс по чанкам (строится один раз, живёт в памяти)."""

    doc_tokens: List[List[str]]
    df: Dict[str, int] = field(default_factory=dict)
    avgdl: float = 0.0

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "_LexicalIndex":
        doc_tokens = [_tokenize(c.text) for c in chunks]
        df: Dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        avgdl = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(doc_tokens=doc_tokens, df=df, avgdl=avgdl)

    def scores(self, query: str) -> np.ndarray:
        """BM25-баллы запроса по всем чанкам (ненормированные)."""
        n = len(self.doc_tokens)
        out = np.zeros(n, dtype=np.float32)
        if n == 0 or self.avgdl == 0:
            return out
        q_tokens = _tokenize(query)
        for qt in q_tokens:
            dfq = self.df.get(qt)
            if not dfq:
                continue
            idf = math.log(1.0 + (n - dfq + 0.5) / (dfq + 0.5))
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(qt)
                if not tf:
                    continue
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * len(toks) / self.avgdl)
                out[i] += idf * (tf * (_BM25_K1 + 1)) / denom
        return out


# Кеш лексических индексов: id(Index) → _LexicalIndex. Индекс базы знаний создаётся
# один раз при старте, поэтому кеш фактически хранит один элемент.
_lex_cache: Dict[int, _LexicalIndex] = {}


def _get_lexical_index(index: Index) -> _LexicalIndex:
    key = id(index)
    lex = _lex_cache.get(key)
    if lex is None:
        lex = _LexicalIndex.build(index.chunks)
        _lex_cache.clear()  # старых индексов не бывает > 1 — не копим память
        _lex_cache[key] = lex
        logger.info("Лексический BM25-индекс построен: %d чанков.", len(index.chunks))
    return lex


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]        # гибридный score (по нему отранжирован топ)
    cos_scores: List[float]    # косинусная компонента (для /debug)
    lex_scores: List[float]    # нормированная BM25-компонента (для /debug)
    max_score: float           # максимальный КОСИНУС — по нему работает гейт-порог
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
    """Гибридный top-k: ранжируем по alpha*cos + (1-alpha)*bm25, гейт — по косинусу."""
    q = embed_query(model, query)

    # Косинусное сходство = скалярное произведение (всё уже нормировано).
    cos = index.embeddings @ q  # shape (N,)

    # Лексическая компонента: BM25, нормированный к [0, 1] по максимуму запроса.
    lex_raw = _get_lexical_index(index).scores(query)
    lex_max = float(lex_raw.max())
    lex = lex_raw / lex_max if lex_max > 0 else lex_raw

    combined = HYBRID_ALPHA * cos + (1.0 - HYBRID_ALPHA) * lex

    k = min(top_k, combined.shape[0])
    top_idx = np.argpartition(-combined, k - 1)[:k]
    top_idx = top_idx[np.argsort(-combined[top_idx])]

    max_cos = float(cos.max()) if cos.size else 0.0

    logger.info(
        "Retrieval: max_cos=%.4f, threshold=%.2f, top_k=%d, lex_max_raw=%.2f",
        max_cos,
        threshold,
        k,
        lex_max,
    )

    return RetrievalResult(
        chunks=[index.chunks[i] for i in top_idx],
        scores=[float(combined[i]) for i in top_idx],
        cos_scores=[float(cos[i]) for i in top_idx],
        lex_scores=[float(lex[i]) for i in top_idx],
        max_score=max_cos,
        passed_threshold=max_cos >= threshold,
    )
