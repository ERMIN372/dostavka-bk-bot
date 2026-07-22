"""Построение, загрузка и кеширование индекса эмбеддингов.

Логика:
  * при старте перебираем все PDF в knowledge_base;
  * считаем общий хеш содержимого файлов;
  * если хеш совпадает с сохранённым в .cache — просто грузим готовый индекс;
  * иначе: заново извлекаем текст+OCR, режем на чанки, считаем эмбеддинги
    локальной моделью sentence-transformers и сохраняем кеш.

Индекс хранится локально (без всякой БД и без векторных БД):
  * embeddings.npz — матрица эмбеддингов (float32) + метаданные чанков;
  * meta.json      — хеш исходных файлов и параметры (модель, размерность).

ВАЖНО про Railway: кеш лежит на диске контейнера. Если не подключён
персистентный volume, при каждом передеплое контейнер стартует с нуля и индекс
пересчитывается заново. Для 84-страничного PDF это нормально (секунды/десятки
секунд на CPU). Подробнее — в README.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from .pdf_processor import Chunk, process_pdf

logger = logging.getLogger(__name__)

# Локальная multilingual-модель эмбеддингов (сильная на русском). Бесплатно, на CPU.
# e5-модели требуют префиксов "query: " / "passage: " — см. retriever/indexer.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large"
)

# Версия конвейера обработки PDF (чанкер/OCR-фильтр). При изменении логики
# нарезки бампается, чтобы кеш с чанками старого формата не пережил обновление.
PIPELINE_VERSION = "page-chunks-v2"

CACHE_DIR_NAME = ".cache"
EMBEDDINGS_FILE = "embeddings.npz"
META_FILE = "meta.json"


@dataclass
class Index:
    """Загруженный в память индекс: эмбеддинги + тексты чанков."""

    embeddings: np.ndarray      # shape (N, dim), L2-нормированные float32
    chunks: List[Chunk]
    model_name: str


def _iter_pdf_files(knowledge_base_dir: str) -> List[str]:
    """Возвращает отсортированный список путей к PDF в папке базы знаний."""
    files: List[str] = []
    for name in sorted(os.listdir(knowledge_base_dir)):
        full = os.path.join(knowledge_base_dir, name)
        if os.path.isfile(full) and name.lower().endswith(".pdf"):
            files.append(full)
    return files


def _compute_files_hash(pdf_files: List[str]) -> str:
    """Единый хеш по содержимому и именам всех PDF — для инвалидации кеша."""
    h = hashlib.sha256()
    for path in pdf_files:
        h.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Загрузка sentence-transformers (тяжёлый импорт держим внутри функции).

    Модель нужна и для индексации (эмбеддинги чанков), и в рантайме для
    эмбеддинга вопросов. Грузим один раз в main.py и переиспользуем.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Загрузка модели эмбеддингов: %s ...", model_name)
    return SentenceTransformer(model_name)


def embed_passages(model, texts: List[str]) -> np.ndarray:
    """Эмбеддинги для чанков-документов (префикс 'passage:' для e5)."""
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-норма → косинусное сходство = скалярное произв.
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def _cache_paths(knowledge_base_dir: str) -> tuple[str, str, str]:
    cache_dir = os.path.join(knowledge_base_dir, CACHE_DIR_NAME)
    return (
        cache_dir,
        os.path.join(cache_dir, EMBEDDINGS_FILE),
        os.path.join(cache_dir, META_FILE),
    )


def _load_cache(knowledge_base_dir: str, files_hash: str) -> Index | None:
    """Пробует загрузить индекс из кеша, если хеш и модель совпадают."""
    _, emb_path, meta_path = _cache_paths(knowledge_base_dir)
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("files_hash") != files_hash:
            logger.info("Хеш исходных PDF изменился — кеш будет пересчитан.")
            return None
        if meta.get("model_name") != EMBEDDING_MODEL_NAME:
            logger.info("Изменилась модель эмбеддингов — кеш будет пересчитан.")
            return None
        if meta.get("pipeline_version") != PIPELINE_VERSION:
            logger.info("Изменилась версия чанкера — кеш будет пересчитан.")
            return None

        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        texts = data["texts"]
        sources = data["sources"]
        page_starts = data["page_starts"]
        page_ends = data["page_ends"]

        chunks = [
            Chunk(
                text=str(texts[i]),
                source=str(sources[i]),
                page_start=int(page_starts[i]),
                page_end=int(page_ends[i]),
            )
            for i in range(len(texts))
        ]
        logger.info("Индекс загружен из кеша: %d чанков.", len(chunks))
        return Index(embeddings=embeddings, chunks=chunks, model_name=EMBEDDING_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 — битый кеш просто пересчитаем
        logger.warning("Не удалось прочитать кеш (%s) — пересчитываем.", exc)
        return None


def _save_cache(
    knowledge_base_dir: str,
    files_hash: str,
    embeddings: np.ndarray,
    chunks: List[Chunk],
) -> None:
    cache_dir, emb_path, meta_path = _cache_paths(knowledge_base_dir)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        texts=np.array([c.text for c in chunks], dtype=object),
        sources=np.array([c.source for c in chunks], dtype=object),
        page_starts=np.array([c.page_start for c in chunks], dtype=np.int32),
        page_ends=np.array([c.page_end for c in chunks], dtype=np.int32),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "files_hash": files_hash,
                "model_name": EMBEDDING_MODEL_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "dim": int(embeddings.shape[1]) if embeddings.size else 0,
                "num_chunks": len(chunks),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Индекс сохранён в кеш: %s", cache_dir)


def build_or_load_index(knowledge_base_dir: str, model=None) -> Index:
    """Главная точка: вернуть индекс, пересчитав только при изменении PDF.

    model — уже загруженная модель эмбеддингов (чтобы не грузить её дважды).
    Если None и требуется пересчёт — модель будет загружена здесь.

    Кидает RuntimeError, если в knowledge_base нет ни одного PDF — бот без базы
    знаний бессмысленен, лучше явно упасть при старте.
    """
    if not os.path.isdir(knowledge_base_dir):
        raise RuntimeError(f"Папка базы знаний не найдена: {knowledge_base_dir}")

    pdf_files = _iter_pdf_files(knowledge_base_dir)
    if not pdf_files:
        raise RuntimeError(
            f"В {knowledge_base_dir} нет ни одного PDF-файла. "
            "Положите PDF с инструкциями в эту папку и перезапустите бота."
        )

    logger.info("Найдено PDF-файлов: %d", len(pdf_files))
    files_hash = _compute_files_hash(pdf_files)

    cached = _load_cache(knowledge_base_dir, files_hash)
    if cached is not None:
        return cached

    # --- Пересчёт индекса ---
    all_chunks: List[Chunk] = []
    for path in pdf_files:
        all_chunks.extend(process_pdf(path, os.path.basename(path)))

    if not all_chunks:
        raise RuntimeError(
            "Из PDF не удалось извлечь ни одного текстового чанка. "
            "Проверьте, что PDF не пустой и установлен tesseract-ocr(-rus)."
        )

    if model is None:
        model = load_embedding_model(EMBEDDING_MODEL_NAME)
    logger.info("Считаем эмбеддинги для %d чанков ...", len(all_chunks))
    embeddings = embed_passages(model, [c.text for c in all_chunks])

    _save_cache(knowledge_base_dir, files_hash, embeddings, all_chunks)
    return Index(embeddings=embeddings, chunks=all_chunks, model_name=EMBEDDING_MODEL_NAME)
