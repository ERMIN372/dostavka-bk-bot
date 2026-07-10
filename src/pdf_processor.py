"""Извлечение текста и OCR-текста изображений из PDF + разбиение на чанки.

Используется PyMuPDF (fitz): он даёт чистый текст на кириллице и удобный доступ
к встроенным изображениям страниц. С каждой страницы:
  1. берётся текстовый слой (fitz page.get_text);
  2. извлекаются растровые изображения, к ним применяется OCR (pytesseract, rus);
  3. OCR-текст приклеивается к тексту той же страницы как дополнительный контекст.

Далее весь текст страниц объединяется и режется на чанки с overlap по смысловым
границам (абзацам), а не жёстко посимвольно.

TODO (на будущее): вместо/в дополнение к OCR можно подключить vision-captioning
(описание схем/картинок через мультимодальную модель). По умолчанию ветка
выключена — сейчас работает только OCR. См. USE_VISION_CAPTIONING ниже.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# --- Заглушка для будущего vision-captioning изображений (по умолчанию OFF) ---
# Если в будущем захочется описывать схемы/картинки мультимодальной моделью —
# реализовать здесь и включить флаг. Сейчас используется только OCR.
USE_VISION_CAPTIONING = False

# Язык(и) для tesseract. Требуется установленный пакет tesseract-ocr-rus.
# rus+eng — на случай латинских вставок (артикулы, англ. термины).
OCR_LANG = "rus+eng"

# Параметры чанкинга. Токены оцениваются приблизительно (см. _estimate_tokens):
# точный токенайзер не нужен, важен порядок величины для стабильных размеров чанков.
CHUNK_TARGET_TOKENS = 650   # целевой размер чанка (~500-800 токенов)
CHUNK_OVERLAP_TOKENS = 100  # overlap между соседними чанками
MIN_CHUNK_TOKENS = 40       # слишком короткие хвосты не выделяем в отдельный чанк


@dataclass
class Chunk:
    """Единица индексации: кусок текста + служебные метаданные (для отладки)."""

    text: str
    source: str      # имя исходного PDF-файла
    page_start: int  # номера страниц (1-based) для отладки; в ответе НЕ используются
    page_end: int


def _estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов.

    Для смешанного русского текста ~4 символа на токен — достаточная аппроксимация
    для нарезки чанков. Точный токенайзер здесь избыточен.
    """
    return max(1, len(text) // 4)


def _ocr_page_images(page: "fitz.Page", doc: "fitz.Document") -> str:
    """OCR по всем растровым изображениям страницы. Возвращает распознанный текст."""
    ocr_parts: List[str] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base["image"]))
            recognized = pytesseract.image_to_string(image, lang=OCR_LANG)
            recognized = recognized.strip()
            if recognized:
                ocr_parts.append(recognized)
        except Exception as exc:  # noqa: BLE001 — не валим индексацию из-за одной картинки
            logger.warning("OCR не удался для изображения xref=%s: %s", xref, exc)
    return "\n".join(ocr_parts)


def _extract_pages(pdf_path: str) -> List[tuple[int, str]]:
    """Извлекает (номер_страницы, текст+OCR) для каждой страницы PDF."""
    pages: List[tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            ocr_text = _ocr_page_images(page, doc)
            combined = text.strip()
            if ocr_text:
                # OCR-текст приклеивается к тексту той же страницы как доп. контекст.
                combined = (combined + "\n" + ocr_text).strip()
            if combined:
                pages.append((page_index + 1, combined))
            logger.info(
                "Страница %d/%d обработана (%d символов%s)",
                page_index + 1,
                doc.page_count,
                len(combined),
                ", + OCR" if ocr_text else "",
            )
    finally:
        doc.close()
    return pages


def _split_paragraphs(text: str) -> List[str]:
    """Режет текст на абзацы по пустым строкам; длинные абзацы — по строкам."""
    raw = [p.strip() for p in text.split("\n\n")]
    paragraphs: List[str] = []
    for p in raw:
        if not p:
            continue
        # Очень длинный «абзац» без пустых строк добиваем по одиночным переводам.
        if _estimate_tokens(p) > CHUNK_TARGET_TOKENS * 2:
            paragraphs.extend(line.strip() for line in p.split("\n") if line.strip())
        else:
            paragraphs.append(p)
    return paragraphs


def _chunk_pages(pages: List[tuple[int, str]], source: str) -> List[Chunk]:
    """Собирает страницы в единый поток абзацев и режет на чанки с overlap.

    Чанкинг идёт по смысловым границам (абзацам): абзацы накапливаются в чанк,
    пока не достигнут целевой размер; при переполнении часть последних абзацев
    (~overlap токенов) переносится в начало следующего чанка.
    """
    # Плоский список (страница, абзац) — чтобы знать примерный диапазон страниц.
    units: List[tuple[int, str]] = []
    for page_num, page_text in pages:
        for para in _split_paragraphs(page_text):
            units.append((page_num, para))

    chunks: List[Chunk] = []
    cur_units: List[tuple[int, str]] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_units, cur_tokens
        if not cur_units:
            return
        text = "\n\n".join(u[1] for u in cur_units).strip()
        if not text:
            cur_units, cur_tokens = [], 0
            return
        chunks.append(
            Chunk(
                text=text,
                source=source,
                page_start=cur_units[0][0],
                page_end=cur_units[-1][0],
            )
        )
        # Формируем overlap: тянем абзацы с конца, пока не наберём ~overlap токенов.
        overlap: List[tuple[int, str]] = []
        overlap_tokens = 0
        for unit in reversed(cur_units):
            t = _estimate_tokens(unit[1])
            if overlap_tokens + t > CHUNK_OVERLAP_TOKENS and overlap:
                break
            overlap.insert(0, unit)
            overlap_tokens += t
        cur_units = overlap
        cur_tokens = overlap_tokens

    for page_num, para in units:
        para_tokens = _estimate_tokens(para)
        if cur_tokens + para_tokens > CHUNK_TARGET_TOKENS and cur_tokens >= MIN_CHUNK_TOKENS:
            flush()
        cur_units.append((page_num, para))
        cur_tokens += para_tokens

    # Финальный хвост.
    if cur_units:
        text = "\n\n".join(u[1] for u in cur_units).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source=source,
                    page_start=cur_units[0][0],
                    page_end=cur_units[-1][0],
                )
            )

    return chunks


def process_pdf(pdf_path: str, source_name: str) -> List[Chunk]:
    """Полный конвейер для одного PDF: текст + OCR → чанки."""
    logger.info("Извлечение текста и OCR из %s ...", source_name)
    pages = _extract_pages(pdf_path)
    chunks = _chunk_pages(pages, source_name)
    logger.info("Файл %s: страниц с текстом %d, чанков %d", source_name, len(pages), len(chunks))
    return chunks
