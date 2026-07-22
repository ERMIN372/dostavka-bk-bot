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
import re
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
#
# ВАЖНО: база знаний — слайдовые PDF (~100-150 токенов текста на страницу),
# где каждая страница = отдельная тема. Крупные чанки (600+) склеивали 5-6
# разнотемных слайдов, из-за чего поиск переставал различать темы (score
# слипались), а нужное правило тонуло в чужом контексте. Поэтому целевой размер
# небольшой, а границы страниц — предпочтительные границы чанков.
CHUNK_TARGET_TOKENS = 350   # целевой размер чанка (~2-3 слайда)
CHUNK_OVERLAP_TOKENS = 100  # overlap между соседними чанками
MIN_CHUNK_TOKENS = 40       # слишком короткие хвосты не выделяем в отдельный чанк

# --- Фильтр OCR-мусора ---
# Скриншоты интерфейсов в PDF дают на OCR кашу вида «Bee craryes! neces a *з
# Datsun on-D0». Такая каша попадала в чанки и портила и поиск, и контекст GPT.
# Оставляем только строки, похожие на осмысленный русский текст.
OCR_MIN_CYRILLIC_RATIO = 0.7  # доля кириллицы среди букв строки
OCR_MIN_RUS_WORDS = 2         # минимум русских слов (3+ буквы) в строке
OCR_MAX_CHARS_PER_PAGE = 600  # OCR — доп. контекст; не даём ему задавить текст слайда


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


_RUS_WORD_RE = re.compile(r"[а-яё]{3,}", re.IGNORECASE)


def _looks_like_russian(line: str) -> bool:
    """Строка похожа на осмысленный русский текст (а не OCR-кашу со скриншота)."""
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 6:
        return False
    cyr = sum(1 for ch in letters if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    if cyr / len(letters) < OCR_MIN_CYRILLIC_RATIO:
        return False
    return len(_RUS_WORD_RE.findall(line)) >= OCR_MIN_RUS_WORDS


def _clean_ocr_text(text: str) -> str:
    """Отсекает OCR-мусор: оставляет только русскоязычные строки, дедуплицирует,
    ограничивает объём (OCR — вспомогательный контекст, не основной текст)."""
    seen: set[str] = set()
    kept: List[str] = []
    total = 0
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line or not _looks_like_russian(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if total + len(line) > OCR_MAX_CHARS_PER_PAGE:
            break
        kept.append(line)
        total += len(line)
    return "\n".join(kept)


def _ocr_page_images(page: "fitz.Page", doc: "fitz.Document") -> str:
    """OCR по всем растровым изображениям страницы. Возвращает распознанный текст,
    очищенный от мусора (см. _clean_ocr_text)."""
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
    return _clean_ocr_text("\n".join(ocr_parts))


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


def _split_long_page(page_num: int, page_text: str, source: str) -> List[Chunk]:
    """Редкая длинная страница (больше целевого размера) — режем по абзацам с overlap."""
    paragraphs = _split_paragraphs(page_text)
    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_tokens = 0
    for para in paragraphs:
        pt = _estimate_tokens(para)
        if cur and cur_tokens + pt > CHUNK_TARGET_TOKENS:
            chunks.append(Chunk("\n\n".join(cur), source, page_num, page_num))
            # overlap: последние абзацы на ~CHUNK_OVERLAP_TOKENS
            keep: List[str] = []
            kt = 0
            for p in reversed(cur):
                t = _estimate_tokens(p)
                if kt + t > CHUNK_OVERLAP_TOKENS and keep:
                    break
                keep.insert(0, p)
                kt += t
            cur, cur_tokens = keep, kt
        cur.append(para)
        cur_tokens += pt
    if cur:
        chunks.append(Chunk("\n\n".join(cur), source, page_num, page_num))
    return chunks


def _chunk_pages(pages: List[tuple[int, str]], source: str) -> List[Chunk]:
    """Режет документ на чанки, предпочитая границы СТРАНИЦ как смысловые границы.

    База знаний — слайдовые PDF: одна страница = одна тема. Поэтому чанк — это
    1-3 соседние страницы до целевого размера; страница никогда не рвётся на
    середине (кроме редких страниц длиннее целевого размера — их режем по
    абзацам). Overlap — последняя страница предыдущего чанка повторяется в
    начале следующего, если она короче ~CHUNK_OVERLAP_TOKENS.
    """
    chunks: List[Chunk] = []
    cur_pages: List[tuple[int, str]] = []
    cur_tokens = 0
    carried_only = False  # в cur_pages лежит только overlap-страница из прошлого чанка

    def flush() -> None:
        nonlocal cur_pages, cur_tokens, carried_only
        if not cur_pages:
            return
        text = "\n\n".join(p[1] for p in cur_pages).strip()
        if text:
            chunks.append(
                Chunk(text, source, cur_pages[0][0], cur_pages[-1][0])
            )
        # Overlap: тянем последнюю страницу в следующий чанк, только если она
        # короткая (иначе чанки будут почти дублироваться).
        last_num, last_text = cur_pages[-1]
        last_tokens = _estimate_tokens(last_text)
        if last_tokens <= CHUNK_OVERLAP_TOKENS:
            cur_pages, cur_tokens = [(last_num, last_text)], last_tokens
            carried_only = True
        else:
            cur_pages, cur_tokens = [], 0
            carried_only = False

    for page_num, page_text in pages:
        ptoks = _estimate_tokens(page_text)

        # Аномально длинная страница — отдельная обработка, по абзацам.
        if ptoks > CHUNK_TARGET_TOKENS:
            flush()
            cur_pages, cur_tokens, carried_only = [], 0, False
            chunks.extend(_split_long_page(page_num, page_text, source))
            continue

        if cur_pages and cur_tokens + ptoks > CHUNK_TARGET_TOKENS:
            flush()
        cur_pages.append((page_num, page_text))
        cur_tokens += ptoks
        carried_only = False

    # Финальный хвост — но не дублируем чанк, состоящий из одной overlap-страницы.
    if cur_pages and not carried_only:
        text = "\n\n".join(p[1] for p in cur_pages).strip()
        if text:
            chunks.append(Chunk(text, source, cur_pages[0][0], cur_pages[-1][0]))

    return chunks


def process_pdf(pdf_path: str, source_name: str) -> List[Chunk]:
    """Полный конвейер для одного PDF: текст + OCR → чанки."""
    logger.info("Извлечение текста и OCR из %s ...", source_name)
    pages = _extract_pages(pdf_path)
    chunks = _chunk_pages(pages, source_name)
    logger.info("Файл %s: страниц с текстом %d, чанков %d", source_name, len(pages), len(chunks))
    return chunks
