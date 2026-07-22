"""Обработка Markdown-баз знаний в формате Q&A.

Формат файла (см. knowledge_base/BK_delivery_knowledge_base_QA.md):
  ## Раздел
  ### Вопрос
  Текст ответа...
  *Источник: ...*   ← служебная строка, в индекс не попадает

Каждая пара «вопрос-ответ» становится ОДНИМ чанком — это идеальная гранулярность
для RAG: заголовок-вопрос семантически совпадает с реальными вопросами
сотрудников, а ответ самодостаточен. Никакого дополнительного чанкинга/overlap
не требуется.
"""

from __future__ import annotations

import logging
import re
from typing import List

from .pdf_processor import Chunk

logger = logging.getLogger(__name__)

# Служебные строки «*Источник: …*» — метаданные для людей, в индекс не включаем:
# в ответах бота ссылки на страницы не нужны (и промпт их запрещает).
_SOURCE_LINE_RE = re.compile(r"^\*Источник:.*\*\s*$", re.IGNORECASE)


def process_markdown(md_path: str, source_name: str) -> List[Chunk]:
    """Читает Q&A-markdown и возвращает чанки: одна пара вопрос-ответ = один чанк.

    В page_start/page_end пишем порядковый номер Q&A-пары (страниц у md нет) —
    удобно для /debug.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    chunks: List[Chunk] = []
    section = ""
    question = ""
    answer_lines: List[str] = []

    def flush() -> None:
        nonlocal question, answer_lines
        answer = "\n".join(answer_lines).strip()
        if question and answer:
            parts = []
            if section:
                parts.append(f"Раздел: {section}")
            parts.append(f"Вопрос: {question}")
            parts.append(f"Ответ: {answer}")
            n = len(chunks) + 1
            chunks.append(Chunk("\n".join(parts), source_name, n, n))
        question, answer_lines = "", []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("### "):
            flush()
            question = line[4:].strip()
        elif line.startswith("## "):
            flush()
            # Убираем нумерацию раздела («## 3. Название» → «Название»).
            section = re.sub(r"^\d+\.\s*", "", line[3:].strip())
        elif line.startswith("# ") or line.startswith(">"):
            continue  # заголовок документа и цитаты-примечания — не контент
        elif _SOURCE_LINE_RE.match(line.strip()):
            continue
        else:
            if question:
                answer_lines.append(line)

    flush()

    logger.info("Файл %s: Q&A-пар (чанков): %d", source_name, len(chunks))
    return chunks
