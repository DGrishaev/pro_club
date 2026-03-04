"""Локальный smoke-тест ingestion без запуска Telegram-бота."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean


def _build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер аргументов."""
    parser = argparse.ArgumentParser(description="Smoke-проверка чанкинга/ingestion.")
    parser.add_argument("--file", required=True, help="Путь к входному файлу (.docx/.pdf/.pptx)")
    return parser


def _short(text: str, limit: int = 120) -> str:
    """Возвращает короткий фрагмент текста для печати в консоль."""
    compact = " ".join((text or "").split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def main() -> None:
    """Запускает smoke-проверку и печатает краткую статистику по чанкам."""
    args = _build_parser().parse_args()

    from python.webAPI.app.utils.llm_implementation.io_separate_file import (
        sf_DataProcessing_keywords_512_chunk_and_Tables,
    )

    chunks = sf_DataProcessing_keywords_512_chunk_and_Tables(args.file).separate_file()
    lengths = [len((getattr(doc, "page_content", "") or "")) for doc in chunks]
    types = Counter((getattr(doc, "metadata", {}) or {}).get("type", "unknown") for doc in chunks)

    print(f"Всего чанков: {len(chunks)}")
    print("Чанки по type:")
    for key, value in sorted(types.items(), key=lambda item: item[0]):
        print(f"  - {key}: {value}")

    if lengths:
        print(
            "Длины page_content: "
            f"min={min(lengths)}, avg={round(mean(lengths), 2)}, max={max(lengths)}"
        )
    else:
        print("Длины page_content: нет данных")

    # Показываем 10 самых коротких чанков, чтобы быстро увидеть избыточную мелкую нарезку.
    print("Топ-10 самых коротких чанков:")
    sorted_chunks = sorted(chunks, key=lambda doc: len((getattr(doc, "page_content", "") or "")))
    for idx, doc in enumerate(sorted_chunks[:10], start=1):
        meta = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""
        print(
            f"  {idx}. len={len(content)} type={meta.get('type')} "
            f"section={meta.get('section')} chunk_id={meta.get('chunk_id')} text='{_short(content)}'"
        )

    heading_chunks = types.get("heading", 0)
    toc_marked = 0
    for doc in chunks:
        meta = getattr(doc, "metadata", {}) or {}
        flags_raw = meta.get("flags_json")
        if not flags_raw:
            continue
        try:
            flags = json.loads(flags_raw)
        except Exception:
            continue
        if isinstance(flags, dict) and flags.get("is_toc") is True:
            toc_marked += 1
    print(
        "Проверка heading/toc: "
        f"heading_chunks={heading_chunks}, toc_marked_chunks={toc_marked} "
        "(при RAG_SKIP_TOC=true и RAG_SKIP_HEADINGS=true ожидается 0)."
    )


if __name__ == "__main__":
    main()
