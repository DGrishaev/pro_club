"""
Простой CLI-скрипт для локальной диагностики этапа поиска (retrieval).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

WEBAPI_ROOT = Path(__file__).resolve().parents[2]
if str(WEBAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBAPI_ROOT))

from app.config import settings_llm
from app.utils.llm_implementation import io_embeddings, io_get_vectror_db, io_search_from_db


def _read_question(file_path: Path, inline: str | None) -> str:
    """Возвращает текст вопроса: из файла user_question.txt или аргумента CLI."""
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    if inline:
        inline = inline.strip()
        if inline:
            return inline
    raise SystemExit(
        f"Не найден текст вопроса. Создайте файл {file_path} или передайте параметр --question."
    )


def _format_result(idx: int, item: dict[str, Any]) -> str:
    """Готовит строку с диагностикой одного найденного чанка."""
    metadata = item.get("metadata") or {}
    score = item.get("score")
    type_name = metadata.get("type")
    doc_id = None
    for key in ("id", "doc_id", "chunk_id", "chunk", "source"):
        if metadata.get(key) is not None:
            doc_id = metadata.get(key)
            break
    snippet = (item.get("content") or "").replace("\n", " ")[:200]
    return (
        f"{idx}. id={doc_id} score={score} type={type_name} source={metadata.get('source')} "
        f"-> {snippet}"
    )


def _normalize_cli_item(item: Any) -> dict[str, Any]:
    """Приводит результат поиска к словарю с единым интерфейсом."""
    if isinstance(item, dict):
        return item
    return {
        "content": getattr(item, "page_content", ""),
        "metadata": getattr(item, "metadata", {}) or {},
        "score": getattr(item, "score", None),
    }


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="Диагностика поиска в Chroma (s_k_five).")
    parser.add_argument("--user", "-u", required=True, help="Имя пользователя/папки в Chroma.")
    parser.add_argument(
        "--question",
        "-q",
        help="Текст вопроса (если не задан, будет прочитан из user_question.txt).",
    )
    parser.add_argument(
        "--question-file",
        "-f",
        default="user_question.txt",
        help="Путь к файлу с вопросом (по умолчанию user_question.txt).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Количество документов в выдаче (по умолчанию 5).",
    )

    args = parser.parse_args()
    question = _read_question(Path(args.question_file), args.question)

    # Автоматически переопределяем RAG_RETRIEVAL_K, чтобы s_k_five возвращал нужное k.
    os.environ["RAG_RETRIEVAL_K"] = str(args.k)

    embed_cls = getattr(io_embeddings, settings_llm.CLASS_NAME_EMBEDDINGS)
    embedding = embed_cls().get_embeddings()

    gvid_cls = getattr(io_get_vectror_db, settings_llm.CLASS_NAME_GET_VECTOR_DB)
    vectordb = gvid_cls(args.user, embedding).get_vectror_db()

    search = io_search_from_db.s_k_five(question, args.user, vectordb)
    results = search.seach_from_db() or []

    print(f"Вопрос: {question}")
    print(f"Найдено документов: {len(results)}")
    for idx, item in enumerate(results, start=1):
        print(_format_result(idx, _normalize_cli_item(item)))


if __name__ == "__main__":
    main()
