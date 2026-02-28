import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import telebot
from langchain_ollama import ChatOllama
from python.webAPI.app.config import settings, settings_llm

DEFAULT_CONFIG_JSON = "python/webAPI/app/utils/config.json"

class ConfigError(RuntimeError):
    pass

def _load_impl_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"LLM implementation config not found: {path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _bootstrap_llm_modules() -> None:
    webapi_root = Path("python/webAPI")
    if str(webapi_root.resolve()) not in sys.path:
        sys.path.insert(0, str(webapi_root.resolve()))


def _build_headers() -> dict[str, str]:
    import base64
    encoded = base64.b64encode(
        f"{settings_llm.REMOTE_AUTH_USER}:{settings_llm.REMOTE_AUTH_PASSWORD}".encode()
    ).decode()
    return {"Authorization": f"Basic {encoded}"}


def _pick_class_name(cfg: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = cfg.get(key)
        if value:
            return value
    raise ConfigError(f"Class name not found in config, expected one of: {', '.join(keys)}")


def _index_file(file_path: Path, user_name: str) -> dict[str, Any]:
    from python.webAPI.app.utils.llm_implementation import (
        io_embeddings,
        io_get_vectror_db,
        io_put_vector_in_db,
        io_separate_file,
    )

    separate_cls = getattr(io_separate_file, settings_llm.CLASS_NAME_SEPARATE_FILE)
    embed_cls = getattr(io_embeddings, settings_llm.CLASS_NAME_EMBEDDINGS)
    put_cls = getattr(io_put_vector_in_db, settings_llm.CLASS_NAME_PUT_VECTOR_IN_DB)
    get_cls = getattr(io_get_vectror_db, settings_llm.CLASS_NAME_GET_VECTOR_DB)

    chunks = separate_cls(str(file_path)).separate_file()
    if not chunks:
        raise ValueError("Файл не удалось распарсить или в нём нет текста")

    embedding = embed_cls().get_embeddings()
    indexed = put_cls(chunks, embedding, user_name).put_vector_in_db()
    vectordb = get_cls(user_name, embedding).get_vectror_db()
    records_count = len(vectordb.get().get("ids", [])) if vectordb else 0

    return {"indexed": indexed, "chunk_count": len(chunks), "records_count": records_count}


def _retrieve_and_answer(question: str, user_name: str) -> dict[str, Any]:
    from python.webAPI.app.utils.llm_implementation import (
        io_embeddings,
        io_get_vectror_db,
        io_promt,
        io_search_from_db,
    )

    embed_cls = getattr(io_embeddings, settings_llm.CLASS_NAME_EMBEDDINGS)
    get_cls = getattr(io_get_vectror_db, settings_llm.CLASS_NAME_GET_VECTOR_DB)
    search_cls = getattr(io_search_from_db, settings_llm.CLASS_NAME_SEARCH)
    prompt_cls = getattr(io_promt, settings_llm.CLASS_NAME_PROMT)

    embedding = embed_cls().get_embeddings()
    vectordb = get_cls(user_name, embedding).get_vectror_db()
    search_result = search_cls(question, user_name, vectordb).seach_from_db()

    found = len(search_result) if isinstance(search_result, list) else 0
    if found == 0:
        return {"answer": "Ничего не найдено в базе. Сначала загрузите документ.", "found": 0}

    prompt_text = prompt_cls(search_result, question).get_promt()

    llm = ChatOllama(
        model=settings_llm.REMOTE_LLM_MODEL,
        base_url=settings_llm.REMOTE_LLM_URL,
        client_kwargs={"headers": _build_headers()},
        temperature=0.1,
    )

    response = llm.invoke(prompt_text)
    answer = getattr(response, "content", str(response))
    return {"answer": answer, "found": found}

def _collect_local_upload_files() -> list[Path]:
    """Возвращает список файлов из локальной папки для команды `/upload`."""
    if not settings.local_upload_dir:
        raise ConfigError(
            "Не задана папка загрузки. Укажите LOCAL_RAG_UPLOAD_DIR (или RAG_UPLOAD_DIR) в .env."
        )

    upload_dir = Path(settings.local_upload_dir).expanduser().resolve()
    if not upload_dir.exists() or not upload_dir.is_dir():
        raise ConfigError(f"Папка для загрузки не найдена: {upload_dir}")

    # Ограничиваем список поддерживаемыми форматами, чтобы избежать лишних ошибок у пользователя.
    supported_ext = {".pdf", ".docx", ".pptx"}
    return sorted(path for path in upload_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_ext)


def _clear_user_rag_db(user_name: str, cfg: dict[str, Any]) -> int:
    """Полностью очищает пользовательскую коллекцию Chroma и возвращает количество удалённых записей."""
    from python.webAPI.app.utils.llm_implementation import io_embeddings, io_get_vectror_db

    embed_name = _pick_class_name(cfg, "class_name_embeddings")
    get_name = _pick_class_name(cfg, "class_name_get_vectror_db", "class_name_get_vector_db")

    embedding = getattr(io_embeddings, embed_name)().get_embeddings()
    vectordb = getattr(io_get_vectror_db, get_name)(user_name, embedding).get_vectror_db()

    all_records = vectordb.get()
    ids = all_records.get("ids", [])
    if ids:
        vectordb.delete(ids=ids)
    return len(ids)


def _get_user_indexed_sources(user_name: str, cfg: dict[str, Any]) -> list[str]:
    """Возвращает уникальные пути исходных файлов, которые уже были сохранены в Chroma."""
    from python.webAPI.app.utils.llm_implementation import io_embeddings, io_get_vectror_db

    embed_name = _pick_class_name(cfg, "class_name_embeddings")
    get_name = _pick_class_name(cfg, "class_name_get_vectror_db", "class_name_get_vector_db")

    embedding = getattr(io_embeddings, embed_name)().get_embeddings()
    vectordb = getattr(io_get_vectror_db, get_name)(user_name, embedding).get_vectror_db()

    records = vectordb.get(include=["metadatas"])
    sources: set[str] = set()
    for metadata in records.get("metadatas", []):
        if isinstance(metadata, dict) and metadata.get("source"):
            sources.add(str(metadata["source"]))
    return sorted(sources)

def _short_exc(exc: Exception, limit: int = 900) -> str:
    txt = str(exc)
    return txt if len(txt) <= limit else txt[:limit] + "…"


def run_bot() -> None:
    _bootstrap_llm_modules()

    # ✅ если нижележащие модули читают env-переменные — прокидываем их
    os.environ["CHROMA_PERSIST_DIR"] = settings.CHROMA_PERSIST_DIR
    os.environ["REMOTE_EMBEDDINGS_URL"] = settings_llm.REMOTE_EMBEDDINGS_URL
    os.environ["REMOTE_AUTH_USER"] = settings_llm.REMOTE_AUTH_USER
    os.environ["REMOTE_AUTH_PASSWORD"] = settings_llm.REMOTE_AUTH_PASSWORD

    impl_cfg_path = getattr(settings_llm, "LLM_IMPLEMENTATION_CONFIG", DEFAULT_CONFIG_JSON)
    impl_cfg = _load_impl_config(impl_cfg_path)

    Path(os.environ["CHROMA_PERSIST_DIR"]).mkdir(parents=True, exist_ok=True)

    bot = telebot.TeleBot(settings.TELEGRAM_BOT_TOKEN)

    @bot.message_handler(commands=["start", "help"])
    def _start(message):
        bot.reply_to(
            message,
            (
                "MVP бот запущен.\n"
                "Команды:\n"
                "/upload — загрузить в RAG все файлы из локальной папки из .env\n"
                "/status — показать список уже загруженных файлов\n"
                "/clear — полностью очистить RAG-базу пользователя\n"
                "Также можно отправить файл напрямую в чат."
            ),
        )

    @bot.message_handler(commands=["upload"])
    def _upload_from_local_dir(message):
        user_name = str(message.from_user.id)
        try:
            files = _collect_local_upload_files()
            if not files:
                bot.reply_to(message, "В папке загрузки нет поддерживаемых файлов (.pdf, .docx, .pptx).")
                return

            uploaded_count = 0
            chunks_total = 0
            last_records_count = 0

            for file_path in files:
                stats = _index_file(file_path, user_name, impl_cfg)
                uploaded_count += 1
                chunks_total += int(stats["chunk_count"])
                last_records_count = int(stats["records_count"])

            bot.reply_to(
                message,
                (
                    f"Загрузка завершена.\n"
                    f"Файлов обработано: {uploaded_count}\n"
                    f"Чанков добавлено: {chunks_total}\n"
                    f"Записей в Chroma: {last_records_count}"
                ),
            )
        except Exception as exc:
            bot.reply_to(message, f"Ошибка команды /upload: {_short_exc(exc)}")

    @bot.message_handler(commands=["clear"])
    def _clear_user_db(message):
        user_name = str(message.from_user.id)
        try:
            deleted_count = _clear_user_rag_db(user_name, impl_cfg)
            bot.reply_to(message, f"База очищена. Удалено записей: {deleted_count}.")
        except Exception as exc:
            bot.reply_to(message, f"Ошибка команды /clear: {_short_exc(exc)}")

    @bot.message_handler(commands=["status"])
    def _status_user_db(message):
        user_name = str(message.from_user.id)
        try:
            sources = _get_user_indexed_sources(user_name, impl_cfg)
            if not sources:
                bot.reply_to(message, "В базе пока нет загруженных документов.")
                return

            pretty_list = "\n".join(f"{idx}. {Path(src).name}" for idx, src in enumerate(sources, start=1))
            bot.reply_to(message, f"Загруженные документы ({len(sources)}):\n{pretty_list}")
        except Exception as exc:
            bot.reply_to(message, f"Ошибка команды /status: {_short_exc(exc)}")

    @bot.message_handler(content_types=["document"])
    def _handle_document(message):
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded = bot.download_file(file_info.file_path)

            user_name = str(message.from_user.id)
            user_dir = Path("data/uploads") / user_name
            user_dir.mkdir(parents=True, exist_ok=True)

            filename = message.document.file_name or f"upload_{message.document.file_id}.bin"
            local_path = user_dir / filename
            local_path.write_bytes(downloaded)

            stats = _index_file(local_path, user_name, impl_cfg)
            bot.reply_to(
                message,
                (
                    f"Файл сохранён: {local_path}\n"
                    f"Чанков: {stats['chunk_count']}\n"
                    f"Записей в Chroma: {stats['records_count']}\n"
                    f"Путь Chroma: {os.environ['CHROMA_PERSIST_DIR']}/{user_name}/db"
                ),
            )
        except Exception as exc:
            bot.reply_to(message, f"Ошибка обработки файла: {_short_exc(exc)}")

    @bot.message_handler(content_types=["text"])
    def _handle_text(message):
        question = (message.text or "").strip()
        if not question:
            bot.reply_to(message, "Пустой запрос.")
            return

        try:
            user_name = str(message.from_user.id)
            result = _retrieve_and_answer(question, user_name, impl_cfg)
            bot.reply_to(message, f"Ответ:\n{result['answer']}\n\n[debug] найдено: {result['found']}")
        except Exception as exc:
            bot.reply_to(message, f"Ошибка retrieval: {_short_exc(exc)}")

    print("Telegram MVP bot started (polling).")
    bot.infinity_polling(skip_pending=True)

if __name__ == "__main__":
    try:
        run_bot()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
