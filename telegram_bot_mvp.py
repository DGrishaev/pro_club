import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import telebot
from langchain_ollama import ChatOllama

ENV_FILE = ".env"
LLM_ENV_FILE = ".env.llm"
DEFAULT_CONFIG_JSON = "python/webAPI/app/utils/config.json"


class ConfigError(RuntimeError):
    pass


@dataclass
class Settings:
    telegram_bot_token: str
    chroma_persist_dir: str
    remote_llm_url: str
    remote_auth_user: str
    remote_auth_password: str
    llm_model: str
    llm_impl_config: str
    local_upload_dir: str | None


def _load_dotenv(dotenv_path: str = ENV_FILE) -> None:
    """Загружает пары KEY=VALUE из файла окружения в process env."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required env var: {name}. Fill `.env` based on `.env.example`.")
    return value


def _resolve_chroma_dir(raw_path: str) -> str:
    """Преобразует путь к ChromaDB в абсолютный локальный путь.

    Это позволяет хранить базу вне каталога проекта, если путь задан в `.env`.
    """
    # Поддерживаем `~` и относительные пути, чтобы пользователь мог указывать путь гибко.
    return str(Path(raw_path).expanduser().resolve())


def load_settings() -> Settings:
    # Сначала читаем базовый `.env`, затем `.env.llm` для настроек LLM/GPU.
    _load_dotenv(ENV_FILE)
    _load_dotenv(LLM_ENV_FILE)

    chroma_dir = _resolve_chroma_dir(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))

    return Settings(
        telegram_bot_token=_required_env("TELEGRAM_BOT_TOKEN"),
        chroma_persist_dir=chroma_dir,
        remote_llm_url=_required_env("REMOTE_LLM_URL"),
        remote_auth_user=_required_env("REMOTE_AUTH_USER"),
        remote_auth_password=_required_env("REMOTE_AUTH_PASSWORD"),
        llm_model=os.getenv("REMOTE_LLM_MODEL", "llama3.1"),
        llm_impl_config=os.getenv("LLM_IMPLEMENTATION_CONFIG", DEFAULT_CONFIG_JSON),
        # Поддерживаем несколько имён переменной для обратной совместимости конфигов.
        local_upload_dir=(os.getenv("LOCAL_UPLOAD_DIR")).strip()
        or None,
    )


def _load_impl_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"LLM implementation config not found: {path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _bootstrap_llm_modules() -> None:
    webapi_root = Path("python/webAPI")
    if str(webapi_root.resolve()) not in sys.path:
        sys.path.insert(0, str(webapi_root.resolve()))


def _build_headers(settings: Settings) -> dict[str, str]:
    encoded = base64.b64encode(f"{settings.remote_auth_user}:{settings.remote_auth_password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _pick_class_name(cfg: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = cfg.get(key)
        if value:
            return value
    raise ConfigError(f"Class name not found in config, expected one of: {', '.join(keys)}")


def _index_file(file_path: Path, user_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from python.webAPI.app.utils.llm_implementation import io_embeddings, io_get_vectror_db, io_put_vector_in_db, io_separate_file

    separate_name = _pick_class_name(cfg, "class_name_separate_file")
    embed_name = _pick_class_name(cfg, "class_name_embeddings")
    put_name = _pick_class_name(cfg, "class_name_put_vector_in_db")
    get_name = _pick_class_name(cfg, "class_name_get_vectror_db", "class_name_get_vector_db")

    separate_cls = getattr(io_separate_file, separate_name)
    embed_cls = getattr(io_embeddings, embed_name)
    put_cls = getattr(io_put_vector_in_db, put_name)
    get_cls = getattr(io_get_vectror_db, get_name)

    chunks = separate_cls(str(file_path)).separate_file()
    if not chunks:
        raise ValueError("Файл не удалось распарсить или в нём нет текста")

    embedding = embed_cls().get_embeddings()
    indexed = put_cls(chunks, embedding, user_name).put_vector_in_db()
    vectordb = get_cls(user_name, embedding).get_vectror_db()
    records_count = len(vectordb.get().get("ids", [])) if vectordb else 0

    return {"indexed": indexed, "chunk_count": len(chunks), "records_count": records_count}


def _retrieve_and_answer(question: str, user_name: str, cfg: dict[str, Any], settings: Settings) -> dict[str, Any]:
    from python.webAPI.app.utils.llm_implementation import (
        io_embeddings,
        io_get_vectror_db,
        io_promt,
        io_search_from_db,
    )

    embed_name = _pick_class_name(cfg, "class_name_embeddings")
    get_name = _pick_class_name(cfg, "class_name_get_vectror_db", "class_name_get_vector_db")
    search_name = _pick_class_name(cfg, "class_name_search")
    prompt_name = _pick_class_name(cfg, "class_name_promt", "class_name_prompt")

    embedding = getattr(io_embeddings, embed_name)().get_embeddings()
    vectordb = getattr(io_get_vectror_db, get_name)(user_name, embedding).get_vectror_db()
    search_result = getattr(io_search_from_db, search_name)(question, user_name, vectordb).seach_from_db()

    found = len(search_result) if isinstance(search_result, list) else 0
    if found == 0:
        return {"answer": "Ничего не найдено в базе. Сначала загрузите документ.", "found": 0}

    prompt_text = getattr(io_promt, prompt_name)(search_result, question).get_promt()

    llm = ChatOllama(
        model=cfg.get("model", settings.llm_model),
        base_url=settings.remote_llm_url,
        client_kwargs={"headers": _build_headers(settings)},
        temperature=0.1,
    )

    try:
        response = llm.invoke(prompt_text)
        answer = getattr(response, "content", str(response))
    except Exception:
        answer = "LLM недоступна, но retrieval выполнен успешно."

    return {"answer": answer, "found": found}


def _collect_local_upload_files(settings: Settings) -> list[Path]:
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


def run_bot() -> None:
    settings = load_settings()
    _bootstrap_llm_modules()

    # Прокидываем обязательные настройки в нижележащие модули RAG.
    os.environ["CHROMA_PERSIST_DIR"] = settings.chroma_persist_dir
    os.environ.setdefault("REMOTE_EMBEDDINGS_URL", os.getenv("REMOTE_EMBEDDINGS_URL", settings.remote_llm_url))
    os.environ["REMOTE_AUTH_USER"] = settings.remote_auth_user
    os.environ["REMOTE_AUTH_PASSWORD"] = settings.remote_auth_password

    impl_cfg = _load_impl_config(settings.llm_impl_config)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    bot = telebot.TeleBot(settings.telegram_bot_token)

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
        """Индексирует все поддерживаемые файлы из локальной папки, указанной в `.env`."""
        user_name = str(message.from_user.id)

        try:
            files = _collect_local_upload_files(settings)
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
            bot.reply_to(message, f"Ошибка команды /upload: {exc}")

    @bot.message_handler(commands=["clear"])
    def _clear_user_db(message):
        """Очищает всю RAG-базу текущего пользователя."""
        user_name = str(message.from_user.id)

        try:
            deleted_count = _clear_user_rag_db(user_name, impl_cfg)
            bot.reply_to(message, f"База очищена. Удалено записей: {deleted_count}.")
        except Exception as exc:
            bot.reply_to(message, f"Ошибка команды /clear: {exc}")

    @bot.message_handler(commands=["status"])
    def _status_user_db(message):
        """Показывает список файлов, которые уже участвуют в RAG у пользователя."""
        user_name = str(message.from_user.id)

        try:
            sources = _get_user_indexed_sources(user_name, impl_cfg)
            if not sources:
                bot.reply_to(message, "В базе пока нет загруженных документов.")
                return

            pretty_list = "\n".join(f"{idx}. {Path(src).name}" for idx, src in enumerate(sources, start=1))
            bot.reply_to(message, f"Загруженные документы ({len(sources)}):\n{pretty_list}")
        except Exception as exc:
            bot.reply_to(message, f"Ошибка команды /status: {exc}")

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
                    f"Путь Chroma: {settings.chroma_persist_dir}/{user_name}/db"
                ),
            )
        except Exception as exc:
            bot.reply_to(message, f"Ошибка обработки файла: {exc}")

    @bot.message_handler(content_types=["text"])
    def _handle_text(message):
        question = (message.text or "").strip()
        if not question:
            bot.reply_to(message, "Пустой запрос.")
            return

        try:
            user_name = str(message.from_user.id)
            result = _retrieve_and_answer(question, user_name, impl_cfg, settings)
            bot.reply_to(
                message,
                f"Ответ:\n{result['answer']}\n\n[debug] найдено: {result['found']}",
            )
        except Exception as exc:
            bot.reply_to(message, f"Ошибка retrieval: {exc}")

    print("Telegram MVP bot started (polling).")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    try:
        run_bot()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
