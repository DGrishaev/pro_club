"""Утилиты для сохранения сырых артефактов RAG."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

try:
    from app.config import settings  # type: ignore
except Exception:
    settings = None

ArtifactKind = Literal["UP", "QU"]

_SUBDIRS: dict[ArtifactKind, tuple[str, ...]] = {
    "UP": ("meta", "input", "parsed", "chunks", "embeddings", "index", "errors"),
    "QU": ("meta", "query", "retrieval", "prompt", "llm", "errors"),
}

_SANITIZE_RE = re.compile(r"[^0-9A-Za-z._-]+")


def _env_bool(value: str | None, default: bool = False) -> bool:
    """Возвращает булево значение на основе текстового представления флага."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sanitize_filename(name: str) -> str:
    """Приводит имя файла к безопасному виду без спецсимволов."""
    if not name:
        return "artifact"
    basename = os.path.basename(name)
    sanitized = _SANITIZE_RE.sub("_", basename)
    return sanitized or "artifact"


def serialize_documents(documents: Iterable[Any]) -> list[dict[str, Any]]:
    """Преобразует документы LangChain в сериализуемый словарь."""
    serialized: list[dict[str, Any]] = []
    for idx, item in enumerate(documents):
        page_content = getattr(item, "page_content", None)
        metadata = getattr(item, "metadata", None)
        if isinstance(item, dict):
            page_content = item.get("page_content")
            metadata = item.get("metadata")
        serialized.append(
            {
                "index": idx,
                "page_content": page_content or "",
                "metadata": metadata or {},
            }
        )
    return serialized


def combine_page_content(items: Iterable[Any]) -> str:
    """Склеивает содержимое документов в одно текстовое представление."""
    parts: list[str] = []
    for obj in items:
        value = getattr(obj, "page_content", None)
        if isinstance(obj, dict):
            value = obj.get("page_content")
        if value:
            parts.append(str(value))
    return "\n\n-----\n\n".join(parts)


def normalize_retrieval_results(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Нормализует результаты поиска для сохранения в JSON."""
    if results is None:
        return []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(results):
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
            normalized.append(
                {
                    "index": idx,
                    "score": score,
                    "metadata": getattr(doc, "metadata", {}) or {},
                    "content": getattr(doc, "page_content", "") or "",
                }
            )
        elif isinstance(item, dict):
            normalized.append(
                {
                    "index": idx,
                    "score": item.get("score"),
                    "metadata": item.get("metadata") or {},
                    "content": item.get("content") or "",
                }
            )
        else:
            normalized.append(
                {
                    "index": idx,
                    "score": None,
                    "metadata": getattr(item, "metadata", {}) or {},
                    "content": getattr(item, "page_content", "") or "",
                }
            )
    return normalized


def build_context_blob(data: Any) -> str:
    """Готовит текст контекста для сохранения в prompt/context_only.txt."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return combine_page_content(data)
    if isinstance(data, dict):
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)
    return str(data)


@dataclass
class _Session:
    """Хранит состояние текущей сессии сохранения артефактов."""

    kind: ArtifactKind
    root: Path
    name: str
    folders: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def ensure_created(self) -> None:
        """Создаёт каталог сессии и вложенные папки при первом обращении."""
        if self.path is not None:
            return
        target = self.root / self.name
        for child in self.folders:
            (target / child).mkdir(parents=True, exist_ok=True)
        self.path = target

    def folder(self, subdir: str) -> Path:
        """Возвращает путь к подкаталогу конкретного типа."""
        if self.path is None:
            raise RuntimeError("Сессия ещё не инициализирована")
        return self.path / subdir


class RagArtifactsManager:
    """Менеджер, который создаёт каталоги и сохраняет артефакты RAG."""

    def __init__(self) -> None:
        flag = os.getenv("DEBUG_RAG")
        if flag is None and settings is not None and hasattr(settings, "DEBUG_RAG"):
            flag = str(getattr(settings, "DEBUG_RAG"))
        self.enabled = _env_bool(flag, False)

        root_raw = os.getenv("LOGS_FOLDER_PATH")
        if not root_raw and settings is not None and hasattr(settings, "LOGS_FOLDER_PATH"):
            root_raw = getattr(settings, "LOGS_FOLDER_PATH")
        self.root = Path(root_raw or "./rag_logs").expanduser()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._counter = 0
        self._lock = threading.Lock()

    def _state(self) -> dict[str, _Session | None]:
        """Возвращает словарь активных сессий в рамках текущего потока."""
        if not hasattr(self._local, "sessions"):
            self._local.sessions = {}
        return self._local.sessions  # type: ignore[return-value]

    def _next_name(self, kind: ArtifactKind) -> str:
        """Генерирует уникальное имя каталога сессии."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with self._lock:
            self._counter += 1
            suffix = self._counter
        return f"{kind}_{timestamp}_{suffix:03d}"

    @contextmanager
    def up_session(self, source_path: str | None = None, extra_meta: dict[str, Any] | None = None):
        """Контекст для пути загрузки/индексации."""
        base_meta = {"source_path": source_path, "user_meta": extra_meta or {}}
        with self._session("UP", base_meta):
            yield

    @contextmanager
    def qu_session(self, question: str | None = None, extra_meta: dict[str, Any] | None = None):
        """Контекст для пути вопрос/ответ."""
        base_meta = {"question": question, "user_meta": extra_meta or {}}
        with self._session("QU", base_meta):
            yield

    @contextmanager
    def _session(self, kind: ArtifactKind, meta: dict[str, Any]):
        """Создаёт и управляет сессией указанного типа."""
        if not self.enabled:
            yield
            return

        session = _Session(kind=kind, root=self.root, name=self._next_name(kind), folders=_SUBDIRS[kind], meta=meta)
        session.ensure_created()
        state = self._state()
        previous = state.get(kind)
        state[kind] = session
        self.write_json(kind, "meta", "session.json", {"started_at": datetime.now().isoformat(), **meta})
        try:
            yield
        except Exception:
            raise
        else:
            print(f"RAG {kind} debug saved to: {session.path}")
        finally:
            state[kind] = previous

    def has_session(self, kind: ArtifactKind) -> bool:
        """Проверяет, активен ли сейчас контекст нужного типа."""
        if not self.enabled:
            return False
        return self._state().get(kind) is not None

    def _ensure_session(self, kind: ArtifactKind) -> _Session | None:
        """Возвращает активную сессию либо None."""
        session = self._state().get(kind)
        return session

    def write_text(self, kind: ArtifactKind, subdir: str, filename: str, text: str | None) -> None:
        """Сохраняет текстовый файл внутри текущей сессии."""
        if not self.enabled or text is None:
            return
        session = self._ensure_session(kind)
        if session is None:
            return
        target = session.folder(subdir) / sanitize_filename(filename)
        target.write_text(str(text), encoding="utf-8")

    def write_json(self, kind: ArtifactKind, subdir: str, filename: str, data: Any) -> None:
        """Сохраняет JSON без усечения данных."""
        if not self.enabled:
            return
        session = self._ensure_session(kind)
        if session is None:
            return
        target = session.folder(subdir) / sanitize_filename(filename)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def write_bytes(self, kind: ArtifactKind, subdir: str, filename: str, blob: bytes | bytearray | None) -> None:
        """Сохраняет бинарный артефакт."""
        if not self.enabled or blob is None:
            return
        session = self._ensure_session(kind)
        if session is None:
            return
        target = session.folder(subdir) / sanitize_filename(filename)
        with target.open("wb") as fp:
            fp.write(blob)

    def copy_file(self, kind: ArtifactKind, subdir: str, source_path: str | None) -> None:
        """Копирует исходный файл в каталог диагностики."""
        if not self.enabled or not source_path:
            return
        session = self._ensure_session(kind)
        if session is None:
            return
        try:
            target = session.folder(subdir) / sanitize_filename(source_path)
            shutil.copyfile(source_path, target)
        except Exception as exc:
            self.log_exception(kind, "copy_file", exc, {"source_path": source_path})

    def log_exception(self, kind: ArtifactKind, step: str, exc: Exception, meta: dict[str, Any] | None = None) -> None:
        """Сохраняет стек и метаданные об исключении."""
        if not self.enabled:
            return
        session = self._ensure_session(kind)
        if session is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = sanitize_filename(f"{timestamp}_{step}")
        stack_path = session.folder("errors") / f"{prefix}_stacktrace.txt"
        with stack_path.open("w", encoding="utf-8") as fp:
            fp.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        meta_filename = f"{prefix}_meta.json"
        self.write_json(kind, "errors", meta_filename, {"step": step, "details": meta or {}})


rag_artifacts = RagArtifactsManager()

__all__ = [
    "rag_artifacts",
    "sanitize_filename",
    "serialize_documents",
    "combine_page_content",
    "normalize_retrieval_results",
    "build_context_blob",
]
