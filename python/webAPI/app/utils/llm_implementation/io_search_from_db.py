# имена классов должны начинаться с s_
# и отображать основные характеристики (по усмотрению разработчика)

import logging
import os
from typing import Any, Iterable

from app.utils.rag_artifacts import (
    rag_artifacts,
    normalize_retrieval_results,
    combine_page_content,
)


logger = logging.getLogger("rag.search")


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Возвращает значение атрибута, не падая при отсутствии."""
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def _describe_embedding_provider(embedding_fn: Any) -> dict[str, Any]:
    """Формирует краткое описание используемой функции эмбеддингов."""
    if embedding_fn is None:
        return {"class": None, "model": None}
    model_name = (
        _safe_getattr(embedding_fn, "model_name")
        or _safe_getattr(embedding_fn, "model_name_or_path")
        or _safe_getattr(embedding_fn, "model_id")
    )
    return {
        "class": embedding_fn.__class__.__name__,
        "module": embedding_fn.__class__.__module__,
        "model": model_name,
    }


def _get_collection_name(vector_store: Any) -> str | None:
    """Пытается извлечь имя коллекции Chroma для логирования."""
    collection = _safe_getattr(vector_store, "_collection") or _safe_getattr(vector_store, "collection")
    if collection is None:
        return _safe_getattr(vector_store, "collection_name")
    name_attr = _safe_getattr(collection, "name")
    if callable(name_attr):
        try:
            return name_attr()
        except Exception:
            return None
    return name_attr


def _serialize_results(raw: Iterable[Any]) -> list[dict[str, Any]]:
    """Преобразует произвольный список результатов в словари с полями content/metadata/score."""
    serialized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        else:
            doc, score = item, None
        serialized.append(
            {
                "content": _safe_getattr(doc, "page_content", ""),
                "metadata": _safe_getattr(doc, "metadata", {}) or {},
                "score": score,
            }
        )
    return serialized


def _resolve_doc_identifier(metadata: dict[str, Any]) -> Any:
    """Пытается извлечь идентификатор чанка из метаданных."""
    for key in ("id", "doc_id", "document_id", "chunk_id", "chunk", "uuid"):
        if metadata.get(key) is not None:
            return metadata.get(key)
    return metadata.get("source")


def _ensure_query_text(prompt: str | None) -> str:
    """Проверяет пользовательский вопрос и подготавливает его к передаче в Chroma."""
    query = (prompt or "").strip()
    if not query:
        raise ValueError("Пустой запрос для поиска. Нужен текст вопроса пользователя.")
    return query


def _strip_paragraph_filter(value: Any) -> Any:
    """Удаляет условия фильтрации по type=paragraph, сохраняя остальные критерии."""
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for key, nested in value.items():
            if key == "type":
                if nested == "paragraph":
                    continue
                if isinstance(nested, dict) and nested.get("$eq") == "paragraph":
                    continue
            prepared = _strip_paragraph_filter(nested)
            if prepared is not None:
                cleaned[key] = prepared
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            prepared = _strip_paragraph_filter(item)
            if prepared is not None:
                cleaned_list.append(prepared)
        return cleaned_list or None
    return value


def _override_vectorstore_filter(vector_store: Any, new_filter: Any) -> None:
    """Пробует обновить search_kwargs внутри vectorstore, чтобы убрать устаревший фильтр."""
    for attr_name in ("search_kwargs", "_search_kwargs"):
        search_kwargs = _safe_getattr(vector_store, attr_name)
        if isinstance(search_kwargs, dict):
            if new_filter is None:
                search_kwargs.pop("filter", None)
                search_kwargs.pop("where", None)
            else:
                search_kwargs["filter"] = new_filter


def _extract_search_filters(vector_store: Any) -> tuple[Any, Any]:
    """Возвращает исходный и очищенный фильтры поиска."""
    raw_kwargs = _safe_getattr(vector_store, "search_kwargs") or _safe_getattr(vector_store, "_search_kwargs")
    raw_filter = None
    if isinstance(raw_kwargs, dict):
        raw_filter = raw_kwargs.get("filter") or raw_kwargs.get("where")
    prepared_filter = _strip_paragraph_filter(raw_filter)
    return raw_filter, prepared_filter


def _log_before_search(prompt: str, user_name: str, class_name: str, k_value: int, vectordb: Any, filters: dict[str, Any]) -> None:
    """Выводит диагностический лог перед выполнением запроса к Chroma."""
    embedding_fn = _safe_getattr(vectordb, "_embedding_function")
    embedding_meta = _describe_embedding_provider(embedding_fn)
    collection_name = _get_collection_name(vectordb)
    prompt_snippet = (prompt or "").replace("\n", " ")[:500]
    logger.info(
        "[RAG SEARCH] user=%s class=%s collection=%s k=%s filters=%s embedding=%s question=%s",
        user_name,
        class_name,
        collection_name,
        k_value,
        filters,
        embedding_meta,
        prompt_snippet,
    )


def _log_after_search(results: list[dict[str, Any]]) -> None:
    """Фиксирует статистику найденных документов и их ранжирование."""
    logger.info("[RAG SEARCH] найдено документов: %s", len(results))
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        snippet = (item.get("content") or "").replace("\n", " ")[:200]
        doc_id = _resolve_doc_identifier(metadata)
        logger.info(
            "[RAG SEARCH] #%s id=%s score=%s type=%s source=%s text=%s",
            idx,
            doc_id,
            item.get("score"),
            metadata.get("type"),
            metadata.get("source"),
            snippet,
        )


def _k_from_env(default: int) -> int:
    try:
        return int(os.getenv("RAG_RETRIEVAL_K", str(default)))
    except ValueError:
        return default


def _log_search_artifacts(prompt: str, user_name: str, class_name: str, k_value: int, results):
    """Сохраняет параметры запроса и найденные документы."""
    if not rag_artifacts.has_session("QU"):
        return
    rag_artifacts.write_json(
        "QU",
        "retrieval",
        "search_request.json",
        {
            "prompt": prompt,
            "user_name": user_name,
            "class_name": class_name,
            "k": k_value,
        },
    )
    normalized = normalize_retrieval_results(results or [])
    rag_artifacts.write_json("QU", "retrieval", "topk.json", normalized)
    joined_payload = [{"page_content": item.get("content", ""), "metadata": item.get("metadata", {})} for item in normalized]
    rag_artifacts.write_text("QU", "retrieval", "topk_joined.txt", combine_page_content(joined_payload))


def _log_search_error(step: str, exc: Exception, meta: dict | None = None) -> None:
    """Сохраняет стек ошибки и дополнительную информацию."""
    if rag_artifacts.has_session("QU"):
        rag_artifacts.log_exception("QU", step, exc, meta or {})


class s_default:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(4)
        try:
            results = self.vectordb.similarity_search(self.prompt, k=k)
            # фиксируем параметры поискового запроса
            _log_search_artifacts(self.prompt, self.user_name, self.__class__.__name__, k, results)
            return results
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_default",
                exc,
                {"user_name": self.user_name, "k": k},
            )
            raise

class k_five_backup:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(5)
        try:
            results = self.vectordb.similarity_search(self.prompt, k=k)
            _log_search_artifacts(self.prompt, self.user_name, self.__class__.__name__, k, results)
            return results
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_default",
                exc,
                {"user_name": self.user_name, "k": k},
            )
            raise

class s_k_five:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(5)
        query = _ensure_query_text(self.prompt)
        original_filter, cleaned_filter = _extract_search_filters(self.vectordb)
        _override_vectorstore_filter(self.vectordb, cleaned_filter)
        log_filters = {"raw": original_filter, "applied": cleaned_filter}
        try:
            _log_before_search(query, self.user_name, self.__class__.__name__, k, self.vectordb, log_filters)
            search_kwargs = {}
            if cleaned_filter is not None:
                search_kwargs["filter"] = cleaned_filter

            if hasattr(self.vectordb, "similarity_search_with_score"):
                raw_results = self.vectordb.similarity_search_with_score(query, k=k, **search_kwargs)
            else:
                logger.warning(
                    "[RAG SEARCH] %s не поддерживает similarity_search_with_score, продолжаю без расстояний",
                    self.vectordb.__class__.__name__,
                )
                docs = self.vectordb.similarity_search(query, k=k, **search_kwargs)
                raw_results = [(doc, None) for doc in docs]

            serialized = _serialize_results(raw_results)
            _log_after_search(serialized)
            _log_search_artifacts(query, self.user_name, self.__class__.__name__, k, serialized)
            return serialized
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_k_five",
                exc,
                {"user_name": self.user_name, "k": k},
            )
            raise


class s_k_2:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(2)
        try:
            results = self.vectordb.similarity_search(self.prompt, k=k)
            _log_search_artifacts(self.prompt, self.user_name, self.__class__.__name__, k, results)
            return results
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_k_2",
                exc,
                {"user_name": self.user_name, "k": k},
            )
            raise


class s_k_150:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))
        k = _k_from_env(150)
        try:
            results = self.vectordb.similarity_search_with_score(self.prompt, k=k)
            filtered_docs = []

            for doc, score in results:
                if score <= threshold:  # score = расстояние
                    filtered_docs.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": score,
                        }
                    )
            _log_search_artifacts(self.prompt, self.user_name, self.__class__.__name__, k, filtered_docs or results)
            return filtered_docs
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_k_150",
                exc,
                {"user_name": self.user_name, "k": k},
            )
            raise
