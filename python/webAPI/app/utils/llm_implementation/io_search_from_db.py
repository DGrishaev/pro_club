# имена классов должны начинаться с s_
# и отображать основные характеристики (по усмотрению разработчика)


import os

from app.utils.rag_artifacts import (
    rag_artifacts,
    normalize_retrieval_results,
    combine_page_content,
)


def _k_from_env(default: int) -> int:
    try:
        return int(os.getenv("RAG_RETRIEVAL_K", str(default)))
    except ValueError:
        return default


def _bool_from_env(name: str, default: bool) -> bool:
    """Читает булево значение из env с безопасным значением по умолчанию."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_from_env(name: str, default: float) -> float:
    """Читает float из env без падения при невалидном значении."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _doc_to_payload(doc, score: float) -> dict:
    """Приводит документ из Chroma/LangChain к единому формату словаря."""
    return {
        "content": getattr(doc, "page_content", ""),
        "metadata": getattr(doc, "metadata", {}) or {},
        "score": score,
    }


def _normalize_scored_results(results) -> list[dict]:
    """Оставляет только элементы с валидным score и нормализует формат."""
    normalized = []
    for item in results or []:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        doc, score = item
        if score is None:
            continue
        normalized.append(_doc_to_payload(doc, score))
    return normalized


def _result_identity(item: dict) -> str:
    """Формирует ключ дедупликации по id/chunk_id/index и резервным полям."""
    metadata = item.get("metadata", {}) or {}
    for key in ("id", "chunk_id", "index", "chunk_index"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}:{value}"
    source = metadata.get("source")
    page = metadata.get("page")
    if source is not None or page is not None:
        return f"source:{source}|page:{page}"
    return f"content:{(item.get('content', '') or '')[:160]}"


def _is_better_score(new_score: float, old_score: float, score_is_distance: bool) -> bool:
    """Сравнивает score с учетом его природы: distance или similarity."""
    return new_score < old_score if score_is_distance else new_score > old_score


def _merge_results(existing: list[dict], incoming: list[dict], score_is_distance: bool) -> list[dict]:
    """Объединяет списки, убирает дубли и оставляет лучший score для каждого ключа."""
    by_key: dict[str, dict] = {}
    for item in (existing or []) + (incoming or []):
        key = _result_identity(item)
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            continue
        if _is_better_score(item["score"], current["score"], score_is_distance):
            by_key[key] = item
    return list(by_key.values())


def _sort_results(results: list[dict], score_is_distance: bool) -> list[dict]:
    """Сортирует результаты по score в корректном направлении."""
    return sorted(results, key=lambda x: x["score"], reverse=not score_is_distance)


def _passes_threshold(best_score: float | None, threshold: float, score_is_distance: bool) -> bool:
    """Проверяет, прошел ли лучший результат порог релевантности."""
    if best_score is None:
        return False
    return best_score <= threshold if score_is_distance else best_score >= threshold


def _short_snippet(text: str, limit: int = 120) -> str:
    """Готовит короткий сниппет для логов без длинных кусков текста."""
    compact = " ".join((text or "").split())
    return compact[:limit]


def _log_top_results(stage_name: str, results: list[dict], score_is_distance: bool, limit: int = 5) -> None:
    """Печатает только короткий top-N для объяснимого retrieval."""
    score_mode = "distance" if score_is_distance else "similarity"
    print(f"[RAG][{stage_name}] score_mode={score_mode}, results={len(results)}")
    for rank, item in enumerate(results[:limit], start=1):
        metadata = item.get("metadata", {}) or {}
        chunk_type = metadata.get("type", "unknown")
        chunk_id = metadata.get("id") or metadata.get("chunk_id") or metadata.get("index") or "n/a"
        snippet = _short_snippet(item.get("content", ""))
        print(f"[RAG][{stage_name}] rank={rank} score={item.get('score')} type={chunk_type} id={chunk_id} snippet={snippet}")


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

class s_k_five:
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

class s_hybrid_light:
    """Легкая двухстадийная стратегия поиска для кодовых/справочных запросов."""

    def __init__(self, prompt, user_name, vectordb, k: int = 5):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb
        self.k = k

    def _search_with_optional_filter(self, query: str, k_value: int, where_filter: dict | None = None):
        """Выполняет similarity_search_with_score с опциональным where-фильтром."""
        if where_filter is None:
            return self.vectordb.similarity_search_with_score(query, k=k_value)
        return self.vectordb.similarity_search_with_score(query, k=k_value, where=where_filter)

    def seach_from_db(self):
        score_is_distance = _bool_from_env("RAG_SCORE_IS_DISTANCE", True)
        score_threshold = _float_from_env("RAG_SCORE_THRESHOLD", 0.45)
        k_main = _k_from_env(self.k)
        k_table = min(3, k_main)
        k_fallback = min(7, k_main + 2)
        merged_results: list[dict] = []

        print(
            f"[RAG][s_hybrid_light] question={self.prompt!r} k={k_main} "
            f"score_threshold={score_threshold} score_mode={'distance' if score_is_distance else 'similarity'}"
        )
        try:
            # Стадия 1: сначала пробуем таблицы, это дешево и часто полезно для кодов/справочников.
            table_where = {"type": "table"}
            print(f"[RAG][s_hybrid_light] stage=table_first where={table_where} k={k_table}")
            try:
                stage1_raw = self._search_with_optional_filter(self.prompt, k_table, where_filter=table_where)
                stage1 = _normalize_scored_results(stage1_raw)
                merged_results = _merge_results(merged_results, stage1, score_is_distance)
                _log_top_results("table_first", _sort_results(stage1, score_is_distance), score_is_distance)
            except Exception as filter_exc:
                # Graceful degradation: если where не поддержан, продолжаем общий поиск.
                print(f"[RAG][s_hybrid_light] stage=table_first skipped reason={type(filter_exc).__name__}: {filter_exc}")

            # Стадия 2: обычный семантический поиск по всей базе.
            print(f"[RAG][s_hybrid_light] stage=semantic_main where=None k={k_main}")
            stage2_raw = self._search_with_optional_filter(self.prompt, k_main)
            stage2 = _normalize_scored_results(stage2_raw)
            merged_results = _merge_results(merged_results, stage2, score_is_distance)
            merged_results = _sort_results(merged_results, score_is_distance)
            _log_top_results("semantic_merged", merged_results, score_is_distance)

            best_score = merged_results[0]["score"] if merged_results else None
            if not _passes_threshold(best_score, score_threshold, score_is_distance):
                # Fallback: расширяем запрос без LLM и без полного скана коллекции.
                query_text = f"{self.prompt} код проекта код списать обучение ОБУЧ"
                print(f"[RAG][s_hybrid_light] stage=fallback_expand_query k={k_fallback} query={query_text!r}")
                fallback_raw = self._search_with_optional_filter(query_text, k_fallback)
                fallback_results = _normalize_scored_results(fallback_raw)
                merged_results = _merge_results(merged_results, fallback_results, score_is_distance)
                merged_results = _sort_results(merged_results, score_is_distance)
                _log_top_results("fallback_merged", merged_results, score_is_distance)
            else:
                print(f"[RAG][s_hybrid_light] quality_gate=passed best_score={best_score} threshold={score_threshold}")

            final_results = merged_results[:k_main]
            _log_search_artifacts(self.prompt, self.user_name, self.__class__.__name__, k_main, final_results)
            return final_results
        except Exception as exc:
            _log_search_error(
                "io_search_from_db.s_hybrid_light",
                exc,
                {"user_name": self.user_name, "k": k_main},
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
