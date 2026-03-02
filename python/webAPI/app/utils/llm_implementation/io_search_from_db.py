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

class k_five:
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
