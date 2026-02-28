# имена классов должны начинаться с pvid_
# и отображать основные характеристики (по усмотрению разработчика)

import os

from langchain_chroma import Chroma
from app.utils.rag_artifacts import rag_artifacts, serialize_documents

try:
    from app.config import settings  # type: ignore
except Exception:
    settings = None


def _resolve_vectorstore_dir(user_name) -> str:
    user_folder_name = user_name if isinstance(user_name, str) else user_name.name
    safe_name = user_folder_name.replace("/", "_").replace("\\", "_")
    root = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    if settings is not None:
        root = os.getenv("CHROMA_PERSIST_DIR", getattr(settings, "MAIN_FOLDER_PATH", root))
    return os.path.join(root, safe_name, "db")


class pvid_default:
    def __init__(self, separate_text, embedding, user_name):
        self.separate_text = separate_text
        self.embedding = embedding
        self.user_name = user_name

    def put_vector_in_db(self):
        if not self.separate_text:
            return False

        db_folder = _resolve_vectorstore_dir(self.user_name)
        os.makedirs(db_folder, exist_ok=True)

        diag_enabled = rag_artifacts.has_session("UP")
        # сохраняем входные данные индексации только в диагностическом режиме
        count_before = None
        if diag_enabled:
            try:
                probe = Chroma(
                    collection_name="main",
                    persist_directory=db_folder,
                    embedding_function=self.embedding,
                )
                existing = probe.get()
                count_before = len(existing.get("ids", []))
            except Exception as exc:
                rag_artifacts.log_exception(
                    "UP",
                    "io_put_vector_in_db.count_before",
                    exc,
                    {"user_name": str(self.user_name)},
                )
            rag_artifacts.write_json(
                "UP",
                "index",
                "put_request.json",
                {
                    "user_name": str(self.user_name),
                    "documents": serialize_documents(self.separate_text),
                },
            )

        try:
            store = Chroma.from_documents(
                collection_name="main",
                documents=self.separate_text,
                embedding=self.embedding,
                persist_directory=db_folder,
            )
            if diag_enabled:
                try:
                    all_docs = store.get()
                    count_after = len(all_docs.get("ids", []))
                except Exception:
                    count_after = None
                # сохраняем итоговую статистику хранилища
                rag_artifacts.write_json(
                    "UP",
                    "index",
                    "db_state.json",
                    {
                        "collection_name": "main",
                        "persist_directory": db_folder,
                        "count_before": count_before,
                        "count_after": count_after,
                    },
                )
            return True
        except Exception as exc:
            if diag_enabled:
                rag_artifacts.log_exception(
                    "UP",
                    "io_put_vector_in_db.put_vector_in_db",
                    exc,
                    {"user_name": str(self.user_name)},
                )
            raise
