# имена классов должны начинаться с gvid_
# и отображать основные характеристики (по усмотрению разработчика)
import os

from langchain_chroma import Chroma
from app.utils.rag_artifacts import rag_artifacts


def _log_db_meta(user_name: str, db_folder: str, class_name: str) -> None:
    """Сохраняет сведения об открываемой коллекции Chroma."""
    if rag_artifacts.has_session("UP"):
        rag_artifacts.write_json(
            "UP",
            "index",
            "get_db_meta.json",
            {
                "user_name": user_name,
                "persist_directory": db_folder,
                "collection_name": "main",
                "class_name": class_name,
            },
        )


def _log_db_error(step: str, exc: Exception, meta: dict | None = None) -> None:
    """Логирует ошибки при работе с Chroma."""
    if rag_artifacts.has_session("UP"):
        rag_artifacts.log_exception("UP", step, exc, meta or {})

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


class gvid_default:
    def __init__(self, user_name, embedding_function=None):
        self.user_name = user_name
        self.embedding_function = embedding_function

    def get_vectror_db(self):
        db_folder = _resolve_vectorstore_dir(self.user_name)
        os.makedirs(db_folder, exist_ok=True)

        if rag_artifacts.has_session("UP"):
            # фиксируем параметры открытия коллекции
            _log_db_meta(str(self.user_name), db_folder, self.__class__.__name__)

        try:
            if self.embedding_function is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self.embedding_function = HuggingFaceEmbeddings(
                    model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cpu"}
                )

            return Chroma(
                collection_name="main",
                persist_directory=db_folder,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            _log_db_error(
                "io_get_vectror_db.gvid_default",
                exc,
                {"user_name": str(self.user_name), "persist_directory": db_folder},
            )
            raise


class gvid_multilingual_e5_large(gvid_default):
    def get_vectror_db(self):
        db_folder = _resolve_vectorstore_dir(self.user_name)
        os.makedirs(db_folder, exist_ok=True)

        if rag_artifacts.has_session("UP"):
            # фиксируем параметры открытия коллекции
            _log_db_meta(str(self.user_name), db_folder, self.__class__.__name__)

        try:
            if self.embedding_function is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self.embedding_function = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "cpu"}
                )

            return Chroma(
                collection_name="main",
                persist_directory=db_folder,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            _log_db_error(
                "io_get_vectror_db.gvid_multilingual_e5_large",
                exc,
                {"user_name": str(self.user_name), "persist_directory": db_folder},
            )
            raise



class gvid_multilingual_e5_large_cosine(gvid_default):
    def get_vectror_db(self):
        db_folder = _resolve_vectorstore_dir(self.user_name)
        os.makedirs(db_folder, exist_ok=True)

        if rag_artifacts.has_session("UP"):
            # фиксируем параметры открытия коллекции
            _log_db_meta(str(self.user_name), db_folder, self.__class__.__name__)

        try:
            if self.embedding_function is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self.embedding_function = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "cpu"}
                )

            return Chroma(
                collection_name="main",
                persist_directory=db_folder,
                embedding_function=self.embedding_function,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            _log_db_error(
                "io_get_vectror_db.gvid_multilingual_e5_large_cosine",
                exc,
                {"user_name": str(self.user_name), "persist_directory": db_folder},
            )
            raise
