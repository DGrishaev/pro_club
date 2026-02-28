# имена классов должны начинаться с e_
# и отображать основные характеристики (по усмотрению разработчика)
import base64

try:
    from app.config import settings_llm  # type: ignore
except Exception:
    settings_llm = None
from app.utils.rag_artifacts import rag_artifacts


def _log_embedding_snapshot(class_name: str, model_name: str, extra: dict | None = None) -> None:
    """Сохраняет информацию о выбранной модели эмбеддингов."""
    if not rag_artifacts.has_session("UP"):
        return
    rag_artifacts.write_json(
        "UP",
        "embeddings",
        "model.json",
        {
            "embedding_class": class_name,
            "model_name": model_name,
        },
    )
    rag_artifacts.write_json(
        "UP",
        "embeddings",
        "embeddings_meta.json",
        {
            "extra": extra or {},
        },
    )


def _log_embedding_error(step: str, exc: Exception, meta: dict | None = None) -> None:
    """Логирует ошибки создания эмбеддингов."""
    if rag_artifacts.has_session("UP"):
        rag_artifacts.log_exception("UP", step, exc, meta or {})


class e_default:
    def __init__(self):
        print("embeding")

    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "cointegrated/LaBSE-en-ru"

        try:
            hf_embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": "cpu"}
            )
            # сохраняем сведения о выбранной модели эмбеддингов
            _log_embedding_snapshot(
                self.__class__.__name__,
                model_name,
                {"provider": "huggingface", "device": "cpu"},
            )
            return hf_embeddings_model
        except Exception as exc:
            _log_embedding_error("io_embeddings.e_default", exc, {"model_name": model_name})
            raise


class e_multilingual_e5_large:
    def __init__(self):
        print("embeding")

    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "intfloat/multilingual-e5-large"

        try:
            hf_embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
            )
            _log_embedding_snapshot(
                self.__class__.__name__,
                model_name,
                {"provider": "huggingface", "device": "cpu"},
            )
            return hf_embeddings_model
        except Exception as exc:
            _log_embedding_error("io_embeddings.e_multilingual_e5_large", exc, {"model_name": model_name})
            raise


class e_remote_ollama:
    """Embeddings via remote Ollama endpoint (GPU side)."""

    def __init__(self):
        pass

    def get_embeddings(self):
        from langchain_ollama import OllamaEmbeddings

        import os

        model_name = os.getenv("REMOTE_EMBEDDINGS_MODEL", "mxbai-embed-large")
        user = os.getenv("REMOTE_AUTH_USER", "")
        password = os.getenv("REMOTE_AUTH_PASSWORD", "")
        base_url = os.getenv("REMOTE_EMBEDDINGS_URL", "")

        if settings_llm is not None:
            user = user or getattr(settings_llm, "REMOTE_AUTH_USER", "")
            password = password or getattr(settings_llm, "REMOTE_AUTH_PASSWORD", "")
            base_url = base_url or getattr(settings_llm, "REMOTE_LLM_URL", "")

        encoded_credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
        headers = {"Authorization": f"Basic {encoded_credentials}"}

        try:
            instance = OllamaEmbeddings(
                model=model_name,
                base_url=base_url,
                client_kwargs={"headers": headers},
            )
            _log_embedding_snapshot(
                self.__class__.__name__,
                model_name,
                {"provider": "ollama", "base_url": base_url},
            )
            return instance
        except Exception as exc:
            _log_embedding_error(
                "io_embeddings.e_remote_ollama",
                exc,
                {"model_name": model_name, "base_url": base_url},
            )
            raise
