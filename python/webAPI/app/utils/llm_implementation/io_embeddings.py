# имена классов должны начинаться с e_
# и отображать основные характеристики (по усмотрению разработчика)
import base64

try:
    from app.config import settings_llm  # type: ignore
except Exception:
    settings_llm = None


class e_default:
    def __init__(self):
        print("embeding")

    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "cointegrated/LaBSE-en-ru"

        hf_embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "cpu"}
        )

        return hf_embeddings_model


class e_multilingual_e5_large:
    def __init__(self):
        print("embeding")

    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "intfloat/multilingual-e5-large"

        hf_embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

        return hf_embeddings_model


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

        return OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            client_kwargs={"headers": headers},
        )
