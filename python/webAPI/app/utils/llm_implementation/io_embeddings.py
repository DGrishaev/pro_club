# имена классов должны начинаться с e_
# и отображать основные характеристики (по усмотрению разработчика)
import base64

from app.config import settings_llm


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

        model_name = "mxbai-embed-large"
        encoded_credentials = base64.b64encode(
            f"{settings_llm.USER_LLM}:{settings_llm.PASSWORD_LLM}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {encoded_credentials}"}

        return OllamaEmbeddings(
            model=model_name,
            base_url=settings_llm.URL_LLM,
            client_kwargs={"headers": headers},
        )
