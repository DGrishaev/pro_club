
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):

    MAIN_FOLDER_PATH: str
    LOGS_FOLDER_PATH: str
    TELEGRAM_BOT_TOKEN: str
    CHROMA_PERSIST_DIR: str
    LOCAL_UPLOAD_DIR: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding='utf-8'
        )

class LLM_Settings(BaseSettings):
    CLASS_NAME_SEPARATE_FILE: str
    CLASS_NAME_EMBEDDINGS: str
    CLASS_NAME_PUT_VECTOR_IN_DB: str
    CLASS_NAME_GET_VECTOR_DB: str
    CLASS_NAME_SEARCH: str
    CLASS_NAME_SEARCH_SEARCH: str
    CLASS_NAME_PROMT: str
    REMOTE_LLM_MODEL: str
    REMOTE_LLM_URL: str
    REMOTE_EMBEDDINGS_URL: str
    REMOTE_AUTH_USER: str
    REMOTE_AUTH_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR /   ".env.llm",
        env_file_encoding='utf-8'
        )


settings = Settings()
settings_llm = LLM_Settings()
