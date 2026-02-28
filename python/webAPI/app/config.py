
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    USER_LLM: Optional[str] = Field(default=None)
    PASSWORD_LLM: Optional[str] = Field(default=None)
    URL_LLM: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=BASE_DIR /   ".env.llm",
        env_file_encoding='utf-8'
        )

    def model_post_init(self, __context):
        """Подставляем значения из REMOTE_* если в .env.llm не заданы USER_LLM/PASSWORD_LLM/URL_LLM."""
        if not self.USER_LLM:
            object.__setattr__(self, "USER_LLM", self.REMOTE_AUTH_USER)
        if not self.PASSWORD_LLM:
            object.__setattr__(self, "PASSWORD_LLM", self.REMOTE_AUTH_PASSWORD)
        if not self.URL_LLM:
            object.__setattr__(self, "URL_LLM", self.REMOTE_LLM_URL)


settings = Settings()
settings_llm = LLM_Settings()
