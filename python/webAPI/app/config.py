
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):

    APP_NAME: str
    DEBUG: bool  
    VERSION: str 
    DESCRIPTION: str
    USE_DB: bool
    DATABASE_URL: str
    SECRET_KEY: str 
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    GIGACHAT_TOKEN: str 

    # CORS (разрешенные источники)
    ALLOWED_ORIGINS: list[str]
    ALLOW_METHODS: list[str]
    ALLOW_HEADERS: list[str]
    ALLOW_CREDENTIALS: bool

    API_V1_STR: str

    URL_1C_ROOT: str
    URL_TELEBOT_ROOT: str
    UID_PROGRAM_1C: str
    UID_PROGRAM_TELEBOT: str
    CONNECTION_FILE_PATH: str

    MAIN_FOLDER_PATH: str
    LOGS_FOLDER_PATH: str

    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding='utf-8')

class LLM_Settings(BaseSettings):
    TASK_FOR_TEST: str
    CLASS_NAME_SEPARATE_FILE: str
    CLASS_NAME_EMBEDDINGS: str
    CLASS_NAME_PUT_VECTOR_IN_DB: str
    CLASS_NAME_GET_VECTOR_DB: str
    CLASS_NAME_SEARCH: str
    CLASS_NAME_SEARCH_SEARCH: str
    CLASS_NAME_PROMT: str
    MODEL: str
    URL_LLM: str
    USER_LLM: str
    PASSWORD_LLM: str

    model_config = SettingsConfigDict(env_file=".env.llm",
                                    env_file_encoding='utf-8')


settings = Settings()
settings_llm = LLM_Settings()
