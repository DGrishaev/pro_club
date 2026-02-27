# pro_club MVP

Минимальный локальный запуск: только Telegram-бот + RAG (локальная Chroma + удалённые embeddings/LLM).

1. `cp .env.example .env`
2. Заполните `.env` (токен Telegram, путь `CHROMA_PERSIST_DIR` к локальной внешней папке для ChromaDB, а также `LOCAL_RAG_UPLOAD_DIR` — путь к локальной папке с документами для команды `/upload`).
3. Заполните `.env.llm` (URL удалённого GPU/Ollama, логин/пароль, модель).
4. `pip install -r requirements.txt`
5. `python telegram_bot_mvp.py`

Что проверяется в MVP:
- бот отвечает на `/start` и принимает файл;
- файл сохраняется локально в `data/uploads/<telegram_user_id>/`;
- текст парсится и чанкуется;
- эмбеддинги считаются через удалённый endpoint;
- локальная Chroma обновляется в `CHROMA_PERSIST_DIR`;
- текстовый вопрос делает retrieval и возвращает ответ + debug (`top-k`).

FastAPI/webAPI в этом режиме не используется и не запускается.


MVP сохраняет старую логику классов из `python/webAPI/app/utils/llm_implementation/` и берёт имена классов из `LLM_IMPLEMENTATION_CONFIG` (по умолчанию `python/webAPI/app/utils/config.json`).


Команды бота:
- `/upload` — берёт все поддерживаемые файлы (`.pdf`, `.docx`, `.pptx`) из `LOCAL_RAG_UPLOAD_DIR` и индексирует их в RAG;
- `/status` — показывает список уже загруженных документов пользователя в Chroma;
- `/clear` — полностью очищает RAG-базу текущего пользователя.
