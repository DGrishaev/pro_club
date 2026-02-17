# pro_club

Локальный dev-режим для Telegram + webAPI + RAG диагностики.

## Текущая архитектура (кратко)

- `python/telegram_bot/` — Telegram-бот, отправляет запросы в backend.
- `python/webAPI/` — backend с RAG-логикой.
  - endpoint: `POST /api/v1/llm/answer`
  - путь RAG: `routes/llm.py` → `services/llm_service.py` → `utils/io_db.py`
  - дальше: `io_separate_file.py` (парсинг/чанкинг) → `io_embeddings.py` (эмбеддинги) → `io_put_vector_in_db.py` + `io_get_vectror_db.py` (Chroma локально) → `io_search_from_db.py` (retrieval) → `io_promt.py` + LLM.

## Быстрый старт (локально)

1. Подготовить env:

```bash
cp python/webAPI/app/.env.example python/webAPI/app/.env
cp python/webAPI/app/.env.llm.example python/webAPI/app/.env.llm
cp .env.example .env
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Запуск backend:

```bash
make api
```

## Единые dev-entrypoints

- `make api` — запуск webAPI.
- `make rag-debug FILE=<путь_к_файлу> QUERY='<вопрос>'` — локальный прогон полного RAG по одному документу.

## Debug / observability для RAG

Через env:

- `RAG_DEBUG=true` — включает debug-логи пайплайна.
- `RAG_DEBUG_SAMPLE_CHUNKS=3` — сколько чанков показывать как sample.
- `RAG_RETRIEVAL_K=5` — top-k retrieval.
- `RAG_SCORE_THRESHOLD=0.4` — порог для `s_k_150`.

### Что логируется в debug mode

- количество чанков и статистика размеров;
- примеры чанков (текст + metadata);
- параметры retrieval и top-N результатов со score;
- число записей в vector store после индексации.

## Точечная диагностика проблемного документа

```bash
python tools/rag_debug.py --file ./path/to/doc.pdf --query "О чем документ?"
```

Скрипт выводит шаги:

1. parsing/chunking
2. embeddings init
3. indexing (локальный vector DB)
4. retrieval (top-k + score)
5. final answer

## Разделение инфраструктуры

- Vector DB хранится **локально** в `MAIN_FOLDER_PATH` (рекомендуется `./data/vectorstore`).
- Эмбеддинги и генерация могут идти на **удаленный GPU** через существующие `URL_LLM/USER_LLM/PASSWORD_LLM`.
- Секреты не логируются.
