# pro_club MVP

Команды бота:
- `/upload` — берёт все поддерживаемые файлы (`.pdf`, `.docx`, `.pptx`) из `LOCAL_RAG_UPLOAD_DIR` и индексирует их в RAG;
- `/status` — показывает список уже загруженных документов пользователя в Chroma;
- `/clear` — полностью очищает RAG-базу текущего пользователя.

Опциональные параметры ingestion через `.env`:
- `RAG_CHUNK_SIZE_CHARS` (по умолчанию `2400`) — размер текстового чанка;
- `RAG_CHUNK_OVERLAP_CHARS` (по умолчанию `300`) — overlap для текстовых чанков;
- `RAG_TABLE_WINDOW_THRESHOLD` (по умолчанию `50`) — порог числа строк таблицы для перехода в windowing;
- `RAG_TABLE_WINDOW_SIZE` (по умолчанию `20`) — размер окна строк в `table_rows_window`;
- `RAG_TABLE_SUMMARY_PREVIEW_ROWS` (по умолчанию `3`) — число preview-строк в `table_summary`;
- `RAG_TABLE_MAX_HEADER_ROWS` (по умолчанию `2`) — максимальная глубина шапки таблицы.
